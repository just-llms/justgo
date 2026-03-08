from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime

import structlog
from amadeus import Client as AmadeusClient
from forex_python.converter import CurrencyRates
from langchain_community.agent_toolkits.amadeus.toolkit import AmadeusToolkit
from langchain_community.tools.amadeus.closest_airport import AmadeusClosestAirport
from langchain_community.tools.amadeus.flight_search import AmadeusFlightSearch
from langchain_community.tools.amadeus.base import AmadeusBaseTool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from api.schemas import FlightOption
from config import settings

logger = structlog.get_logger()

NEAREST_AIRPORTS_PROMPT = """List the 3 nearest airports to the given location, in order of proximity (closest first).
Return ONLY a JSON array of 3-letter IATA codes, nothing else. Example: ["BLR", "MYS", "COK"].
If there are fewer than 3 airports, return that many. Use only valid IATA codes."""

# Resolve forward refs to amadeus.Client (TYPE_CHECKING-only in langchain_community Amadeus code)
_client_ns = {"Client": AmadeusClient}
AmadeusBaseTool.model_rebuild(_types_namespace=_client_ns)
AmadeusClosestAirport.model_rebuild(_types_namespace=_client_ns)
AmadeusFlightSearch.model_rebuild(_types_namespace=_client_ns)
AmadeusToolkit.model_rebuild(_types_namespace=_client_ns)

# forex_python uses ISO 4217 codes (EUR, USD). Amadeus may return "EURO".
_CURRENCY_ALIASES = {"EURO": "EUR"}


def _convert_to_inr(amount: float, from_currency: str) -> tuple[float, str]:
    """Convert amount to INR. Returns (amount_inr, 'INR') or (original, original_currency) on failure."""
    if not from_currency or amount is None:
        return (amount or 0.0, from_currency or "INR")
    code = _CURRENCY_ALIASES.get(from_currency.upper(), from_currency.upper())[:3]
    if code == "INR":
        return (amount, "INR")
    try:
        c = CurrencyRates()
        rate = c.get_rate(code, "INR")
        return (round(amount * rate, 2), "INR")
    except Exception as e:
        logger.warning("currency_conversion_skipped", from_currency=code, error=str(e))
        return (amount, from_currency)


def _convert_flights_to_inr(flights: list[FlightOption]) -> list[FlightOption]:
    """Return a new list of FlightOptions with price and currency converted to INR."""
    result: list[FlightOption] = []
    for f in flights:
        price_inr, currency_inr = _convert_to_inr(f.price, f.currency)
        result.append(
            FlightOption(
                airline=f.airline,
                flight_number=f.flight_number,
                departure_time=f.departure_time,
                arrival_time=f.arrival_time,
                duration_minutes=f.duration_minutes,
                stops=f.stops,
                price=price_inr,
                currency=currency_inr,
                booking_url=f.booking_url,
                source=f.source,
            )
        )
    return result


def _get_toolkit() -> AmadeusToolkit:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=settings.google_api_key,
    )
    return AmadeusToolkit(llm=llm)


def _parse_iata(response) -> str | None:
    """Extract a 3-letter IATA code from the closest_airport tool response."""
    text = str(response.content if hasattr(response, "content") else response)
    match = re.search(r'"iataCode"\s*:\s*"([A-Z]{3})"', text)
    if match:
        return match.group(1)
    match = re.search(r"\b([A-Z]{3})\b", text)
    if match:
        return match.group(1)
    return None


def _parse_iata_list(text: str) -> list[str]:
    """Extract up to 3 IATA codes from Gemini response (JSON array of strings)."""
    codes: list[str] = []
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            for x in parsed[:3]:
                if isinstance(x, str) and len(x) == 3 and x.isalpha():
                    codes.append(x.upper())
    except json.JSONDecodeError:
        for m in re.findall(r"\b([A-Z]{3})\b", text.upper()):
            if m not in codes:
                codes.append(m)
                if len(codes) >= 3:
                    break
    return codes[:3]


async def _get_nearest_airports_llm(location: str, max_count: int = 3) -> list[str]:
    """Return up to max_count nearest airport IATA codes for location (order of proximity)."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=settings.google_api_key,
    )
    response = await llm.ainvoke([
        SystemMessage(content=NEAREST_AIRPORTS_PROMPT),
        HumanMessage(content=f"Location: {location}"),
    ])
    content = response.content
    if isinstance(content, list):
        content = "".join(
            p.get("text", p) if isinstance(p, dict) else str(p) for p in content
        )
    text = str(content) if content else ""
    return _parse_iata_list(text)


def _map_offer_to_flight(offer: dict) -> FlightOption | None:
    """Convert a single Amadeus offer dict to a FlightOption."""
    try:
        segments = offer.get("segments", [])
        if not segments:
            return None

        price_total = float(offer["price"]["total"])
        currency = offer["price"].get("currency", "EUR")
        if isinstance(currency, dict):
            currency = "EUR"

        carriers = []
        flight_numbers = []
        for seg in segments:
            carriers.append(seg.get("carrier", "Unknown"))
            flight_numbers.append(seg.get("flightNumber", ""))

        airline = " / ".join(dict.fromkeys(carriers))
        flight_number = " / ".join(fn for fn in flight_numbers if fn)

        dep_time_str = segments[0]["departure"]["at"]
        arr_time_str = segments[-1]["arrival"]["at"]

        dep_dt = datetime.strptime(dep_time_str, "%Y-%m-%dT%H:%M:%S")
        arr_dt = datetime.strptime(arr_time_str, "%Y-%m-%dT%H:%M:%S")
        duration_minutes = int((arr_dt - dep_dt).total_seconds() / 60)

        stops = len(segments) - 1

        return FlightOption(
            airline=airline,
            flight_number=flight_number,
            departure_time=dep_time_str,
            arrival_time=arr_time_str,
            duration_minutes=max(duration_minutes, 0),
            stops=stops,
            price=price_total,
            currency=currency,
            booking_url=None,
            source="Amadeus",
        )
    except Exception:
        return None


def _score_and_rank(flights: list[FlightOption], top_n: int = 3) -> list[FlightOption]:
    """Apply composite scoring from TECH_SPEC and return top N."""
    if not flights:
        return []

    prices = [f.price for f in flights]
    durations = [f.duration_minutes for f in flights]

    min_price, max_price = min(prices), max(prices)
    min_dur, max_dur = min(durations), max(durations)

    price_range = max_price - min_price if max_price != min_price else 1.0
    dur_range = max_dur - min_dur if max_dur != min_dur else 1.0

    scored: list[tuple[float, FlightOption]] = []
    for f in flights:
        norm_price = (f.price - min_price) / price_range
        norm_dur = (f.duration_minutes - min_dur) / dur_range
        stop_penalty = f.stops * 0.5
        score = 0.5 * norm_price + 0.3 * norm_dur + 0.2 * stop_penalty
        scored.append((score, f))

    scored.sort(key=lambda x: x[0])
    return [f for _, f in scored[:top_n]]


async def flights_agent_node(state: dict) -> dict:
    user_input = state["user_input"]
    budget = state["budget"]
    flight_budget = budget.flight_budget if hasattr(budget, "flight_budget") else budget.get("flight_budget", 0)

    from_city = user_input["from_city"]
    destination = user_input["to_destination"]
    departure_date = str(user_input["departure_date"])

    logger.info(
        "flights_agent_started",
        from_city=from_city,
        destination=destination,
        flight_budget=flight_budget,
    )

    try:
        toolkit = await asyncio.to_thread(_get_toolkit)
        tools = toolkit.get_tools()

        airport_tool = None
        search_tool = None
        for t in tools:
            if t.name == "closest_airport":
                airport_tool = t
            elif t.name == "single_flight_search":
                search_tool = t

        if not airport_tool or not search_tool:
            logger.error("flights_missing_tools", available=[t.name for t in tools])
            return {"flights": [], "errors": ["Flights: Amadeus toolkit missing required tools"]}

        # Single origin: closest airport to from_city
        origin_resp = await asyncio.to_thread(
            airport_tool.run, {"location": from_city}
        )
        origin_iata = _parse_iata(origin_resp)
        if not origin_iata:
            logger.error("flights_origin_resolution_failed", origin_raw=str(origin_resp)[:200])
            return {"flights": [], "errors": ["Flights: could not resolve origin airport"]}

        # Up to 3 nearest destination airports (try in order until we get flights)
        dest_candidates = await _get_nearest_airports_llm(destination, max_count=3)
        if not dest_candidates:
            logger.error("flights_no_dest_airports", destination=destination)
            return {"flights": [], "errors": ["Flights: could not resolve destination airport(s)"]}

        logger.info(
            "flights_dest_candidates",
            origin_iata=origin_iata,
            dest_iata_list=dest_candidates,
        )

        search_params = {
            "originLocationCode": origin_iata,
            "departureDateTimeEarliest": f"{departure_date}T00:00:00",
            "departureDateTimeLatest": f"{departure_date}T23:59:59",
        }

        all_options: list[FlightOption] = []
        used_dest: str | None = None

        for attempt, dest_iata in enumerate(dest_candidates, 1):
            search_results = await asyncio.to_thread(
                search_tool.run,
                {"destinationLocationCode": dest_iata, **search_params},
            )

            if isinstance(search_results, str):
                try:
                    search_results = json.loads(search_results)
                except json.JSONDecodeError:
                    search_results = []

            if not search_results or search_results == [None]:
                logger.info(
                    "flights_no_results_for_dest",
                    attempt=attempt,
                    origin_iata=origin_iata,
                    dest_iata=dest_iata,
                )
                continue

            for offer in search_results:
                if offer is None:
                    continue
                option = _map_offer_to_flight(offer)
                if option:
                    all_options.append(option)

            if all_options:
                used_dest = dest_iata
                logger.info(
                    "flights_found_on_attempt",
                    attempt=attempt,
                    dest_iata=dest_iata,
                    count=len(all_options),
                )
                break

        if not all_options:
            logger.warning(
                "flights_no_results_after_3_dests",
                origin_iata=origin_iata,
                tried_dests=dest_candidates,
            )
            error_detail = (
                "Flights: No flights found. We searched from origin "
                f"{origin_iata} ({from_city}) to these destination airports: "
                f"{', '.join(dest_candidates)} ({destination}). "
                "None had available flights for the requested date."
            )
            return {"flights": [], "errors": [error_detail]}

        all_options = await asyncio.to_thread(_convert_flights_to_inr, all_options)
        top_flights = _score_and_rank(all_options, top_n=3)

        logger.info(
            "flights_agent_completed",
            origin_iata=origin_iata,
            dest_used=used_dest,
            total_parsed=len(all_options),
            returned=len(top_flights),
        )

        return {"flights": top_flights}

    except Exception as e:
        logger.error("flights_agent_failed", error=str(e), exc_info=True)
        return {"flights": [], "errors": [f"Flights agent: {e}"]}
