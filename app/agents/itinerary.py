from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings

logger = structlog.get_logger()

PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "itinerary.txt").read_text()


def _to_date(val) -> date | None:
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        try:
            return datetime.strptime(val, "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _get(obj, key: str, default=None):
    """Get attribute from Pydantic model or key from dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _format_flights(flights: list) -> str:
    if not flights:
        return "No flight options were found."
    lines: list[str] = []
    for i, f in enumerate(flights, 1):
        airline = _get(f, "airline", "") or ""
        fnum = _get(f, "flight_number", "") or ""
        dep = _get(f, "departure_time", "") or ""
        arr = _get(f, "arrival_time", "") or ""
        dur = _get(f, "duration_minutes", "")
        stops = _get(f, "stops", "")
        price = _get(f, "price", "")
        currency = _get(f, "currency", "") or ""
        url = _get(f, "booking_url")
        line = (
            f"{i}. **{airline}** {fnum}\n"
            f"   Departure: {dep} → Arrival: {arr} ({dur} min, {stops} stop(s))\n"
            f"   Price: {currency} {price}"
        )
        if url:
            line += f" | [Book]({url})"
        lines.append(line)
    return "\n".join(lines)


def _format_hotels(hotels: list) -> str:
    if not hotels:
        return "No hotel options were found."
    lines: list[str] = []
    for i, h in enumerate(hotels, 1):
        name = _get(h, "name", "") or ""
        loc = _get(h, "location", "") or ""
        ppn = _get(h, "price_per_night", "")
        total = _get(h, "total_price", "")
        rating = _get(h, "rating")
        amenities = _get(h, "amenities", []) or []
        url = _get(h, "booking_url")
        rating_str = f" | Rating: {rating}/5" if rating else ""
        amenities_str = f" | Amenities: {', '.join(amenities)}" if amenities else ""
        line = (
            f"{i}. **{name}** — {loc}\n"
            f"   ₹{ppn}/night (total ₹{total}){rating_str}{amenities_str}"
        )
        if url:
            line += f"\n   [Book]({url})"
        lines.append(line)
    return "\n".join(lines)


def _format_research(research: list) -> str:
    if not research:
        return "No research data available."
    lines: list[str] = []
    for r in research:
        cat = _get(r, "category", "") or ""
        title = _get(r, "title", "") or ""
        summary = _get(r, "summary", "") or ""
        url = _get(r, "source_url")
        cite = f" [Source]({url})" if url else ""
        lines.append(f"- **[{cat}] {title}:** {summary}{cite}")
    return "\n".join(lines)


def _format_transport(transport: list) -> str:
    if not transport:
        return "No transport data available."
    lines: list[str] = []
    for t in transport:
        mode = _get(t, "mode", "") or ""
        desc = _get(t, "description", "") or ""
        cost = _get(t, "estimated_cost")
        avail = _get(t, "availability_notes", "") or ""
        url = _get(t, "source_url")
        cost_str = f" — {cost}" if cost else ""
        avail_str = f" ({avail})" if avail else ""
        cite = f" [Source]({url})" if url else ""
        lines.append(f"- **{mode}:** {desc}{cost_str}{avail_str}{cite}")
    return "\n".join(lines)


def _format_errors(errors: list[str]) -> str:
    if not errors:
        return "None."
    return "\n".join(f"- {e}" for e in errors)


def _extract_content(response) -> str:
    content = response.content
    if isinstance(content, list):
        if not content:
            return ""
        first = content[0]
        if isinstance(first, str):
            return "\n".join(content)
        if isinstance(first, dict) and "text" in first:
            return "".join(part.get("text", "") for part in content)
        return str(content)
    if isinstance(content, (bytes, bytearray)):
        return content.decode()
    return str(content) if content else ""


async def itinerary_agent_node(state: dict) -> dict:
    user_input = state["user_input"]
    budget = state.get("budget")

    destination = user_input.get("to_destination", "")
    dep_date = _to_date(user_input.get("departure_date"))
    ret_date = _to_date(user_input.get("return_date"))
    num_days = (ret_date - dep_date).days if dep_date and ret_date else 1

    logger.info("itinerary_agent_started", destination=destination, num_days=num_days)

    try:
        flights = state.get("ranked_flights") or state.get("flights") or []
        hotels = state.get("ranked_hotels") or state.get("hotels") or []
        research = state.get("research") or []
        transport = state.get("transport") or []
        errors = state.get("errors") or []

        budget_str = "Not available"
        if budget:
            b = budget
            budget_str = (
                f"Total: ₹{_get(b, 'total_budget', '')}\n"
                f"  - Flights: ₹{_get(b, 'flight_budget', '')}\n"
                f"  - Hotels: ₹{_get(b, 'hotel_budget', '')}\n"
                f"  - Transport: ₹{_get(b, 'transport_budget', '')}\n"
                f"  - Activities: ₹{_get(b, 'activities_budget', '')}"
            )

        trip_style = ", ".join(user_input.get("trip_style", [])) or "General"
        notes = user_input.get("notes") or "None"

        user_message = f"""## Trip Details
- From: {user_input.get('from_city', '')} → To: {destination}
- Dates: {dep_date} to {ret_date} ({num_days} days)
- Travelers: {user_input.get('adults', 1)} adults, {user_input.get('children', 0)} children
- Budget: ₹{user_input.get('budget_min', '')} - ₹{user_input.get('budget_max', '')}
- Style: {trip_style}
- Notes: {notes}

## Budget Allocation
{budget_str}

## Available Flights
{_format_flights(flights)}

## Available Hotels
{_format_hotels(hotels)}

## Destination Research
{_format_research(research)}

## Transport Options
{_format_transport(transport)}

## Errors / Context
{_format_errors(errors)}

Now create the complete day-by-day itinerary."""

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            google_api_key=settings.google_api_key,
        )

        response = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])

        itinerary = _extract_content(response).strip()

        logger.info(
            "itinerary_agent_completed",
            length=len(itinerary),
        )

        return {"itinerary": itinerary}

    except Exception as e:
        logger.error("itinerary_agent_failed", error=str(e), exc_info=True)
        return {"itinerary": "", "errors": [f"Itinerary agent: {e}"]}
