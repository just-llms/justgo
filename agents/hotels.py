from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from api.schemas import HotelOption
from config import settings
from tools.airbnb import airbnb_search
from tools.tavily import tavily_search

logger = structlog.get_logger()

PROMPTS_DIR = Path(__file__).parent / "prompts"
SEARCH_PAYLOAD_PROMPT = (PROMPTS_DIR / "hotels.txt").read_text()

TAVILY_STRUCTURING_PROMPT = """You are a travel accommodation analyst. You will receive raw web search results about hotels/stays at a destination.

Extract and structure the information into a JSON array of objects, each with these fields:
- "name": property/hotel name
- "location": location description
- "price_per_night": estimated price per night in INR (number)
- "total_price": estimated total price for the stay in INR (number)
- "rating": rating out of 5 (number or null)
- "amenities": list of amenity strings (can be empty)
- "booking_url": URL to book or view the listing (null if unavailable)

Rules:
- Return ONLY valid JSON, no markdown fences or extra text.
- Include at most 5 results.
- Derive total_price from price_per_night × num_nights if only per-night price is available.
- If only a total price is mentioned, derive price_per_night = total_price / num_nights.
"""


def _normalize_gemini_content(content) -> str:
    """Convert Gemini response.content (str, list, bytes) to a plain string."""
    if isinstance(content, list):
        if not content:
            return "{}"
        first = content[0]
        if isinstance(first, str):
            return "\n".join(content)
        if isinstance(first, dict) and "text" in first:
            return "".join(part.get("text", "") for part in content)
        return str(content)
    if isinstance(content, (bytes, bytearray)):
        return content.decode()
    return content


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return text


def _parse_rating(label: str | None) -> float | None:
    """Extract numeric rating from e.g. '4.93 out of 5 average rating, 121 reviews'."""
    if not label:
        return None
    m = re.search(r"([\d.]+)\s+out\s+of\s+5", label)
    return float(m.group(1)) if m else None


def _parse_price(display_price: dict | None, num_nights: int) -> tuple[float, float]:
    """Extract (price_per_night, total_price) from Airbnb structuredDisplayPrice.

    Returns (0.0, 0.0) when parsing fails — caller should skip the listing.
    """
    if not display_price:
        return 0.0, 0.0

    details_str = ""
    explanation = display_price.get("explanationData") or {}
    details_str = explanation.get("priceDetails", "")

    primary = display_price.get("primaryLine") or {}
    primary_label = primary.get("accessibilityLabel", "")

    total = 0.0
    per_night = 0.0

    # Try "₹23,075 for 5 nights" from primaryLine
    total_match = re.search(r"[\₹Rs.]*\s*([\d,]+(?:\.\d+)?)\s+for\s+\d+\s+night", primary_label)
    if total_match:
        total = float(total_match.group(1).replace(",", ""))

    # Try "5 nights x ₹5,625.92: ₹28,129.6" from priceDetails
    if not total and details_str:
        per_night_match = re.search(r"(\d+)\s*nights?\s*x\s*[\₹Rs.]*\s*([\d,]+(?:\.\d+)?)", details_str)
        if per_night_match:
            n = int(per_night_match.group(1))
            pn = float(per_night_match.group(2).replace(",", ""))
            per_night = pn
            total = pn * n

    if total and not per_night and num_nights > 0:
        per_night = total / num_nights
    if per_night and not total and num_nights > 0:
        total = per_night * num_nights

    return per_night, total


def _map_listing(listing: dict, destination: str, num_nights: int) -> HotelOption | None:
    """Convert a single Airbnb search result to a HotelOption, or None if unparseable."""
    try:
        demand = listing.get("demandStayListing") or {}
        desc = demand.get("description") or {}
        name_obj = desc.get("name") or {}
        name = name_obj.get("localizedStringWithTranslationPreference", "")
        if not name:
            return None

        rating = _parse_rating(listing.get("avgRatingA11yLabel"))
        per_night, total = _parse_price(listing.get("structuredDisplayPrice"), num_nights)
        if total <= 0:
            return None

        structured = listing.get("structuredContent") or {}
        amenities = []
        primary_line = structured.get("primaryLine", "")
        if primary_line:
            amenities = [primary_line]

        return HotelOption(
            name=name,
            location=destination,
            price_per_night=round(per_night, 2),
            total_price=round(total, 2),
            rating=rating,
            amenities=amenities,
            booking_url=listing.get("url"),
            source="Airbnb",
        )
    except Exception:
        return None


async def _generate_search_payload(
    user_input: dict, hotel_budget: int, num_nights: int
) -> dict:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=settings.google_api_key,
    )

    dep = user_input["departure_date"]
    ret = user_input["return_date"]
    if isinstance(dep, date):
        dep = dep.isoformat()
    if isinstance(ret, date):
        ret = ret.isoformat()

    user_msg = (
        f"Destination: {user_input['to_destination']}\n"
        f"Check-in: {dep}\n"
        f"Check-out: {ret}\n"
        f"Number of nights: {num_nights}\n"
        f"Adults: {user_input['adults']}\n"
        f"Children: {user_input.get('children', 0)}\n"
        f"Hotel budget (total): {hotel_budget} INR\n"
        f"Trip style: {', '.join(user_input.get('trip_style', []))}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=SEARCH_PAYLOAD_PROMPT),
        HumanMessage(content=user_msg),
    ])

    content = response.content
    if isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict):
        return content[0]
    content = _normalize_gemini_content(content)
    content = _strip_fences(content)
    return json.loads(content)


async def _search_via_tavily(
    user_input: dict, hotel_budget: int, num_nights: int
) -> list[HotelOption]:
    """Fallback: search Tavily for hotel results and use Gemini to structure them."""
    destination = user_input["to_destination"]
    per_night = hotel_budget // max(num_nights, 1)

    queries = [
        f"best hotels in {destination} under ₹{per_night} per night",
        f"top rated homestays and Airbnb in {destination}",
    ]

    raw_results: list[dict] = []
    for query in queries:
        try:
            result = await tavily_search(query)
            if isinstance(result, str):
                result = json.loads(result)
            raw_results.append({"query": query, "result": result})
        except Exception as exc:
            logger.warning("hotels_tavily_query_failed", query=query, error=str(exc))

    if not raw_results:
        return []

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=settings.google_api_key,
    )

    user_msg = (
        f"Destination: {destination}\n"
        f"Number of nights: {num_nights}\n"
        f"Budget (total): {hotel_budget} INR\n\n"
        f"Raw search results:\n{json.dumps(raw_results, indent=2, default=str)}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=TAVILY_STRUCTURING_PROMPT),
        HumanMessage(content=user_msg),
    ])

    content = _normalize_gemini_content(response.content)
    content = _strip_fences(content)
    parsed = json.loads(content)
    if not isinstance(parsed, list):
        parsed = []

    options = []
    for item in parsed:
        try:
            item.setdefault("source", "Tavily")
            options.append(HotelOption(**item))
        except Exception:
            continue
    return options


async def hotels_agent_node(state: dict) -> dict:
    user_input = state["user_input"]
    budget = state["budget"]
    hotel_budget = budget.hotel_budget if hasattr(budget, "hotel_budget") else budget.get("hotel_budget", 0)
    destination = user_input["to_destination"]

    dep = user_input["departure_date"]
    ret = user_input["return_date"]
    if isinstance(dep, str):
        dep = date.fromisoformat(dep)
    if isinstance(ret, str):
        ret = date.fromisoformat(ret)
    num_nights = max((ret - dep).days, 1)

    try:
        logger.info("hotels_agent_started", destination=destination, hotel_budget=hotel_budget)

        # Step 1 — Gemini: build search payload
        payload = await _generate_search_payload(user_input, hotel_budget, num_nights)
        logger.info("hotels_search_payload", payload=payload)

        # Step 2 — MCP Airbnb search
        mcp_result = await airbnb_search(
            location=payload.get("location", destination),
            check_in=payload.get("checkIn", dep.isoformat()),
            check_out=payload.get("checkOut", ret.isoformat()),
            adults=payload.get("adults", user_input["adults"]),
            children=payload.get("children", user_input.get("children", 0)),
            min_price=payload.get("minPrice"),
            max_price=payload.get("maxPrice"),
        )

        options: list[HotelOption] = []

        if mcp_result and mcp_result.get("searchResults"):
            listings = mcp_result["searchResults"]
            logger.info("airbnb_mcp_results", count=len(listings))

            for listing in listings:
                opt = _map_listing(listing, destination, num_nights)
                if opt is not None:
                    options.append(opt)

            # Filter: price within budget, rating >= 4.0
            options = [
                o for o in options
                if o.total_price <= hotel_budget
                and (o.rating is None or o.rating >= 4.0)
            ]
        else:
            logger.info("hotels_agent_fallback_tavily", reason="MCP returned no results")
            options = await _search_via_tavily(user_input, hotel_budget, num_nights)
            options = [
                o for o in options
                if o.total_price <= hotel_budget
                and (o.rating is None or o.rating >= 4.0)
            ]

        # Sort: rating desc (None last), then price asc
        options.sort(key=lambda o: (-(o.rating or 0), o.total_price))
        top = options[:3]

        logger.info("hotels_agent_completed", count=len(top))
        return {"hotels": top}

    except Exception as exc:
        logger.error("hotels_agent_failed", error=str(exc), exc_info=True)
        return {"hotels": [], "errors": [f"Hotels agent: {exc}"]}
