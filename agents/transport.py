from __future__ import annotations

import json
from pathlib import Path

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from api.schemas import TransportOption
from config import settings
from tools.tavily import tavily_search

logger = structlog.get_logger()

PROMPTS_DIR = Path(__file__).parent / "prompts"
QUERY_SYSTEM_PROMPT = (PROMPTS_DIR / "transport.txt").read_text()

STRUCTURING_PROMPT = """You are a local transport analyst. You will receive raw web search results about transportation options at a travel destination.

Extract and structure the information into a JSON array of objects, each with these fields:
- "mode": transport mode (e.g. "Taxi", "Local bus", "Bike rental", "Auto rickshaw")
- "description": short description of the option and how to use it
- "estimated_cost": approximate cost as a string (e.g. "₹50 per km") or null if unknown
- "availability_notes": when/where available (e.g. "24/7", "Near mall road") or empty string
- "source_url": URL where this information came from (null if unavailable)

Rules:
- Return ONLY valid JSON, no markdown fences or extra text.
- Deduplicate by mode/description. Include at most 8-10 options.
- Prefer options with cost information when available.
"""


async def _generate_queries(user_input: dict) -> list[str]:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=settings.google_api_key,
    )

    user_msg = (
        f"Destination: {user_input['to_destination']}\n"
        f"Trip style: {', '.join(user_input.get('trip_style', []))}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=QUERY_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    content = response.content
    if isinstance(content, list):
        if not content:
            return []
        first = content[0]
        if isinstance(first, str):
            return content
        if isinstance(first, dict) and "text" in first:
            content = "".join(part.get("text", "") for part in content)
        else:
            content = str(content)
    if isinstance(content, (bytes, bytearray)):
        content = content.decode()
    parsed = json.loads(content)
    if not isinstance(parsed, list):
        return []
    return [str(q) for q in parsed]


async def _structure_results(
    raw_results: list[dict], user_input: dict
) -> list[TransportOption]:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=settings.google_api_key,
    )

    user_msg = (
        f"Destination: {user_input['to_destination']}\n\n"
        f"Raw search results:\n{json.dumps(raw_results, indent=2, default=str)}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=STRUCTURING_PROMPT),
        HumanMessage(content=user_msg),
    ])

    content = response.content
    if isinstance(content, list):
        if not content:
            content = "[]"
        else:
            first = content[0]
            if isinstance(first, str):
                content = "\n".join(content)
            elif isinstance(first, dict) and "text" in first:
                content = "".join(part.get("text", "") for part in content)
            else:
                content = str(content)
    if isinstance(content, (bytes, bytearray)):
        content = content.decode()
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    parsed = json.loads(content)
    if not isinstance(parsed, list):
        parsed = []

    options = []
    for item in parsed:
        try:
            options.append(TransportOption(**item))
        except Exception:
            continue
    return options


async def transport_agent_node(state: dict) -> dict:
    user_input = state["user_input"]
    destination = user_input["to_destination"]

    try:
        logger.info("transport_agent_started", destination=destination)

        queries = await _generate_queries(user_input)
        logger.info("transport_queries_generated", count=len(queries), queries=queries)

        raw_results: list[dict] = []
        for query in queries:
            try:
                result = await tavily_search(query)
                if isinstance(result, str):
                    result = json.loads(result)
                raw_results.append({"query": query, "result": result})
            except Exception as e:
                logger.warning("transport_tavily_query_failed", query=query, error=str(e))

        if not raw_results:
            logger.warning("transport_no_results", destination=destination)
            return {"transport": [], "errors": ["Transport agent: no search results returned"]}

        options = await _structure_results(raw_results, user_input)
        logger.info("transport_agent_completed", options_count=len(options))

        return {"transport": options}

    except Exception as e:
        logger.error("transport_agent_failed", error=str(e), exc_info=True)
        return {"transport": [], "errors": [f"Transport agent: {e}"]}
