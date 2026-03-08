from __future__ import annotations

import json
from pathlib import Path

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from api.schemas import ResearchInsight
from config import settings
from tools.tavily import tavily_search

logger = structlog.get_logger()

PROMPTS_DIR = Path(__file__).parent / "prompts"
QUERY_SYSTEM_PROMPT = (PROMPTS_DIR / "research.txt").read_text()

STRUCTURING_PROMPT = """You are a travel research analyst. You will receive raw search results about a travel destination.

Extract and structure the information into a JSON array of objects, each with these fields:
- "category": one of "attraction", "weather", "road_condition", "tip", "warning"
- "title": a short descriptive title
- "summary": summary of the insight
- "source_url": the URL where this information came from (null if unavailable)

Rules:
- Return ONLY valid JSON, no markdown fences or extra text.
- Deduplicate similar insights.
- Prioritise warnings and weather information.
- Include at most 10 insights.
"""


async def _generate_queries(user_input: dict) -> list[str]:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=settings.google_api_key,
    )
    user_msg = (
        f"Destination: {user_input['to_destination']}\n"
        f"From: {user_input['from_city']}\n"
        f"Dates: {user_input['departure_date']} to {user_input['return_date']}\n"
        f"Trip style: {', '.join(user_input.get('trip_style', []))}\n"
        f"Notes: {user_input.get('notes', 'None')}"
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
    return json.loads(content)


async def _structure_results(raw_results: list[dict], user_input: dict) -> list[ResearchInsight]:
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
    return [ResearchInsight(**item) for item in parsed]


async def research_agent_node(state: dict) -> dict:
    user_input = state["user_input"]
    destination = user_input["to_destination"]
    try:
        logger.info("research_agent_started", destination=destination)
        queries = await _generate_queries(user_input)
        logger.info("research_queries_generated", count=len(queries), queries=queries)
        raw_results: list[dict] = []
        for query in queries:
            try:
                result = await tavily_search(query)
                if isinstance(result, str):
                    result = json.loads(result)
                raw_results.append({"query": query, "result": result})
            except Exception as e:
                logger.warning("tavily_query_failed", query=query, error=str(e))
        if not raw_results:
            logger.warning("research_no_results", destination=destination)
            return {"research": [], "errors": ["Research agent: no search results returned"]}
        insights = await _structure_results(raw_results, user_input)
        logger.info("research_agent_completed", insights_count=len(insights))
        return {"research": insights}
    except Exception as e:
        logger.error("research_agent_failed", error=str(e), exc_info=True)
        return {"research": [], "errors": [f"Research agent: {e}"]}
