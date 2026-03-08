from __future__ import annotations

from langchain_tavily import TavilySearch
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings

tavily_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
    include_answer=True,
    tavily_api_key=settings.tavily_api_key,
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
async def tavily_search(query: str) -> dict:
    return await tavily_tool.ainvoke({"query": query})
