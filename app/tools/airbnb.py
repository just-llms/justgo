from __future__ import annotations

import asyncio
import json

import structlog
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import settings

logger = structlog.get_logger()


async def airbnb_search(
    location: str,
    check_in: str,
    check_out: str,
    adults: int,
    *,
    children: int = 0,
    min_price: int | None = None,
    max_price: int | None = None,
    page: int = 1,
) -> dict | None:
    """Call the @openbnb/mcp-server-airbnb MCP server and return parsed results.

    Returns the parsed JSON dict on success, or None on any failure so the
    caller can fall back to Tavily.
    """
    server_params = StdioServerParameters(
        command=settings.airbnb_mcp_command,
        args=settings.airbnb_mcp_args,
    )

    args: dict = {
        "location": location,
        "checkIn": check_in,
        "checkOut": check_out,
        "adults": adults,
        "children": children,
        "page": page,
    }
    if min_price is not None:
        args["minPrice"] = min_price
    if max_price is not None:
        args["maxPrice"] = max_price

    try:
        result = await asyncio.wait_for(
            _call_mcp(server_params, args),
            timeout=settings.airbnb_mcp_timeout_seconds,
        )
        return result
    except asyncio.TimeoutError:
        logger.warning("airbnb_mcp_timeout", timeout=settings.airbnb_mcp_timeout_seconds)
        return None
    except Exception as exc:
        logger.warning("airbnb_mcp_error", error=str(exc))
        return None


async def _call_mcp(server_params: StdioServerParameters, args: dict) -> dict | None:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool("airbnb_search", args)

            if not result.content:
                return None

            first = result.content[0]
            text = getattr(first, "text", None)
            if text is None:
                return None
            return json.loads(text)
