from __future__ import annotations

import asyncio

from fastapi import APIRouter

from api.schemas import TripRequest
from graph.builder import compiled_graph

router = APIRouter(prefix="/api/v1")


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.post("/plan")
async def create_plan(request: TripRequest):
    input_data = {
        "user_input": request.model_dump(mode="json"),
        "research": [],
        "flights": [],
        "hotels": [],
        "transport": [],
        "errors": [],
    }

    final_state = await compiled_graph.ainvoke(input_data)

    # Let aiohttp/connector cleanup tasks run (avoids "Task was destroyed but it is pending!")
    await asyncio.sleep(0.25)

    research_list = final_state.get("research") or []
    research_data = [
        r.model_dump() if hasattr(r, "model_dump") else r
        for r in research_list
    ]
    budget = final_state.get("budget")
    budget_data = (
        budget.model_dump() if budget is not None and hasattr(budget, "model_dump") else budget
    )
    flights_list = final_state.get("flights") or []
    flights_data = [
        f.model_dump() if hasattr(f, "model_dump") else f
        for f in flights_list
    ]
    hotels_list = final_state.get("hotels") or []
    hotels_data = [
        h.model_dump() if hasattr(h, "model_dump") else h
        for h in hotels_list
    ]
    transport_list = final_state.get("transport") or []
    transport_data = [
        t.model_dump() if hasattr(t, "model_dump") else t
        for t in transport_list
    ]

    return {
        "status": "done",
        "budget": budget_data,
        "research": research_data,
        "research_count": len(research_data),
        "flights": flights_data,
        "hotels": hotels_data,
        "transport": transport_data,
        "itinerary": final_state.get("itinerary", ""),
        "errors": final_state.get("errors", []),
    }
