from __future__ import annotations

import structlog

from app.api.schemas import BudgetAllocation
from app.config import settings

logger = structlog.get_logger()


def budget_allocator_node(state: dict) -> dict:
    user_input = state["user_input"]
    total = user_input["budget_max"]

    allocation = BudgetAllocation(
        total_budget=total,
        flight_budget=int(total * settings.budget_flight_pct),
        hotel_budget=int(total * settings.budget_hotel_pct),
        transport_budget=int(total * settings.budget_transport_pct),
        activities_budget=int(total * settings.budget_activities_pct),
    )

    logger.info(
        "budget_allocated",
        total=total,
        flights=allocation.flight_budget,
        hotels=allocation.hotel_budget,
        transport=allocation.transport_budget,
        activities=allocation.activities_budget,
    )

    return {"budget": allocation, "status": "budget_allocated"}
