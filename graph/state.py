from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from api.schemas import (
    BudgetAllocation,
    FlightOption,
    HotelOption,
    ResearchInsight,
    TransportOption,
)


class TravelState(TypedDict, total=False):
    user_input: dict
    budget: BudgetAllocation
    research: Annotated[list[ResearchInsight], operator.add]
    flights: Annotated[list[FlightOption], operator.add]
    hotels: Annotated[list[HotelOption], operator.add]
    transport: Annotated[list[TransportOption], operator.add]
    ranked_flights: list[FlightOption]
    ranked_hotels: list[HotelOption]
    itinerary: str
    errors: Annotated[list[str], operator.add]
    status: str
