from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TripRequest(BaseModel):
    from_city: str = Field(..., min_length=2, max_length=100, examples=["Bengaluru"])
    to_destination: str = Field(..., min_length=2, max_length=100, examples=["Manali"])
    departure_date: date = Field(..., examples=["2026-10-22"])
    return_date: date = Field(..., examples=["2026-10-24"])
    adults: int = Field(..., ge=1, le=9)
    children: int = Field(default=0, ge=0, le=9)
    budget_min: int = Field(..., ge=1000, description="Minimum budget in INR")
    budget_max: int = Field(..., ge=1000, description="Maximum budget in INR")
    trip_style: list[str] = Field(
        default_factory=list,
        examples=[["Adventure", "Relaxed"]],
        description="Tags: Adventure, Relaxed, Cultural, Luxury, Budget, Family",
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=500,
        examples=["Window seats preferred, vegetarian food"],
    )

    @field_validator("return_date")
    @classmethod
    def return_after_departure(cls, v: date, info) -> date:
        if "departure_date" in info.data and v <= info.data["departure_date"]:
            raise ValueError("return_date must be after departure_date")
        return v

    @field_validator("budget_max")
    @classmethod
    def max_gte_min(cls, v: int, info) -> int:
        if "budget_min" in info.data and v < info.data["budget_min"]:
            raise ValueError("budget_max must be >= budget_min")
        return v


class BudgetAllocation(BaseModel):
    total_budget: int
    flight_budget: int
    hotel_budget: int
    transport_budget: int
    activities_budget: int


class FlightOption(BaseModel):
    airline: str
    flight_number: str
    departure_time: str
    arrival_time: str
    duration_minutes: int
    stops: int
    price: float
    currency: str = "INR"
    booking_url: Optional[str] = None
    source: str = "Amadeus"


class HotelOption(BaseModel):
    name: str
    location: str
    price_per_night: float
    total_price: float
    rating: Optional[float] = None
    amenities: list[str] = Field(default_factory=list)
    booking_url: Optional[str] = None
    source: str = ""


class TransportOption(BaseModel):
    mode: str
    description: str
    estimated_cost: Optional[str] = None
    availability_notes: str = ""
    source_url: Optional[str] = None


class ResearchInsight(BaseModel):
    category: str
    title: str
    summary: str
    source_url: Optional[str] = None
