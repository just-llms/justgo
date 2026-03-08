from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.flights import flights_agent_node
from agents.hotels import hotels_agent_node
from agents.itinerary import itinerary_agent_node
from agents.research import research_agent_node
from agents.transport import transport_agent_node
from graph.budget import budget_allocator_node
from graph.state import TravelState


def _ranking_layer_node(state: dict) -> dict:
    """Pass-through ranking layer -- real scoring will be added later."""
    return {
        "ranked_flights": state.get("flights", []),
        "ranked_hotels": state.get("hotels", []),
    }


def _route_to_agents(state: dict) -> list[Send]:
    """Fan-out to all four agents in parallel via Send() API."""
    payload = {
        "user_input": state["user_input"],
        "budget": state["budget"],
    }
    return [
        Send("research_agent", payload),
        Send("flights_agent", payload),
        Send("hotels_agent", payload),
        Send("transport_agent", payload),
    ]


def build_graph() -> StateGraph:
    graph = StateGraph(TravelState)

    graph.add_node("budget_allocator", budget_allocator_node)
    graph.add_node("research_agent", research_agent_node)
    graph.add_node("flights_agent", flights_agent_node)
    graph.add_node("hotels_agent", hotels_agent_node)
    graph.add_node("transport_agent", transport_agent_node)
    graph.add_node("ranking_layer", _ranking_layer_node)
    graph.add_node("itinerary_agent", itinerary_agent_node)

    graph.add_edge(START, "budget_allocator")
    graph.add_conditional_edges("budget_allocator", _route_to_agents)

    graph.add_edge("research_agent", "ranking_layer")
    graph.add_edge("flights_agent", "ranking_layer")
    graph.add_edge("hotels_agent", "ranking_layer")
    graph.add_edge("transport_agent", "ranking_layer")

    graph.add_edge("ranking_layer", "itinerary_agent")
    graph.add_edge("itinerary_agent", END)

    return graph


compiled_graph = build_graph().compile()
