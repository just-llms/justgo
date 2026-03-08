# JustGo

AI-powered travel planning API that generates day-by-day itineraries using a multi-agent system. Submit your trip preferences (origin, destination, dates, budget, style) and get back flight options, hotels, local transport, destination research, and a cohesive markdown itinerary with citations.

## Architecture

The app is built with **FastAPI** and **LangGraph**. A single plan request runs:

1. **Budget allocator** — Splits the total budget (from `budget_max`) into flight, hotel, transport, and activities using configurable percentages.
2. **Parallel agents** (Research, Flights, Hotels, Transport) — Run at the same time:
   - **Research** — Gemini generates 3–5 search queries, Tavily runs them, then Gemini structures results into research insights.
   - **Flights** — Resolves origin/destination to IATA (Amadeus + Gemini), searches Amadeus for outbound flights, converts prices to INR (forex-python), scores and returns top 3.
   - **Hotels** — Gemini builds a search payload, **Airbnb MCP** (`@openbnb/mcp-server-airbnb`) returns listings; filters by budget/rating and returns top 3 (with Tavily fallback if MCP fails).
   - **Transport** — Gemini generates local transport queries, Tavily runs them, Gemini structures into transport options.
3. **Ranking layer** — Pass-through; forwards flights and hotels to the itinerary step.
4. **Itinerary agent** — Merges all outputs (and any `errors` from failed agents) into one prompt; Gemini produces a day-by-day markdown itinerary with recommendations, citations, and cost breakdown.

## Requirements

- **Python 3.10+**
- **Node.js & npx** (for the Airbnb MCP server used by the Hotels agent)

## Setup

1. **Clone and create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment variables**

   Copy `.env.example` to `.env` and set:

   | Variable | Required | Description |
   |----------|----------|-------------|
   | `GOOGLE_API_KEY` | Yes | Google AI (Gemini) API key |
   | `TAVILY_API_KEY` | Yes | Tavily Search API key |
   | `AMADEUS_CLIENT_ID` | Yes | Amadeus API client ID |
   | `AMADEUS_CLIENT_SECRET` | Yes | Amadeus API client secret |
   | `LANGSMITH_API_KEY` | No | LangSmith tracing (optional) |
   | `LANGSMITH_TRACING` | No | Set to `true` to enable |
   | `LANGSMITH_PROJECT` | No | Project name (default: `justgo`) |

   Budget split defaults (override with env if needed): flights 40%, hotels 35%, transport 10%, activities 15%.

3. **Airbnb MCP (Hotels agent)**

   The Hotels agent uses `npx -y @openbnb/mcp-server-airbnb`. Ensure Node.js and npx are installed so the MCP server can be spawned. If it’s unavailable, the agent falls back to Tavily-based hotel search.

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

- API: `http://127.0.0.1:8000`
- OpenAPI docs: `http://127.0.0.1:8000/docs`

## API

### Health

- **GET** `/api/v1/health`  
  Returns `{"status": "ok"}`.

### Create plan

- **POST** `/api/v1/plan`  
  Body: JSON matching the `TripRequest` schema.

**Example request:**

```json
{
  "from_city": "Bengaluru",
  "to_destination": "Manali",
  "departure_date": "2026-10-22",
  "return_date": "2026-10-24",
  "adults": 2,
  "children": 0,
  "budget_min": 50000,
  "budget_max": 120000,
  "trip_style": ["Adventure", "Relaxed"],
  "notes": "Window seats preferred, vegetarian food"
}
```

**Example response:**

```json
{
  "status": "done",
  "budget": { "total_budget": 120000, "flight_budget": 48000, ... },
  "research": [...],
  "research_count": 10,
  "flights": [{ "airline": "...", "price": 12000, "currency": "INR", ... }],
  "hotels": [...],
  "transport": [...],
  "itinerary": "## Day 1 - Bengaluru to Manali\n\n...",
  "errors": []
}
```

Budgets are in **INR**. Flight prices from Amadeus (often EUR) are converted to INR using forex-python. If an agent fails, a message is appended to `errors` and the rest of the response still returns (e.g. empty `flights` with an error string so the itinerary can mention that no flights were found).

## Project structure

```
app/
  main.py              # FastAPI app, CORS, LangSmith env, logging
  config.py            # Pydantic Settings (env vars)
  api/
    routes.py          # GET /health, POST /plan
    schemas.py         # TripRequest, BudgetAllocation, FlightOption, etc.
  graph/
    state.py           # TravelState TypedDict
    budget.py          # budget_allocator_node
    builder.py         # LangGraph: budget → parallel agents → ranking → itinerary
  agents/
    research.py        # Tavily + Gemini
    flights.py         # Amadeus + Gemini, forex to INR
    hotels.py         # Airbnb MCP + Gemini (Tavily fallback)
    transport.py       # Tavily + Gemini
    itinerary.py       # Gemini: merge state → markdown itinerary
    prompts/           # .txt system prompts per agent
  tools/
    tavily.py          # TavilySearch wrapper
    airbnb.py          # MCP client for @openbnb/mcp-server-airbnb
  utils/
    logging.py         # structlog setup
```

## Tech stack

- **FastAPI** — HTTP API  
- **LangGraph** — Multi-agent graph (state, parallel nodes, conditional edges)  
- **LangChain** — Gemini (ChatGoogleGenerativeAI), Tavily, Amadeus toolkit  
- **Amadeus** — Flight search and airport (IATA) resolution  
- **Tavily** — Web search for research and transport  
- **MCP** — Airbnb listings via `@openbnb/mcp-server-airbnb`  
- **forex-python** — Currency conversion (e.g. EUR → INR) for flight prices  
- **Pydantic** — Request/response and settings  
- **structlog** — Logging  
- **LangSmith** — Optional tracing

## License

See repository license file.
