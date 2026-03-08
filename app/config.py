from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    google_api_key: str = ""
    tavily_api_key: str = ""
    amadeus_client_id: str = ""
    amadeus_client_secret: str = ""

    langsmith_api_key: str = ""
    langsmith_tracing: bool = False
    langsmith_project: str = "justgo"

    airbnb_mcp_command: str = "npx"
    airbnb_mcp_args: list[str] = ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"]
    airbnb_mcp_timeout_seconds: int = 30

    log_level: str = "INFO"
    max_concurrent_plans: int = 10
    agent_timeout_seconds: int = 60

    budget_flight_pct: float = 0.40
    budget_hotel_pct: float = 0.35
    budget_transport_pct: float = 0.10
    budget_activities_pct: float = 0.15

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
