import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config import settings
from utils.logging import setup_logging

if settings.langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_tracing else "false"
if settings.langsmith_project:
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

setup_logging(settings.log_level)

app = FastAPI(
    title="JustGo API",
    version="1.0.0",
    description="AI-powered travel planning with multi-agent orchestration",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
