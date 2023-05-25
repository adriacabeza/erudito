from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ingest import ingest
from src.query import query

description = """
Service to talk with Erudito
"""

app = FastAPI(title="Erudito", version="0.0.1", description=description)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/ingest")
def ingest_endpoint(
    documentation_path: str,
    model_path: str = "models/gpt4all/gpt4all-lora-quantized-new.bin",
):
    ingest(documentation_path=documentation_path, model_path=model_path)
    return {"result": "🙌 Vector store with embeddings created"}


class QueryPayload(BaseModel):
    question: str = "What is the Cloud SIEM Investigator?"
    model_path: str = ""
    index_path: Optional[str]


@app.post("/query")
def query_entrypoint(payload: QueryPayload) -> dict:
    return {
        "result": query(
            question=payload.question,
            model_path=payload.model_path,
            index_path=Path(payload.index_path),
        )
    }
