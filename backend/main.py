"""FastAPI service for TruthLens."""

from __future__ import annotations

import logging
import requests as req

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analyzer import analyze_text
from human_eval import get_next_sentence, get_results, submit_annotation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TruthLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    """Analyze request body."""
    text: str
    fast_mode: bool = True


class AnnotationInput(BaseModel):
    """Annotation request body."""
    label: str


@app.on_event("startup")
async def startup_check():
    """Check Ollama connection on startup."""
    try:
        r = req.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            logger.info("✅ Ollama connected. Models: %s", models)
            if "gemma3:4b" in models:
                logger.info("✅ gemma3:4b is available and ready!")
            else:
                logger.error("❌ gemma3:4b NOT found! Run: ollama pull gemma3:4b")
        else:
            logger.error("❌ Ollama returned status %d", r.status_code)
    except Exception as e:
        logger.error("❌ Cannot connect to Ollama: %s", e)


@app.post("/analyze")
async def analyze(payload: TextInput) -> dict:
    """Analyze text with TruthLens.

    fast_mode=True  → single pass, faster (~15-30 sec per sentence)
    fast_mode=False → 3-sample voting, slower but with confidence scores
    """
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        logger.info(
            "Analyze request. fast_mode=%s chars=%d",
            payload.fast_mode, len(payload.text)
        )
        result = analyze_text(payload.text, fast_mode=payload.fast_mode)
        logger.info("Analysis done. trust_score=%d", result["trust_score"])
        return result
    except Exception as exc:
        logger.exception("Analyze endpoint failure")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/")
async def root() -> dict[str, str]:
    """Health check."""
    return {"status": "TruthLens backend is running!"}


@app.get("/health")
async def health() -> dict:
    """Detailed health check."""
    ollama_status = "unknown"
    available_models = []
    try:
        r = req.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            available_models = [m["name"] for m in r.json().get("models", [])]
            ollama_status = "running"
        else:
            ollama_status = f"error: status {r.status_code}"
    except Exception as e:
        ollama_status = f"not reachable: {str(e)}"

    return {
        "status": "running",
        "ollama": ollama_status,
        "available_models": available_models,
        "target_model": "gemma3:4b",
        "model_available": "gemma3:4b" in available_models
    }


@app.get("/test")
async def test() -> dict:
    """Test Gemma connection with a sample text."""
    try:
        logger.info("Test endpoint called")
        result = analyze_text(
            "Einstein was born in Germany. He invented the television.",
            fast_mode=True
        )
        return {
            "status": "success",
            "trust_score": result["trust_score"],
            "sentences_analyzed": len(result["sentences"]),
            "domain": result["domain"],
            "result": result
        }
    except Exception as exc:
        logger.exception("Test endpoint failure")
        return {
            "status": "error",
            "error": str(exc),
            "message": "Check Ollama is running: ollama serve"
        }


@app.get("/annotate/next")
async def annotate_next() -> dict:
    """Get next sentence for human annotation."""
    return get_next_sentence()


@app.post("/annotate/submit")
async def annotate_submit(payload: AnnotationInput) -> dict:
    """Submit one annotation label."""
    return submit_annotation(payload.label)


@app.get("/annotate/results")
async def annotate_results() -> dict:
    """Get annotation agreement metrics."""
    return get_results()