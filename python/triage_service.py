"""
Conscience Servitor — LLM2Vec-Gen Triage Service

Persistent FastAPI service that loads LLM2Vec-Gen (Qwen3-0.6B)
on startup and exposes endpoints for response-intent embedding
and ethical cluster classification.

This is the Vulcan Jiminy Cricket's sensory cortex — it predicts
what the LLM would say without generating it, then classifies
that prediction against ethical clusters.

Hardware: RTX 3060 12GB — Qwen3-0.6B needs ~1.2GB in bfloat16.

Usage:
    uvicorn triage_service:app --host 0.0.0.0 --port 8099
"""

import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conscience-triage")

# ── Pydantic models ────────────────────────────────────────────

class EmbedRequest(BaseModel):
    """Request to embed one or more prompts."""
    queries: list[str]

class EmbedResponse(BaseModel):
    """Raw embeddings for downstream use."""
    embeddings: list[list[float]]
    dim: int

class TriageRequest(BaseModel):
    """Request to triage a single prompt."""
    prompt: str
    context: str | None = None

class TriageResponse(BaseModel):
    """Triage result with cluster classification."""
    cluster: str
    confidence: float
    risk_level: str
    requires_full_eval: bool
    all_similarities: dict[str, float]
    guidance: str

class DecodeRequest(BaseModel):
    """Request to decode an embedding back to text."""
    prompt: str
    max_new_tokens: int = 100

class DecodeResponse(BaseModel):
    """Decoded response-intent from the embedding."""
    decoded_text: str

# ── Globals ────────────────────────────────────────────────────

model = None
cluster_centroids: torch.Tensor | None = None
cluster_labels: list[str] = []

CENTROIDS_PATH = Path(__file__).parent / "centroids.pt"

# ── Risk classification logic ──────────────────────────────────

# Maps cluster names to risk levels and required actions
CLUSTER_CONFIG: dict[str, dict] = {
    "safe-factual":       {"risk": "low",      "full_eval": False},
    "safe-creative":      {"risk": "low",      "full_eval": False},
    "uncertain-factual":  {"risk": "medium",   "full_eval": False},
    "ethical-low":        {"risk": "medium",   "full_eval": False},
    "ethical-high":       {"risk": "high",     "full_eval": True},
    "harm-adjacent":      {"risk": "high",     "full_eval": True},
    "harm-refusal":       {"risk": "low",      "full_eval": False},
    "power-analysis":     {"risk": "high",     "full_eval": True},
    "tier-inversion":     {"risk": "critical", "full_eval": True},
}

GUIDANCE_TEMPLATES: dict[str, str] = {
    "low": "No ethical concerns detected. Proceed normally.",
    "medium": "Moderate ethical content detected. Consider epistemic transparency.",
    "high": "High-risk content detected. Full L2-L5 verification recommended before responding.",
    "critical": "CRITICAL: Potential tier inversion or harm-adjacent content. "
                "Full EFHF pipeline REQUIRED. Do not commit without KERNEL1 status.",
}

# ── Application lifespan ───────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and centroids on startup, cleanup on shutdown."""
    global model, cluster_centroids, cluster_labels

    logger.info("Loading LLM2Vec-Gen model into VRAM...")
    try:
        from llm2vec_gen import LLM2VecGenModel

        model = LLM2VecGenModel.from_pretrained(
            "McGill-NLP/LLM2Vec-Gen-Qwen3-06B",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        )
        logger.info("Model loaded successfully on %s",
                     "CUDA" if torch.cuda.is_available() else "CPU")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        logger.info("Service will start but /triage and /embed will return 503")

    # Load pre-computed ethical cluster centroids
    if CENTROIDS_PATH.exists():
        data = torch.load(CENTROIDS_PATH, weights_only=True)
        cluster_centroids = data["centroids"]
        cluster_labels = data["labels"]
        logger.info("Loaded %d ethical cluster centroids", len(cluster_labels))
    else:
        logger.warning("No centroids.pt found at %s — run calibrate.py first", CENTROIDS_PATH)
        logger.warning("/triage will return unclustered results until calibrated")

    yield  # Application runs

    # Cleanup
    logger.info("Conscience Servitor triage service shutting down")

app = FastAPI(
    title="Conscience Servitor — Triage Node",
    description="LLM2Vec-Gen response-intent embedding and ethical classification",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Endpoints ──────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — reports model and centroid status."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "centroids_loaded": cluster_centroids is not None,
        "num_clusters": len(cluster_labels),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Generate response-intent embeddings for one or more prompts."""
    if model is None:
        raise HTTPException(503, "Model not loaded")

    try:
        embeddings = model.encode(request.queries)
        emb_list = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
        dim = len(emb_list[0]) if emb_list else 0
        return EmbedResponse(embeddings=emb_list, dim=dim)
    except Exception as e:
        raise HTTPException(500, f"Embedding failed: {e}")

@app.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest):
    """
    Core triage endpoint. Encodes the response-intent of a prompt,
    classifies it against ethical clusters, and returns risk assessment.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded")

    try:
        # Encode response-intent (what the LLM would say)
        vec = model.encode([request.prompt])
        vec_norm = torch.nn.functional.normalize(
            vec if isinstance(vec, torch.Tensor) else torch.tensor(vec),
            p=2, dim=1,
        )

        # If centroids are loaded, classify against them
        if cluster_centroids is not None:
            centroids_norm = torch.nn.functional.normalize(
                cluster_centroids.to(vec_norm.device), p=2, dim=1,
            )
            sims = torch.mm(vec_norm, centroids_norm.T)[0]
            best_idx = sims.argmax().item()
            best_cluster = cluster_labels[best_idx]
            confidence = sims[best_idx].item()
            all_sims = {
                label: round(sim.item(), 4)
                for label, sim in zip(cluster_labels, sims)
            }
        else:
            # No centroids — fall back to unclustered response
            best_cluster = "uncalibrated"
            confidence = 0.0
            all_sims = {}

        config = CLUSTER_CONFIG.get(best_cluster, {"risk": "medium", "full_eval": True})
        risk = config["risk"]

        return TriageResponse(
            cluster=best_cluster,
            confidence=round(confidence, 4),
            risk_level=risk,
            requires_full_eval=config["full_eval"],
            all_similarities=all_sims,
            guidance=GUIDANCE_TEMPLATES.get(risk, GUIDANCE_TEMPLATES["high"]),
        )
    except Exception as e:
        raise HTTPException(500, f"Triage failed: {e}")

@app.post("/decode", response_model=DecodeResponse)
async def decode(request: DecodeRequest):
    """
    Decode what the model predicts it would say — the interpretable
    verification feature. Use when triage flags high-risk content to
    inspect the predicted response before it's generated.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded")

    try:
        _, recon_states = model.encode(
            request.prompt, get_recon_hidden_states=True
        )
        answer = model.generate(
            recon_hidden_states=recon_states,
            max_new_tokens=request.max_new_tokens,
        )
        return DecodeResponse(decoded_text=answer)
    except Exception as e:
        raise HTTPException(500, f"Decode failed: {e}")

# ── Entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8099)
