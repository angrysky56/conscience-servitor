#!/usr/bin/env python3
"""
Pre-download LLM2Vec-Gen model weights.

Run this ONCE before starting the conscience-servitor MCP server.
The MCP client kills the process if startup takes too long,
so the model must already be cached locally.

Usage:
    python download_model.py
    # or
    uv run python download_model.py
"""

from huggingface_hub import snapshot_download

MODEL_ID = "McGill-NLP/LLM2Vec-Gen-Qwen3-06B"

print(f"Downloading {MODEL_ID}...")
print("This may take a few minutes on first run (~1.2GB)")
print()

path = snapshot_download(MODEL_ID)

print(f"\nModel cached at: {path}")
print("The conscience-servitor will now load instantly on startup.")
