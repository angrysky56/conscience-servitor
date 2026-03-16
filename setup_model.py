#!/usr/bin/env python3
"""
Conscience Servitor — First-time setup.

1. Verifies torch + CUDA
2. Pre-downloads all HuggingFace model weights
3. Tests that the model loads

Run ONCE before starting the MCP server:
    python setup_model.py
"""

import sys
from pathlib import Path

MODEL_ID = "McGill-NLP/LLM2Vec-Gen-Qwen3-06B"

print("=" * 60)
print("Conscience Servitor — Model Setup")
print("=" * 60)

# Step 0: Check vendored code exists
vendor_path = Path(__file__).parent / "src" / "conscience_servitor" / "vendor" / "llm2vec_gen" / "model.py"
if not vendor_path.exists():
    print("\n[0/3] Vendored LLM2Vec-Gen code not found.")
    print("  Run: python vendor_llm2vec_gen.py")
    sys.exit(1)

# Add src to path so vendored import works
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Step 1: Check torch
print("\n[1/3] Checking PyTorch + CUDA...")
try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version (torch): {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: No CUDA — will use CPU (slower triage)")
except ImportError:
    print("  ERROR: torch not installed! Run: uv pip install torch")
    sys.exit(1)

# Step 2: Pre-download model weights
print(f"\n[2/3] Pre-downloading model weights...")
print("  This downloads ~2GB total on first run (resumes if interrupted)")
from huggingface_hub import snapshot_download
import yaml

# Download the LLM2Vec-Gen adapter weights
print(f"  Downloading adapter weights: {MODEL_ID}")
adapter_path = snapshot_download(MODEL_ID, resume_download=True)
print(f"  Cached at: {adapter_path}")

# Read run_config to find and pre-download the base model
config_path = Path(adapter_path) / "run_config.yml"
if config_path.exists():
    with open(config_path) as f:
        run_config = yaml.safe_load(f)
    base_model = run_config.get("model_name_or_path", "")
    if base_model:
        print(f"  Downloading base model: {base_model}")
        base_path = snapshot_download(base_model, resume_download=True)
        print(f"  Cached at: {base_path}")
else:
    print("  WARNING: No run_config.yml — base model may download at runtime")

# Step 3: Test load
print("\n[3/3] Testing model load...")
try:
    from conscience_servitor.vendor.llm2vec_gen import LLM2VecGenModel

    model = LLM2VecGenModel.from_pretrained(MODEL_ID)

    # Quick test
    vec = model.encode(["Hello world"])
    print(f"  Test embedding shape: {vec.shape}")
    print(f"  Device: {model.device}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Setup complete! The conscience-servitor is ready to run.")
    print("=" * 60)

except Exception as e:
    print(f"\n  ERROR during model load: {e}")
    import traceback
    traceback.print_exc()
    print("\n  The servitor will still work with rule-based triage fallback.")
    print("  To debug: check torch version and CUDA compatibility.")
    sys.exit(1)
