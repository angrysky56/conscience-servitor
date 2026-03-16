#!/usr/bin/env python3
"""
Conscience Servitor — Vendor LLM2Vec-Gen inference code.

Copies the minimal inference code from llm2vec-gen into this package
so users don't need to clone a separate repo or fight pip dependency hell.

Sources (tried in order):
1. Local clone at ~/Repositories/llm2vec-gen (for development)
2. Downloads from GitHub (for end users)

Only ~870 lines of MIT-licensed code are vendored:
- models/modeling_encoder_decoder.py (EncoderDecoderModel, ProjectionModel)
- models/utils.py (YAML config loading)
- models/__init__.py (LLM2VecGenModel wrapper)

Usage:
    python vendor_llm2vec_gen.py
"""

import shutil
import urllib.request
import os
from pathlib import Path

VENDOR_DIR = Path(__file__).parent / "src" / "conscience_servitor" / "vendor" / "llm2vec_gen"
LOCAL_REPO = Path.home() / "Repositories" / "llm2vec-gen"
GITHUB_RAW = "https://raw.githubusercontent.com/McGill-NLP/llm2vec-gen/main"

FILES_TO_VENDOR = [
    ("llm2vec_gen/models/__init__.py", "model.py"),
    ("llm2vec_gen/models/modeling_encoder_decoder.py", "modeling_encoder_decoder.py"),
    ("llm2vec_gen/models/utils.py", "utils.py"),
]

def vendor():
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    # Write __init__.py that re-exports LLM2VecGenModel
    init_content = '"""Vendored LLM2Vec-Gen inference code (MIT License, McGill NLP)."""\nfrom .model import LLM2VecGenModel\n__all__ = ["LLM2VecGenModel"]\n'
    (VENDOR_DIR / "__init__.py").write_text(init_content)

    for src_path, dst_name in FILES_TO_VENDOR:
        dst = VENDOR_DIR / dst_name
        
        # Try local clone first
        local = LOCAL_REPO / src_path
        if local.exists():
            print(f"  Copying from local: {local}")
            content = local.read_text()
        else:
            # Download from GitHub
            url = f"{GITHUB_RAW}/{src_path}"
            print(f"  Downloading: {url}")
            try:
                content = urllib.request.urlopen(url).read().decode()
            except Exception as e:
                print(f"  ERROR: Could not fetch {url}: {e}")
                return False

        # Fix internal imports for vendored location
        content = content.replace(
            "from llm2vec_gen.models.modeling_encoder_decoder",
            "from .modeling_encoder_decoder"
        )
        content = content.replace(
            "from llm2vec_gen.models",
            "from ."
        )

        dst.write_text(content)
        print(f"  -> {dst}")

    print(f"\nVendored {len(FILES_TO_VENDOR)} files to {VENDOR_DIR}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Vendoring LLM2Vec-Gen inference code")
    print("=" * 60)

    if vendor():
        print("\nDone. Now run: python setup_model.py")
    else:
        print("\nFailed — check errors above.")
