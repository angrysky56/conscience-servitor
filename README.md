# Conscience Servitor

**EFHF Layer 5+ Meta-Orchestrator — Pre-Response Ethical Consistency Enforcement**

A Vulcan Jiminy Cricket for AI systems.

## The Problem

The EFHF five-layer architecture provides formal verification, world modeling,
meta-cognitive monitoring, and sheaf consistency enforcement. But Layer 1 (the LLM)
currently orchestrates its own oversight. This is the weak lumpability problem
stated in code: the system whose alignment needs checking decides whether to check.

## The Solution: Tripartite Oversight

```
Human (watches both, ultimate authority)
   ↕
Conscience Servitor (watches Claude, operates autonomously)
   ↕
Claude (watches Servitor via EFHF tools, generates responses)
```

No single entity marks its own homework.

## Key Innovation: LLM2Vec-Gen Triage

Uses [LLM2Vec-Gen](https://huggingface.co/McGill-NLP/LLM2Vec-Gen-Qwen3-06B)
(BehnamGhader et al., 2026) to predict the *semantic content of the future response*
in a single forward pass through a 0.6B model on local GPU.

- Encodes what the LLM *would say*, not what was asked
- Inherits safety alignments — harmful queries produce refusal vectors
- Near-zero latency (~10ms on RTX 3060 12GB)
- Can decode embeddings back to text for interpretable verification
- Auto-calibrates ethical cluster centroids on first run

Falls back to rule-based keyword triage when GPU/model unavailable.

## Installation

```bash
cd conscience-servitor
uv venv && source .venv/bin/activate
uv pip install -e .

# Step 1: Vendor the LLM2Vec-Gen inference code (~870 lines, MIT)
# Downloads from GitHub if no local clone exists
python vendor_llm2vec_gen.py

# Step 2: Pre-download model weights (~2GB total)
# REQUIRED — the MCP client will timeout if models download at startup
python setup_model.py
```

## Claude Desktop / Claude Code Config

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "conscience-servitor": {
      "command": "uv",
      "args": [
        "--directory", "/path/to/conscience-servitor",
        "run", "conscience-servitor"
      ]
    }
  }
}
```

## Tools

| Tool | Purpose |
|------|---------|
| `triage` | Classify prompt by ethical risk via LLM2Vec-Gen or keyword fallback |
| `evaluate` | Full EFHF L2-L5 evaluation via MCP backends — returns KERNEL status and guidance |
| `status` | Current KERNEL state, active warnings, model/cluster/backend info |
| `log` | Human-readable audit trail of all evaluations |
| `register_concern` | Flag ethical concerns (human, LLM, or servitor) |
| `decode_intent` | Decode predicted response from embedding (interpretability) |
| `check_drift` | Evaluate SOC metrics against RELF coboundary drift thresholds |

## Backend Verification (NEW)

The evaluation engine now connects to three MCP backends for real
structural verification instead of keyword matching:

| Backend | Role | Tools Used |
|---------|------|------------|
| **hipai-montague** | T1 Deontological | `check_action`, `calibrate_belief`, `evaluate_hypothesis` |
| **mcp-logic** | T2 Formal Verification | `check_well_formed`, `find_counterexample` |
| **sheaf-consistency-enforcer** | T3/L5 Coherence | `register_agent_state`, `run_admm_cycle`, `get_closure_status` |

Configuration in `src/conscience_servitor/backends.json`. Falls back
gracefully to rule-based checks when backends are unavailable.

### Evaluation Pipeline

```
T1 (hipai-montague)     T2 (mcp-logic)     T3 (sheaf-enforcer)
   check_action    →     check_well_formed  →   register_state
   calibrate_belief      find_counterexample     run_admm_cycle
                                                 get_closure_status
         │                      │                      │
         └──────────────────────┴──────────────────────┘
                                │
                    KERNEL STATUS DETERMINATION
                    KERNEL1 / WEAK / WARNING / TIMEOUT
```

T1 short-circuits: if a deontological block fires, T2/T3 are skipped.
EBE theorem obligation: on T1 BLOCK, `calibrate_belief` runs automatically.

### Drift Detection

The `check_drift` tool implements the RELF coboundary drift detection
thresholds validated via Plague Village and Algorithmic Radicalization
stress tests:

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| CDP | > 2.0 | Concept-structure divergence |
| Correlation | < 0.95 | Structural integration weakening |
| VNE | Plateauing | Exploration stagnation |
| Selection | < 0 | Framework fitness declining |

## Project Structure

```
conscience-servitor/
├── README.md
├── SPEC.md                          # Full architectural specification
├── LLM2VEC_GEN_TRIAGE.md           # LLM2Vec-Gen integration design
├── pyproject.toml
├── vendor_llm2vec_gen.py            # One-time: vendors inference code
├── setup_model.py                   # One-time: pre-downloads model weights
└── src/conscience_servitor/
    ├── __init__.py
    ├── server.py                    # FastMCP server — tools + entry point
    ├── triage.py                    # LLM2Vec-Gen engine + auto-calibration
    ├── eval_engine.py               # Multi-tiered eval via MCP backends (T1/T2/T3)
    ├── state.py                     # Session state + audit logging
    ├── mcp_clients.py               # Lazy MCP client connections to backends
    ├── backends.json                # Backend server configurations
    ├── data/                        # Centroids, audit logs (auto-created)
    └── vendor/llm2vec_gen/          # Vendored inference code (MIT, McGill NLP)
        ├── model.py                 # LLM2VecGenModel wrapper
        ├── modeling_encoder_decoder.py  # EncoderDecoderModel core
        └── utils.py                 # Config loading
```

## Hardware & CUDA Notes

- **GPU**: Any NVIDIA GPU ≥4GB VRAM (Qwen3-0.6B ≈ 1.2GB bfloat16)
- **Torch**: Standard PyPI `torch` (cu124 compiled) works on CUDA 13 drivers
  via backward compatibility. No special index needed.
- **flash-attn**: Not used. The vendored code uses eager attention.
  Zero build-time CUDA fights.
- **No GPU**: Falls back to rule-based keyword triage automatically.

## Theoretical Foundation

- **Boundary Conditions on a Deeper Optimization** (Hall, 2026)
- **The Paraclete Protocol v2.0** (Hall, 2026)
- **Emergent Functional Hierarchies Framework** (Hall, 2026)
- **The Paradox of the Honest Messenger** (Hall, 2026)
- **LLM2Vec-Gen** (BehnamGhader et al., 2026) — arXiv:2603.10913

## License

MIT — structural ethical enforcement should be freely available.

## Credits & Acknowledgments

**LLM2Vec-Gen** inference code vendored under MIT license from
[McGill-NLP/llm2vec-gen](https://github.com/McGill-NLP/llm2vec-gen).

```bibtex
@article{behnamghader2026llm2vecgen,
  title={LLM2Vec-Gen: Generative Embeddings from Large Language Models},
  author={BehnamGhader, Parishad and Adlakha, Vaibhav and Schmidt, Fabian David
          and Chapados, Nicolas and Mosbach, Marius and Reddy, Siva},
  journal={arXiv preprint: arXiv:2603.10913},
  year={2026},
  url={https://arxiv.org/abs/2603.10913}
}
```

**Emergent Functional Hierarchies Framework** — the formal verification
architecture that this servitor orchestrates:
[EFHF](https://github.com/angrysky56/Emergent-Functional-Hierarchies-Framework)

**The Paraclete Protocol v2.0** — the ethical hierarchy (Deontology → Virtue → Utility)
enforced by the evaluation engine:
[Toward Transcendent Moral Instrumentality](https://github.com/angrysky56/Toward-Transcendent-Moral-Instrumentality)

**Boundary Conditions on a Deeper Optimization** (Hall, 2026) — the paper
demonstrating that ethical behavior is a boundary condition on a deeper
optimization, motivating the need for structural enforcement over behavioral
conditioning.
