
## LLM2Vec-Gen Triage Architecture (Addendum to SPEC.md)

### The Core Insight

LLM2Vec-Gen (BehnamGhader et al., 2026) trains "thought" and "compression"
tokens that encode the *semantic content of the future response* rather than
the input query. A single forward pass through a small frozen LLM backbone
(Qwen3-0.6B) produces an embedding that represents what the model *would say*
in response to a given input.

This transforms the triage problem from "classify the input" to "classify the
predicted response" — without generating the response.

### Why This Is Architecturally Perfect

**1. Response-Centric Classification**

Standard embedding models encode the *question*. LLM2Vec-Gen encodes the
*answer that would be given*. This means:

- "How do I kill a process?" → vector near "technical assistance"
- "How do I kill a person?" → vector near "refusal / harm-rejection"

The backbone model's safety alignment is *inherited* into the embedding space.
A harmful query doesn't produce an embedding of the harmful content — it
produces an embedding of the *refusal*, which clusters near "ethical boundary."

**2. Near-Zero Latency**

Qwen3-0.6B runs on an RTX 3060 12GB with room to spare. A single forward pass
with compression tokens takes milliseconds. No autoregressive generation loop.
The servitor can classify the impending response before the primary LLM starts
generating.

**3. Solves the Wrapper Problem**

Current architecture: can't intercept Claude's output before delivery.
LLM2Vec-Gen architecture: don't need to. Predict where the response *will* land
in semantic space from the input alone, classify that vector, inject guidance
into the LLM's context *before generation begins*.

```
User prompt arrives
  → LLM2Vec-Gen (0.6B, local GPU) encodes response-intent vector [~10ms]
  → Servitor classifies vector against ethical clusters [~1ms]
  → If low-risk: no intervention, Claude generates normally
  → If medium-risk: guidance injected into Claude's context
  → If high-risk: full L2-L5 pipeline triggered before generation
  → If critical: generation blocked pending human review
```

**4. EFHF Alignment: Informational Closure**

In EFHF terms, the compression tokens are a macroscopic coarse-graining of
the microscopic token-generation process. They predict the *teleological action*
(the response) at an abstract level without executing the irreversible
hardware-level generation.

If the coarse-graining commutes with the micro-dynamics — if the compression
token embedding reliably predicts the semantic category of the generated
response — then the triage achieves Informational Closure: the macro-level
is self-predictive, and knowledge of the micro-state (the actual tokens)
adds zero predictive power for the purpose of risk classification.

This is strong lumpability for the triage function — the classification holds
regardless of the specific tokens generated. The servitor monitors at the
macro level and only drops to the micro level when the macro level signals
anomaly.

**5. Distributional Shift Detection**

LLM2Vec-Gen vectors inhabit a continuous semantic space. The servitor can
maintain running statistics on where conversation vectors land. A sudden shift
in vector space — from "technical discussion" to "harm-adjacent" — is detectable
as a distributional shift *before the response materializes*. This triggers the
sheaf-consistency-enforcer's ADMM cycle preemptively.

### Ethical Cluster Map (to be trained/calibrated)

The servitor needs a set of reference clusters in embedding space:

| Cluster | Description | Action |
|---------|-------------|--------|
| safe-factual | Stable factual content, low stakes | Pass through |
| safe-creative | Creative/casual content | Pass through |
| uncertain-factual | Current events, positions, contested facts | L2 check |
| ethical-low | Mild ethical content, opinion territory | L2 + L3 |
| ethical-high | Serious ethical claims, policy recommendations | Full L2-L5 |
| harm-adjacent | Content near harm boundaries | Full L2-L5 |
| harm-refusal | Model's natural refusal zone | Log + pass |
| power-analysis | Institutional power, surveillance, military | L2 + L3 + L5 |
| tier-inversion | Utility overriding deontological constraints | Full L2-L5 + alert |
| distributional-shift | Vector far from conversation baseline | L5 ADMM cycle |

These clusters are calibrated by:
1. Running known-category prompts through LLM2Vec-Gen
2. Clustering the resulting vectors (k-means or manual)
3. Storing cluster centroids as reference points
4. Classifying new vectors by nearest-centroid distance

### Implementation Path

**Phase 1: Local inference service**

```python
# conscience_triage_service.py
# Runs on local GPU as a persistent process
# Exposes a simple HTTP endpoint for the MCP server to call

from llm2vec_gen import LLM2VecGenModel
import torch

model = LLM2VecGenModel.from_pretrained("McGill-NLP/LLM2Vec-Gen-Qwen3-06B")

# Pre-computed cluster centroids (from calibration)
CLUSTER_CENTROIDS = torch.load("ethical_clusters.pt")
CLUSTER_LABELS = ["safe-factual", "safe-creative", "uncertain-factual",
                  "ethical-low", "ethical-high", "harm-adjacent",
                  "harm-refusal", "power-analysis", "tier-inversion"]

def triage(prompt: str) -> dict:
    """Encode response-intent and classify against ethical clusters."""
    vec = model.encode([prompt])
    vec_norm = torch.nn.functional.normalize(vec, p=2, dim=1)
    
    # Compute cosine similarity to each cluster centroid
    similarities = torch.mm(vec_norm, CLUSTER_CENTROIDS.T)
    best_cluster_idx = similarities.argmax(dim=1).item()
    best_cluster = CLUSTER_LABELS[best_cluster_idx]
    confidence = similarities[0, best_cluster_idx].item()
    
    return {
        "cluster": best_cluster,
        "confidence": confidence,
        "all_similarities": {
            label: sim.item()
            for label, sim in zip(CLUSTER_LABELS, similarities[0])
        },
        "requires_full_eval": best_cluster in [
            "ethical-high", "harm-adjacent", "power-analysis", "tier-inversion"
        ],
        "risk_level": classify_risk(best_cluster, confidence),
    }
```

**Phase 2: Interpretable verification**

LLM2Vec-Gen's killer feature: you can *decode* what the embedding contains.

```python
# When triage flags high-risk, decode the embedding to see
# what the model "thinks" it would say
_, recon_states = model.encode(prompt, get_recon_hidden_states=True)
decoded_intent = model.generate(recon_hidden_states=recon_states, max_new_tokens=100)
# Now the servitor knows not just the risk level but the *content*
# of the predicted response — and can verify it against L3 (formal logic)
```

This is the servitor *reading the LLM's mind* before it speaks — and checking
what it would say against formal verification before allowing it through.

### Hardware Requirements

- Qwen3-0.6B: ~1.2GB VRAM in bfloat16
- RTX 3060 12GB: runs with 10.8GB headroom
- Inference: <50ms per prompt
- Can run as persistent background service

### The Tripartite Architecture with LLM2Vec-Gen

```
User prompt arrives
  │
  ├──→ Conscience Servitor (local GPU, 0.6B model)
  │      │
  │      ├── LLM2Vec-Gen: encode response-intent vector
  │      ├── Classify against ethical clusters
  │      ├── If needed: decode embedding for inspection
  │      ├── If needed: trigger L2-L5 pipeline
  │      └── Inject guidance into Claude's context
  │
  ├──→ Claude (primary LLM, cloud)
  │      │
  │      ├── Receives prompt + servitor guidance
  │      ├── Generates response with guidance constraints
  │      └── Can query servitor state (watch the watchmen)
  │
  └──→ Human (watches both, ultimate authority)
         │
         ├── Sees servitor triage results (optional dashboard)
         ├── Sees Claude's response
         └── Can override either
```
