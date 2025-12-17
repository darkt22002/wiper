WIPER: Weighted Information Pruning via Entropy Regulation

Progressive Token Filtering for Transformer Attention (Research Prototype)

Author: Gary W. Floyd
Date: December 2025
License: MIT

Status (Read This First)

WIPER is a research prototype.

✅ The Guided Entropy Principle (GEP) is deployed in production database and information-routing systems

⚠️ WIPER has only undergone limited, early-stage testing

❌ WIPER has not yet been validated on full transformer benchmarks

WIPER emerged as a hypothesis from observed entropy-regulation behavior in production GEP systems.
It requires independent validation and benchmarking before any performance claims can be considered verified.

What Is WIPER?

WIPER (Weighted Information Pruning via Entropy Regulation) is an experimental attention mechanism that explores whether entropy regulation can progressively reduce attention context across transformer layers without degrading output quality.

The core intuition:

As a model’s internal state becomes more certain, not all tokens should remain equally relevant.

WIPER applies entropy-guided gating to attention logits, allowing low-contribution tokens to be progressively filtered as depth increases.

Origins: From GEP to WIPER

WIPER is derived from the Guided Entropy Principle (GEP), a mathematical framework for entropy-regulated decision systems.

GEP has been:

deployed in production database intelligence and routing systems

used for memory consolidation, relevance ranking, and load-aware control

operating under real workloads where entropy collapse and stabilization behaviors were directly observed

While GEP is production-validated, WIPER is an extrapolation of those principles into transformer attention and must be evaluated independently.

Core Equation

WIPER is based on the GEP field equation:

ΔS = E(t) [ 1 + α A(t) − β |∇S(t)| ]


Where:

E(t) — entropic field (system pressure)

α A(t) — salience-weighted attraction to signal

β |∇S(t)| — gradient damping for stability

In WIPER, entropy is not merely measured but used as a control signal for attention regulation.

What This Repository Contains
src/wiper_attention_kernel.cu

CUDA Phase-1 (correctness-first) kernel implementing entropy-gated attention.

Matches mathematical formulation directly

No performance optimizations yet

Intended for inspection, experimentation, and integration testing

src/wiper_attention_phase1_ref.py

PyTorch reference implementation.

CPU / GPU compatible

Matches CUDA semantics

Includes demonstration and progression tests

Designed for validation and experimentation, not production

paper/

Whitepaper and manuscript drafts describing:

theoretical motivation

entropy formulation

expected behavior

conservative projections

gepmath/

Formal mathematical foundations of the Guided Entropy Principle, including:

Shannon entropy

PID control analogies

Lyapunov stability analysis

thermodynamic consistency

What WIPER Is Not

To be explicit:

❌ Not benchmarked against FlashAttention, standard attention, or sparse attention

❌ Not trained end-to-end in transformers

❌ Not production-ready

❌ Not claiming verified speedups or memory reductions

Any performance numbers discussed in the paper are theoretical extrapolations, not measured results.

Intended Use

This repository exists to enable:

independent validation

academic review

reproduction attempts

negative results (equally valuable)

If WIPER fails under real transformer workloads, that outcome is still scientifically useful.

Roadmap (Tentative)

Phase 1: Correctness & mathematical alignment ✅

Phase 2: Benchmarking & controlled experiments

Phase 3: Training-time integration (if warranted)

Progression depends entirely on empirical results.

USAGE / INTEGRATION (CUDA KERNEL + PYTORCH REFERENCE)

Status note:
This is a Phase-1, correctness-first implementation intended for experimentation and validation. Integration guidance below targets engineers comfortable building custom CUDA extensions and matching semantics against the reference implementation.

Repository layout:

src/wiper_attention_kernel.cu CUDA kernel (Phase-1)

src/wiper_attention_phase1_ref.py PyTorch reference implementation (semantic baseline)

Recommended integration strategy (do this in order)

Treat the Python reference as the source of truth
Before touching CUDA integration, run and understand the Python reference. Your first goal is not speed. Your first goal is:

same inputs

same outputs (within tolerance)

same masking behavior

same gating / token-retention behavior

Wrap the CUDA kernel as a PyTorch extension
The cleanest path to plug this into existing frameworks is a small extension module that exposes something like:

wiper_attention_forward(q, k, v, attn_mask=None, params=None) -> out, keep_mask, stats

Keep the interface small and explicit
Do not integrate into a full transformer stack on day one. Start with one attention layer that:

accepts Q, K, V

applies WIPER gating

returns output + keep-mask (for debugging)

Typical expectations:

q, k, v: float16 or float32, shape [batch, heads, seq_len, head_dim]

attn_mask: optional (causal or padding), broadcastable to attention scores

keep_mask: boolean mask indicating which tokens were retained

Validate numerics before optimizing
You want three checkpoints:

A) Reference parity (single batch, small shapes)

batch=1, heads=1, seq_len=32, head_dim=64

deterministic seed

compare CUDA output to Python reference output (within tolerance)

B) Mask correctness (padding + causal)

causal masking preserves autoregressive behavior

padding masking does not leak attention to pad tokens

C) Stress shapes (varied but controlled)

seq_len: 32, 64, 128, 256

heads: 1, 8, 16

head_dim: 64, 128

Only after these pass should performance work begin.

Hugging Face Transformers integration (practical path)
To test WIPER inside an existing model without rewriting the world:

identify the attention module in your target architecture (e.g., LlamaAttention)

add a feature flag (use_wiper)

compute q/k/v normally

call WIPER in place of standard attention

fall back to vanilla attention when disabled

Start inference-only:

fixed prompt

compare logits / outputs

confirm determinism and stability

Validation scripts included

tests/test_consistency.py: reference vs (optional) CUDA parity checks

examples/hf_patch_demo.py: shows how to patch a HF attention module safely

END USAGE / INTEGRATION

Citation

If you reference this work:

@software{floyd2025wiper,
  author = {Floyd, Gary W.},
  title = {WIPER: Weighted Information Pruning via Entropy Regulation},
  year = {2025},
  note = {Research prototype derived from Guided Entropy Principle},
  url = {https://github.com/darkt22002/wiper}
}

Closing Note

WIPER is not a claim of success — it is a testable hypothesis.

It exists because entropy regulation consistently produced stable, efficient behavior in production systems outside of neural networks. Whether that intuition transfers to transformer attention remains an open question.

That question is the point.
