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
