#!/usr/bin/env python3
"""
WIPER Phase-1 validation harness.

Goal:
- Provide a reproducible place to compare the Python reference implementation
  to a future CUDA extension wrapper.

Current behavior:
- Always runs "reference-only" checks using src/wiper_attention_phase1_ref.py.
- If a CUDA extension named `wiper_cuda` is importable, it will also run
  parity checks (CUDA vs reference).

This keeps the repo honest: tests are useful today and become parity tests
once the CUDA wrapper exists.
"""

from __future__ import annotations

import os
import sys
import math
import time
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch


# ----------------------------
# Configuration
# ----------------------------

@dataclass
class Case:
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    dtype: torch.dtype
    device: str


def set_determinism(seed: int = 1234) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism knobs (best-effort)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_reference() -> Any:
    """
    Import the reference implementation module from src/.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_dir = os.path.join(repo_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    import wiper_attention_phase1_ref as ref  # type: ignore
    return ref


def try_load_cuda_ext() -> Optional[Any]:
    """
    Attempts to import an optional CUDA extension module.
    You will provide this later when you wrap the kernel (pybind/ATen extension).
    Expected API (suggested):
      wiper_cuda.forward(q, k, v, attn_mask=None, params=None) -> out, keep_mask, stats
    """
    try:
        import wiper_cuda  # type: ignore
        return wiper_cuda
    except Exception:
        return None


def make_qkv(case: Case) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(case.batch, case.heads, case.seq_len, case.head_dim,
                    device=case.device, dtype=case.dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


def causal_mask(seq_len: int, device: str) -> torch.Tensor:
    # shape [1, 1, seq, seq], True means "keep", False means "mask"
    m = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    m = torch.tril(m)
    return m.view(1, 1, seq_len, seq_len)


def pad_mask(seq_len: int, pad_to: int, device: str) -> torch.Tensor:
    """
    Creates a mask where first seq_len tokens are valid and the rest are padded.
    Output shape [1, 1, pad_to, pad_to] for simplicity.
    """
    assert pad_to >= seq_len
    valid = torch.zeros(pad_to, device=device, dtype=torch.bool)
    valid[:seq_len] = True

    # allow attention only among valid tokens (very strict)
    m = valid[:, None] & valid[None, :]
    return m.view(1, 1, pad_to, pad_to)


def max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().detach().cpu().item())


def run_reference_forward(refmod: Any,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          attn_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """
    This is intentionally loose because your reference file may expose different function names.
    Adjust the call here to match your actual reference API once confirmed.

    Common patterns:
      - refmod.wiper_attention(q, k, v, mask=attn_mask)
      - refmod.forward(q, k, v, attn_mask)
      - refmod.main() style script

    We try a few likely entry points.
    """
    # Try likely function names
    candidates = [
        ("wiper_attention", {"mask": attn_mask}),
        ("wiper_attention", {"attn_mask": attn_mask}),
        ("forward", {"mask": attn_mask}),
        ("forward", {"attn_mask": attn_mask}),
    ]

    for fn_name, kwargs in candidates:
        if hasattr(refmod, fn_name):
            fn = getattr(refmod, fn_name)
            # remove None kwargs
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            out = fn(q, k, v, **kwargs)
            # Normalize return structure
            if isinstance(out, tuple):
                # (out, keep_mask, stats?) variants
                result = {"out": out[0]}
                if len(out) > 1:
                    result["keep_mask"] = out[1]
                if len(out) > 2:
                    result["stats"] = out[2]
                return result
            return {"out": out}

    raise RuntimeError(
        "Could not find a usable forward function in src/wiper_attention_phase1_ref.py.\n"
        "Expected something like wiper_attention(...) or forward(...).\n"
        "Edit tests/test_consistency.py:run_reference_forward() to match your reference API."
    )


def run_cuda_forward(cudamod: Any,
                     q: torch.Tensor,
                     k: torch.Tensor,
                     v: torch.Tensor,
                     attn_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    if not hasattr(cudamod, "forward"):
        raise RuntimeError("CUDA module is importable but does not expose wiper_cuda.forward(...)")

    out = cudamod.forward(q, k, v, attn_mask)  # minimal signature for now
    if isinstance(out, tuple):
        result = {"out": out[0]}
        if len(out) > 1:
            result["keep_mask"] = out[1]
        if len(out) > 2:
            result["stats"] = out[2]
        return result
    return {"out": out}


def main() -> int:
    set_determinism(1234)
    refmod = load_reference()
    cudamod = try_load_cuda_ext()

    # Default test cases: small and readable
    cases = [
        Case(batch=1, heads=1, seq_len=32, head_dim=64, dtype=torch.float32, device="cpu"),
    ]

    # If CUDA available, add CUDA cases
    if torch.cuda.is_available():
        cases += [
            Case(batch=1, heads=1, seq_len=32, head_dim=64, dtype=torch.float32, device="cuda"),
            Case(batch=1, heads=8, seq_len=64, head_dim=64, dtype=torch.float16, device="cuda"),
        ]

    print("WIPER test harness")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA extension present: {cudamod is not None}")
    print("")

    for case in cases:
        print(f"[case] device={case.device} dtype={case.dtype} "
              f"B={case.batch} H={case.heads} S={case.seq_len} D={case.head_dim}")

        q, k, v = make_qkv(case)

        # Mask tests
        masks = [
            ("none", None),
            ("causal", causal_mask(case.seq_len, case.device)),
        ]

        for mname, mask in masks:
            print(f"  mask={mname}")

            # Reference
            t0 = time.time()
            ref_out = run_reference_forward(refmod, q, k, v, attn_mask=mask)
            torch.cuda.synchronize() if case.device == "cuda" else None
            t1 = time.time()

            print(f"    ref: out shape={tuple(ref_out['out'].shape)} time={t1 - t0:.4f}s")

            if cudamod is not None and case.device == "cuda":
                t2 = time.time()
                cu_out = run_cuda_forward(cudamod, q, k, v, attn_mask=mask)
                torch.cuda.synchronize()
                t3 = time.time()

                err = max_abs_err(ref_out["out"].float(), cu_out["out"].float())
                print(f"    cuda: out shape={tuple(cu_out['out'].shape)} time={t3 - t2:.4f}s max_abs_err={err:.6e}")

                # Loose tolerances for early-phase parity
                tol = 2e-3 if case.dtype in (torch.float16, torch.bfloat16) else 1e-5
                if err > tol:
                    raise AssertionError(f"Parity check failed: err={err} > tol={tol}")

        print("")

    print("OK: tests completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
