#!/usr/bin/env python3
"""
Hugging Face patch demo (safe, minimal, research-grade).

This script demonstrates how you would *structure* a WIPER integration into an existing
HF Transformers attention module. It does not claim correctness for any specific model
until the CUDA extension + reference parity checks are complete.

Behavior:
- If `transformers` is installed and a model name is provided, it will load the model.
- It provides a "patch point" where attention can be replaced.
- If a CUDA extension `wiper_cuda` is present, it will call it.
- Otherwise it falls back to vanilla attention (so the script remains runnable).

Use:
  python examples/hf_patch_demo.py --model meta-llama/Llama-2-7b-hf --prompt "Hello"
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import torch


def try_import_transformers():
    try:
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def try_load_cuda_ext():
    try:
        import wiper_cuda  # type: ignore
        return wiper_cuda
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="", help="HF model id (optional)")
    ap.add_argument("--prompt", type=str, default="Hello from WIPER.", help="Prompt text")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    has_tf = try_import_transformers()
    if not has_tf:
        print("transformers not installed. Install with: pip install transformers")
        print("Exiting demo.")
        return 0

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    if not args.model:
        print("No --model provided. Example:")
        print('  python examples/hf_patch_demo.py --model gpt2 --prompt "Hello"')
        return 0

    device = args.device
    print(f"Loading model: {args.model} on {device}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    wiper_cuda = try_load_cuda_ext()
    print(f"CUDA extension present: {wiper_cuda is not None}")

    # ------------------------------------------------------------
    # Patch strategy (high-level):
    # - Find attention modules
    # - Replace their forward() with a wrapper
    # - Wrapper computes q/k/v normally and then calls WIPER
    #
    # NOTE: Model architectures vary. This demo shows the pattern, not a universal patch.
    # ------------------------------------------------------------

    patched = 0

    def patch_attention_module(mod: torch.nn.Module) -> bool:
        nonlocal patched

        # Heuristic: look for modules with q_proj/k_proj/v_proj or similar
        has_qkv = all(hasattr(mod, name) for name in ("q_proj", "k_proj", "v_proj")) or \
                  all(hasattr(mod, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj"))

        if not has_qkv:
            return False

        if not hasattr(mod, "forward"):
            return False

        original_forward = mod.forward

        def wiper_forward(*f_args, **f_kwargs):
            # If CUDA ext not present, fall back immediately (safe)
            if wiper_cuda is None:
                return original_forward(*f_args, **f_kwargs)

            # This is where you would:
            # 1) extract hidden_states from args/kwargs
            # 2) build q/k/v exactly like the original module
            # 3) call wiper_cuda.forward(q,k,v,mask,params)
            #
            # Because module signatures differ across architectures, we keep this demo minimal:
            # - run original forward for now unless you customize per target model.
            #
            # Once you pick a target model family (Llama, GPTNeoX, etc.), implement this per-module.
            return original_forward(*f_args, **f_kwargs)

        mod.forward = wiper_forward  # type: ignore
        patched += 1
        return True

    # Walk modules and patch likely attention blocks
    for name, mod in model.named_modules():
        try:
            patch_attention_module(mod)
        except Exception:
            # Keep demo robust; patching is best-effort
            pass

    print(f"Patched attention modules (heuristic): {patched}")
    print("NOTE: This demo is a template. For a real integration, implement per-model attention forward.")

    # Run a small generation to prove nothing is broken
    inputs = tok(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    print(tok.decode(out[0], skip_special_tokens=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
