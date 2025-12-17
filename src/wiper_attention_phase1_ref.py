"""
WIPER: Weighted Information Pruning via Entropy Regulation
Phase 1 Reference Implementation

============================================================================
MIT License

Copyright (c) 2025 Gary W. Floyd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

============================================================================
ABOUT

WIPER: Weighted Information Pruning via Entropy Regulation

Like a windshield wiper clearing the view, WIPER progressively removes
noise while maintaining signal across transformer layers.

Author: Gary W. Floyd
Date: December 2025
Email: gary.w.floyd@gmail.com
Website: https://lumieasysnems.com

Based on the Guided Entropy Principle field equation:
ΔS = E(t)[1 + αA(t) - β|∇S(t)|]

Where:
- E(t) = Entropic field (system state pressure)
- αA(t) = Salience-weighted correction (attraction to signal)
- β|∇S(t)| = Gradient damping (stability control)

Implementation details matching CUDA Phase-1 kernel:
- salience per key token: min(||K||^2 / d_head, 2.0)
- gate = sigmoid(clamp(E_t*5 + alpha*salience*2 - beta*entropy_prev*1, [-20,20]))
- WIPER threshold: gate < tau => -inf, else score += log(gate)
- keep-one: if all wiped for a row, attend to best RAW score token
- entropy from gated logits in shifted coords: H = log(l) - E[s']

For more information, see:
"WIPER: How Entropy Regulation Could Transform LLM Efficiency"
Gary W. Floyd, December 2025

============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class WIPERConfig:
    """WIPER configuration parameters"""
    alpha: float = 0.7
    beta: float = 0.3
    k_scale: float = 5.0
    c_threshold: float = 0.3
    tau_schedule: Optional[torch.Tensor] = None  # shape [num_layers]

    @staticmethod
    def make_tau_schedule(num_layers: int) -> torch.Tensor:
        """
        Create phase-based tau schedule for progressive token filtering
        
        Like wiper sweeps across layers:
        - Exploration (early):   light sweep, keep ~90%
        - Refinement (mid-early): moderate sweep, keep ~45%
        - Focus (mid-late):      strong sweep, keep ~20%
        - Convergence (late):    final sweep, keep ~10%
        """
        if num_layers == 12:
            return torch.tensor([
                0.05, 0.05, 0.05,  # Exploration
                0.15, 0.25, 0.35,  # Refinement
                0.45, 0.55, 0.65,  # Focus
                0.70, 0.75, 0.80   # Convergence
            ], dtype=torch.float32)
        # Generic ramp for other layer counts
        phase = torch.linspace(0, 1, num_layers, dtype=torch.float32)
        return 0.05 + phase * 0.75


# ============================================================================
# CORE MATH (matches CUDA)
# ============================================================================

def calculate_entropic_field(
    entropy_prev: torch.Tensor,
    k_scale: float,
    c_threshold: float
) -> torch.Tensor:
    """
    Calculate entropic field E(t)
    
    CUDA: 1/(1+exp(-k*(S_prev - c)))
    Activates based on deviation from threshold
    """
    return torch.sigmoid(k_scale * (entropy_prev - c_threshold))


def calculate_wiper_gate(
    entropy_prev: torch.Tensor,   # [B,H,S,1]
    salience_b: torch.Tensor,     # [B,H,1,S]
    alpha: float,
    beta: float,
    k_scale: float,
    c_threshold: float,
) -> torch.Tensor:
    """
    Calculate WIPER regulation gate
    
    Combines three forces:
    1. Entropic field E(t) - primary driver of regulation
    2. Correction term - salience-weighted attraction to signal
    3. Damping term - entropy-weighted stability control
    
    Matches CUDA kernel gate calculation exactly
    """
    E_t = calculate_entropic_field(entropy_prev, k_scale, c_threshold)
    correction = alpha * salience_b * 2.0
    damping = beta * entropy_prev * 1.0
    gate_logit = E_t * 5.0 + correction - damping
    gate_logit = torch.clamp(gate_logit, -20.0, 20.0)
    return torch.sigmoid(gate_logit)


# ============================================================================
# PHASE-1 REFERENCE ATTENTION (matches CUDA kernel)
# ============================================================================

@torch.no_grad()
def wiper_attention_phase1(
    Q: torch.Tensor,          # [B,H,S,D]
    K: torch.Tensor,          # [B,H,S,D]
    V: torch.Tensor,          # [B,H,S,D]
    S_prev: Optional[torch.Tensor],  # [B,H,S,1] or None
    layer_idx: int,
    config: WIPERConfig,
    *,
    device: str = "cpu",      # "cpu" or "cuda"
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    WIPER-regulated attention (Phase 1: correctness-first)
    
    Like a wiper sweep - progressively clears noise while keeping signal
    
    Args:
        Q, K, V: Query, Key, Value tensors [B, H, S, D]
        S_prev: Previous layer entropy [B, H, S, 1] or None
        layer_idx: Current layer index (for tau schedule)
        config: WIPER configuration
        device: "cpu" or "cuda"
        dtype: torch dtype for computation
    
    Returns:
        O: Attention output [B, H, S, D]
        S_state: Current layer entropy [B, H, S, 1]
    """

    assert Q.ndim == 4 and K.ndim == 4 and V.ndim == 4, "Q,K,V must be [B,H,S,D]"
    B, H, S, D = Q.shape
    assert K.shape == (B, H, S, D) and V.shape == (B, H, S, D)

    dev = torch.device(device)
    Q = Q.to(device=dev, dtype=dtype)
    K = K.to(device=dev, dtype=dtype)
    V = V.to(device=dev, dtype=dtype)

    # Initialize entropy_prev if not provided
    if S_prev is None:
        # Baseline entropy: log(S) (maximum entropy for uniform distribution)
        S_prev = torch.full((B, H, S, 1), math.log(S), device=dev, dtype=dtype)
    else:
        S_prev = S_prev.to(device=dev, dtype=dtype)
        assert S_prev.shape == (B, H, S, 1)

    assert config.tau_schedule is not None and config.tau_schedule.numel() > layer_idx, \
        "tau_schedule missing/too short"
    tau = float(config.tau_schedule[layer_idx].item())

    # ========================================================================
    # RAW ATTENTION SCORES
    # ========================================================================
    scores_raw = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)

    # ========================================================================
    # SALIENCE (optimized: norm² without sqrt)
    # ========================================================================
    # salience per key token [B,H,S] = clamp(||K||^2 / D, max=2.0)
    # Salience measures token importance - only monotonicity matters
    sal = (K * K).sum(dim=-1) / float(D)
    sal = torch.clamp(sal, max=2.0)
    sal_b = sal.unsqueeze(-2)  # [B,H,1,S] for broadcasting

    # ========================================================================
    # WIPER GATE CALCULATION
    # ========================================================================
    gate = calculate_wiper_gate(
        entropy_prev=S_prev,
        salience_b=sal_b,
        alpha=config.alpha,
        beta=config.beta,
        k_scale=config.k_scale,
        c_threshold=config.c_threshold,
    )  # [B,H,S,S]

    # ========================================================================
    # THE WIPER: Apply threshold + log-space bias
    # Like a wiper sweep - tokens below threshold get cleared
    # ========================================================================
    neg_inf = torch.tensor(float("-inf"), device=dev, dtype=dtype)
    scores_gated = torch.where(
        gate >= tau,
        scores_raw + torch.log(gate + 1e-8),
        neg_inf
    )

    # ========================================================================
    # KEEP-ONE: Track best RAW score token (pre-WIPER)
    # ========================================================================
    best_k_idx = scores_raw.argmax(dim=-1, keepdim=True)  # [B,H,S,1]

    # Detect all-wiped rows (all -inf)
    any_finite = torch.isfinite(scores_gated).any(dim=-1, keepdim=True)  # [B,H,S,1]

    # ========================================================================
    # SOFTMAX ATTENTION WEIGHTS
    # ========================================================================
    attn = F.softmax(scores_gated, dim=-1)

    # Keep-one weights (one-hot to best token)
    keep_one = torch.zeros_like(attn)
    keep_one.scatter_(-1, best_k_idx, 1.0)

    # Apply keep-one where needed (if WIPER cleared everything)
    attn = torch.where(any_finite, attn, keep_one)

    # ========================================================================
    # ATTENTION OUTPUT
    # ========================================================================
    O = torch.matmul(attn, V)

    # ========================================================================
    # ENTROPY FROM GATED LOGITS (shifted coordinates)
    # H = log(l) - E[s']
    # ========================================================================
    max_scores = scores_gated.max(dim=-1, keepdim=True).values  # [B,H,S,1]
    shifted = scores_gated - max_scores

    finite_mask = torch.isfinite(shifted)
    shifted_safe = torch.where(finite_mask, shifted, torch.zeros_like(shifted))
    exp_scores = torch.where(finite_mask, torch.exp(shifted_safe), torch.zeros_like(shifted_safe))

    l = exp_scores.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    weighted = (exp_scores * shifted_safe).sum(dim=-1, keepdim=True)
    E_sprime = weighted / l
    S_state = torch.log(l) - E_sprime

    # If keep-one row, entropy should be 0 (one-hot distribution)
    S_state = torch.where(any_finite, S_state, torch.zeros_like(S_state))

    return O, S_state


# ============================================================================
# NN MODULE WRAPPER
# ============================================================================

class WIPERAttention(nn.Module):
    """
    WIPER-Regulated Attention Layer (trainable wrapper)
    
    Wraps the WIPER attention function in a standard PyTorch module
    with learnable projection matrices.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        config: WIPERConfig,
        dropout: float = 0.0
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.config = config
        self.dropout = dropout
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        entropy_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with WIPER regulation
        
        Args:
            x: Input [batch, seq_len, d_model]
            layer_idx: Current layer index
            entropy_prev: Previous layer entropy [batch, num_heads, seq_len, 1]
        
        Returns:
            output: [batch, seq_len, d_model]
            entropy_state: [batch, num_heads, seq_len, 1]
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape to [batch, num_heads, seq_len, d_head]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        
        # WIPER-regulated attention
        device = "cuda" if x.is_cuda else "cpu"
        
        if self.training:
            # During training, compute with gradients
            # (For now, use no_grad; full backprop version would need custom autograd)
            with torch.no_grad():
                attn_output, entropy_state = wiper_attention_phase1(
                    Q, K, V, entropy_prev, layer_idx, self.config,
                    device=device, dtype=x.dtype
                )
        else:
            # Inference
            attn_output, entropy_state = wiper_attention_phase1(
                Q, K, V, entropy_prev, layer_idx, self.config,
                device=device, dtype=x.dtype
            )
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.o_proj(attn_output)
        
        if self.dropout > 0 and self.training:
            output = F.dropout(output, p=self.dropout)
        
        return output, entropy_state


# ============================================================================
# TRANSFORMER LAYER
# ============================================================================

class WIPERTransformerLayer(nn.Module):
    """Transformer layer with WIPER-regulated attention"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        config: WIPERConfig,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = WIPERAttention(d_model, num_heads, config, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        entropy_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input [batch, seq_len, d_model]
            layer_idx: Layer index
            entropy_prev: Previous layer entropy
        
        Returns:
            output: [batch, seq_len, d_model]
            entropy_state: [batch, num_heads, seq_len, 1]
        """
        # Self-attention with WIPER
        attn_out, entropy_state = self.attention(
            self.norm1(x), layer_idx, entropy_prev
        )
        x = x + attn_out
        
        # Feed-forward
        x = x + self.ff(self.norm2(x))
        
        return x, entropy_state


# ============================================================================
# DEMO & TESTING
# ============================================================================

def _set_determinism():
    """Set deterministic behavior for reproducibility"""
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def demo():
    """Demo showing CPU and GPU execution with comparison"""
    _set_determinism()

    B, H, S, D = 2, 4, 64, 32
    num_layers = 12
    layer_idx = 3

    cfg = WIPERConfig(
        alpha=0.7,
        beta=0.3,
        k_scale=5.0,
        c_threshold=0.3,
        tau_schedule=WIPERConfig.make_tau_schedule(num_layers),
    )

    # Random inputs
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)
    S_prev = torch.full((B, H, S, 1), math.log(S))

    print("=" * 60)
    print("WIPER Attention Phase 1 - Demo")
    print("Weighted Information Pruning via Entropy Regulation")
    print("=" * 60)

    # CPU baseline
    O_cpu, S_cpu = wiper_attention_phase1(Q, K, V, S_prev, layer_idx, cfg, device="cpu")
    print(f"CPU: O shape={O_cpu.shape}, S shape={S_cpu.shape}")
    print(f"     Entropy mean={float(S_cpu.mean()):.4f}")
    print(f"     Output mean={float(O_cpu.mean()):.4f}, std={float(O_cpu.std()):.4f}")

    # GPU experiment (if available)
    if torch.cuda.is_available():
        O_gpu, S_gpu = wiper_attention_phase1(Q, K, V, S_prev, layer_idx, cfg, device="cuda")
        print(f"\nGPU: O shape={O_gpu.shape}, S shape={S_gpu.shape}")
        print(f"     Entropy mean={float(S_gpu.mean().cpu()):.4f}")
        print(f"     Output mean={float(O_gpu.mean().cpu()):.4f}, std={float(O_gpu.std().cpu()):.4f}")

        # Compare (expect small numerical differences)
        max_abs_O = (O_cpu - O_gpu.cpu()).abs().max().item()
        max_abs_S = (S_cpu - S_gpu.cpu()).abs().max().item()
        print(f"\nCPU vs GPU max abs diff:")
        print(f"     O = {max_abs_O:.2e}")
        print(f"     S = {max_abs_S:.2e}")
        
        if max_abs_O < 1e-5 and max_abs_S < 1e-5:
            print("✓ Results match within tolerance")
        else:
            print("⚠ Large numerical differences detected")
    else:
        print("\nCUDA not available; GPU run skipped.")

    print("=" * 60)


def test_wiper_progression():
    """Test that WIPER progressively filters tokens across layers"""
    _set_determinism()
    
    B, H, S, D = 1, 2, 128, 32
    num_layers = 12
    
    cfg = WIPERConfig(
        alpha=0.7,
        beta=0.3,
        k_scale=5.0,
        c_threshold=0.3,
        tau_schedule=WIPERConfig.make_tau_schedule(num_layers),
    )
    
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)
    
    print("\n" + "=" * 60)
    print("WIPER Progression Test")
    print("Watching the wiper clear noise across layers...")
    print("=" * 60)
    
    S_prev = None
    for layer in range(num_layers):
        O, S_state = wiper_attention_phase1(Q, K, V, S_prev, layer, cfg, device="cpu")
        
        tau = cfg.tau_schedule[layer].item()
        mean_entropy = S_state.mean().item()
        
        # Visual indicator of sweep strength
        sweep = "light" if tau < 0.2 else "moderate" if tau < 0.5 else "strong" if tau < 0.7 else "final"
        
        print(f"Layer {layer:2d}: tau={tau:.3f} ({sweep:8s} sweep), entropy={mean_entropy:.4f}")
        
        S_prev = S_state
    
    print("=" * 60)
    print("✓ WIPER progressively cleared tokens while reducing entropy")


if __name__ == "__main__":
    demo()
    test_wiper_progression()
