// ============================================================================
// WIPER: Weighted Information Pruning via Entropy Regulation
// CUDA Kernel Implementation (Phase 1: Correctness)
// ============================================================================
//
// MIT License
//
// Copyright (c) 2025 Gary W. Floyd
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// ============================================================================
// ABOUT
// ============================================================================
//
// WIPER: Weighted Information Pruning via Entropy Regulation
//
// Implements the Guided Entropy Principle for transformer attention
// with progressive token filtering. Like a windshield wiper clearing
// the view, WIPER progressively removes noise while maintaining signal.
//
// Author: Gary W. Floyd
// Date: December 2025
// Email: gary.w.floyd@gmail.com
// Website: https://lumieasysnems.com
//
// Based on the Guided Entropy Principle field equation:
// ΔS = E(t)[1 + αA(t) - β|∇S(t)|]
//
// Where:
// - E(t) = Entropic field (system state pressure)
// - αA(t) = Salience-weighted correction (attraction to signal)
// - β|∇S(t)| = Gradient damping (stability control)
//
// PHASE 1 DESIGN:
// - One thread per query row (correct parallel decomposition)
// - Scalar accumulators (no redundant computation)
// - Keep-one enforcement (prevents NaN from all-wiped attention)
// - Clamped gate logits (prevents overflow)
// - Optimized salience (norm² without sqrt)
// - Per-block softmax update (fewer exponentials)
// - Numerically stable entropy calculation
//
// For more information, see:
// "WIPER: How Entropy Regulation Could Transform LLM Efficiency"
// Gary W. Floyd, December 2025
//
// ============================================================================

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

// Configuration constants
#define BLOCK_N 64      // Key/Value block size (for tiling across seq_len)
#define WARP_SIZE 32
#define MAX_D_MODEL 128 // Maximum head dimension
#define MAX_LAYERS 64   // For constant memory tau schedule

// WIPER configuration structure
struct WIPERConfig {
    float alpha;           // Salience correction strength (default: 0.7)
    float beta;            // Entropy damping strength (default: 0.3)
    float k_scale;         // Field activation steepness (default: 5.0)
    float c_threshold;     // Field activation threshold (default: 0.3)
    float* tau_schedule;   // Per-layer tau values [num_layers]
};

// Constant memory for tau schedule (read-only, cached)
__constant__ float c_tau_schedule[MAX_LAYERS];

// ============================================================================
// DEVICE FUNCTIONS
// ============================================================================

__device__ float calculate_entropic_field(
    float entropy_prev,
    float k_scale,
    float c_threshold
) {
    // Entropic Field E(t): Activates based on deviation from threshold
    // Sigmoid activation for smooth, bounded field strength [0, 1]
    return 1.0f / (1.0f + expf(-k_scale * (entropy_prev - c_threshold)));
}

__device__ float calculate_wiper_gate(
    float entropy_prev,
    float salience,
    float alpha,
    float beta,
    float k_scale,
    float c_threshold
) {
    // WIPER Gate Calculation: Combines three forces
    
    // 1. Entropic Field E(t) - Primary driver of regulation
    float E_t = calculate_entropic_field(entropy_prev, k_scale, c_threshold);
    
    // 2. Correction Term (Attraction to Signal) - Salience-weighted boost
    float correction = alpha * salience * 2.0f;
    
    // 3. Damping Term (Stability) - Entropy-weighted suppression
    float damping = beta * entropy_prev * 1.0f;
    
    // Combined gate logit
    float gate_logit = E_t * 5.0f + correction - damping;
    
    // CLAMP to prevent overflow in exp()
    gate_logit = fmaxf(-20.0f, fminf(20.0f, gate_logit));
    
    // Sigmoid to [0, 1] range
    return 1.0f / (1.0f + expf(-gate_logit));
}

// ============================================================================
// MAIN KERNEL: ONE THREAD PER QUERY ROW
// ============================================================================

__global__ void wiper_attention_kernel(
    const float* Q,          // Query: [batch, heads, seq_len, d_model]
    const float* K,          // Key: [batch, heads, seq_len, d_model]
    const float* V,          // Value: [batch, heads, seq_len, d_model]
    float* O,                // Output: [batch, heads, seq_len, d_model]
    float* S_state,          // Current layer entropy: [batch, heads, seq_len]
    const float* S_prev,     // Previous layer entropy: [batch, heads, seq_len]
    WIPERConfig config,      // WIPER configuration
    const int layer_idx,     // Current layer index
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int d_model
) {
    // ========================================================================
    // THREAD ASSIGNMENT: Each thread owns ONE query row
    // ========================================================================
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int q_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (q_idx >= seq_len) return;
    
    // ========================================================================
    // SHARED MEMORY: Load K and V in blocks
    // ========================================================================
    __shared__ float K_smem[BLOCK_N][MAX_D_MODEL];
    __shared__ float V_smem[BLOCK_N][MAX_D_MODEL];
    __shared__ float salience[BLOCK_N];
    
    // ========================================================================
    // REGISTER ALLOCATION: Per-query scalar accumulators
    // ========================================================================
    float Q_vec[MAX_D_MODEL];       // Query vector (loaded once)
    float acc[MAX_D_MODEL];         // Output accumulator
    float row_max = -INFINITY;      // Running max for numerical stability
    float row_sum = 0.0f;           // l = Σ exp(s'_i)
    float row_weighted_sum = 0.0f;  // Σ exp(s'_i) * s'_i for entropy
    
    // KEEP-ONE: Track best RAW score (before WIPER)
    float best_raw_score = -INFINITY;
    int best_k_idx = 0;
    
    // Initialize output accumulator
    #pragma unroll
    for (int d = 0; d < MAX_D_MODEL; d++) {
        acc[d] = 0.0f;
    }
    
    // ========================================================================
    // LOAD QUERY VECTOR ONCE
    // ========================================================================
    int q_base = ((batch_idx * num_heads + head_idx) * seq_len + q_idx) * d_model;
    
    #pragma unroll
    for (int d = 0; d < d_model; d++) {
        Q_vec[d] = Q[q_base + d];
    }
    
    // ========================================================================
    // GET PREVIOUS ENTROPY AND TAU (from constant memory)
    // ========================================================================
    int entropy_idx = (batch_idx * num_heads + head_idx) * seq_len + q_idx;
    float entropy_prev = S_prev[entropy_idx];
    float tau = c_tau_schedule[layer_idx];  // Read from constant memory (cached)
    
    // ========================================================================
    // ITERATE OVER KEY/VALUE BLOCKS
    // ========================================================================
    int num_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        
        // ====================================================================
        // COOPERATIVE LOAD: All threads load K, V, salience
        // ====================================================================
        int tid = threadIdx.x;
        int block_threads = blockDim.x;
        
        for (int n = tid; n < BLOCK_N; n += block_threads) {
            int k_idx = block_idx * BLOCK_N + n;
            
            if (k_idx < seq_len) {
                int kv_base = ((batch_idx * num_heads + head_idx) * seq_len + k_idx) * d_model;
                
                // ============================================================
                // OPTIMIZED SALIENCE: norm² / d_model (no sqrt)
                // Salience measures token importance - only monotonicity matters
                // Computed in fp32 even if K is fp16/bf16
                // ============================================================
                float sal_sq = 0.0f;
                
                #pragma unroll
                for (int d = 0; d < d_model; d++) {
                    float k_val = K[kv_base + d];
                    K_smem[n][d] = k_val;
                    V_smem[n][d] = V[kv_base + d];
                    sal_sq += k_val * k_val;  // Accumulate while loading
                }
                
                // Normalize by d_model and clamp to reasonable range
                salience[n] = fminf(sal_sq / (float)d_model, 2.0f);
                
            } else {
                // Pad out-of-bounds with zeros
                #pragma unroll
                for (int d = 0; d < d_model; d++) {
                    K_smem[n][d] = 0.0f;
                    V_smem[n][d] = 0.0f;
                }
                salience[n] = 0.0f;
            }
        }
        __syncthreads();  // ONLY sync after loading
        
        // ====================================================================
        // PHASE 1: COMPUTE WIPED SCORES FOR ENTIRE BLOCK
        // ====================================================================
        float S_row[BLOCK_N];
        
        for (int n = 0; n < BLOCK_N; n++) {
            int k_idx = block_idx * BLOCK_N + n;
            
            if (k_idx >= seq_len) {
                S_row[n] = -INFINITY;
                continue;
            }
            
            // ----------------------------------------------------------------
            // RAW SCORE: Q · K / sqrt(d)
            // ----------------------------------------------------------------
            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < d_model; d++) {
                score += Q_vec[d] * K_smem[n][d];  // Q loaded once, no global access
            }
            score /= sqrtf((float)d_model);
            
            // ----------------------------------------------------------------
            // TRACK BEST RAW SCORE (before WIPER, for keep-one)
            // ----------------------------------------------------------------
            if (score > best_raw_score) {
                best_raw_score = score;
                best_k_idx = k_idx;
            }
            
            // ----------------------------------------------------------------
            // WIPER GATE CALCULATION
            // ----------------------------------------------------------------
            float gate = calculate_wiper_gate(
                entropy_prev,
                salience[n],
                config.alpha,
                config.beta,
                config.k_scale,
                config.c_threshold
            );
            
            // ----------------------------------------------------------------
            // THE WIPER: Apply threshold + log-space bias
            // Like a wiper sweep - tokens below threshold get cleared
            // ----------------------------------------------------------------
            if (gate < tau) {
                // Below threshold → wipe this token (zero attention weight)
                S_row[n] = -INFINITY;
            } else {
                // Above threshold → keep and apply gate in log-space
                S_row[n] = score + logf(gate + 1e-8f);
            }
        }
        
        // ====================================================================
        // PHASE 2: FIND BLOCK MAX (only finite scores)
        // ====================================================================
        float m_block = -INFINITY;
        for (int n = 0; n < BLOCK_N; n++) {
            if (isfinite(S_row[n])) {
                m_block = fmaxf(m_block, S_row[n]);
            }
        }
        
        // ====================================================================
        // PHASE 3: SINGLE RESCALE FOR THIS BLOCK
        // ====================================================================
        if (isfinite(m_block)) {
            float m_new = fmaxf(row_max, m_block);
            float scale = expf(row_max - m_new);
            
            #pragma unroll
            for (int d = 0; d < d_model; d++) {
                acc[d] *= scale;
            }
            row_sum *= scale;
            row_weighted_sum *= scale;
            
            row_max = m_new;
            
            // ================================================================
            // PHASE 4: ACCUMULATE ALL TOKENS IN BLOCK
            // ================================================================
            for (int n = 0; n < BLOCK_N; n++) {
                if (!isfinite(S_row[n])) continue;  // Skip wiped tokens
                
                float s_prime = S_row[n] - row_max;
                float exp_score = expf(s_prime);
                
                row_sum += exp_score;
                
                // ENTROPY: Accumulate in shifted coordinates
                // DO NOT "optimize" this - the sign is correct
                row_weighted_sum += exp_score * s_prime;
                
                #pragma unroll
                for (int d = 0; d < d_model; d++) {
                    acc[d] += exp_score * V_smem[n][d];
                }
            }
        }
        
        // No __syncthreads() here - next iteration overwrites shared memory
    }
    
    // ========================================================================
    // KEEP-ONE ENFORCEMENT: Attend to best token if everything wiped
    // ========================================================================
    if (!isfinite(row_max) || row_sum < 1e-20f) {
        // Fallback: Attend to best_k_idx (token with highest raw score)
        // Even if WIPER would clear it, we keep it for numerical stability
        row_max = 0.0f;
        row_sum = 1.0f;
        row_weighted_sum = 0.0f;
        
        int kv_base = ((batch_idx * num_heads + head_idx) * seq_len + best_k_idx) * d_model;
        #pragma unroll
        for (int d = 0; d < d_model; d++) {
            acc[d] = V[kv_base + d];
        }
    }
    
    // ========================================================================
    // FINALIZE: Normalize output + calculate entropy
    // ========================================================================
    
    // Clamp l for additional numerical safety
    float l = fmaxf(row_sum, 1e-20f);
    
    // Write normalized output
    int out_base = ((batch_idx * num_heads + head_idx) * seq_len + q_idx) * d_model;
    #pragma unroll
    for (int d = 0; d < d_model; d++) {
        O[out_base + d] = acc[d] / l;
    }
    
    // Calculate entropy: H = log(l) - E[s']
    // This is the numerically stable formulation in shifted coordinates
    float E_sprime = row_weighted_sum / l;
    float final_entropy = logf(l) - E_sprime;
    
    // Write entropy state for next layer
    S_state[entropy_idx] = final_entropy;
}

// ============================================================================
// HOST FUNCTION: KERNEL LAUNCHER
// ============================================================================

void launch_wiper_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    float* d_S_state,
    const float* d_S_prev,
    WIPERConfig config,
    int layer_idx,
    int batch_size,
    int num_heads,
    int seq_len,
    int d_model,
    cudaStream_t stream
) {
    // Validate d_model
    if (d_model > MAX_D_MODEL) {
        printf("ERROR: d_model (%d) exceeds MAX_D_MODEL (%d)\n", d_model, MAX_D_MODEL);
        return;
    }
    
    // Grid: [batch, heads, ceil(seq_len / threads_per_block)]
    // Each block processes threads_per_block query rows
    int threads_per_block = 64;  // Matches BLOCK_N for clean utilization
    int num_blocks_seq = (seq_len + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(batch_size, num_heads, num_blocks_seq);
    dim3 block(threads_per_block);
    
    wiper_attention_kernel<<<grid, block, 0, stream>>>(
        d_Q, d_K, d_V, d_O,
        d_S_state, d_S_prev,
        config,
        layer_idx,
        batch_size, num_heads, seq_len, d_model
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in WIPER Attention Kernel: %s\n", cudaGetErrorString(err));
    }
}

// ============================================================================
// TAU SCHEDULE CREATION (uses constant memory)
// ============================================================================

void set_wiper_tau_schedule(int num_layers) {
    if (num_layers > MAX_LAYERS) {
        printf("ERROR: num_layers (%d) exceeds MAX_LAYERS (%d)\n", num_layers, MAX_LAYERS);
        return;
    }
    
    float h_tau[MAX_LAYERS];
    
    if (num_layers == 12) {
        // Phase-based schedule for 12-layer transformer
        // Like wiper sweeps - progressive clearing across layers
        // Exploration (0-2):   light sweep, keep ~90%
        // Refinement (3-5):    moderate sweep, keep ~45%
        // Focus (6-8):         strong sweep, keep ~20%
        // Convergence (9-11):  final sweep, keep ~10%
        float schedule[12] = {
            0.05f, 0.05f, 0.05f,  // Exploration
            0.15f, 0.25f, 0.35f,  // Refinement
            0.45f, 0.55f, 0.65f,  // Focus
            0.70f, 0.75f, 0.80f   // Convergence
        };
        memcpy(h_tau, schedule, 12 * sizeof(float));
    } else {
        // Generic linear ramp for other layer counts
        for (int i = 0; i < num_layers; i++) {
            float phase = (float)i / (float)(num_layers - 1);
            h_tau[i] = 0.05f + phase * 0.75f;  // 0.05 → 0.80
        }
    }
    
    // Copy to constant memory (cached, read-only)
    cudaMemcpyToSymbol(c_tau_schedule, h_tau, num_layers * sizeof(float));
}

// ============================================================================
// EXAMPLE USAGE
// ============================================================================

/*
int main() {
    // Model configuration (e.g., LLaMA-7B)
    int batch_size = 4;
    int num_heads = 32;
    int seq_len = 2048;
    int d_model = 128;     // 4096 / 32 heads
    int num_layers = 32;
    
    // Set WIPER tau schedule in constant memory
    set_wiper_tau_schedule(num_layers);
    
    // WIPER configuration
    WIPERConfig config;
    config.alpha = 0.7f;
    config.beta = 0.3f;
    config.k_scale = 5.0f;
    config.c_threshold = 0.3f;
    config.tau_schedule = nullptr;  // Not used, we use constant memory
    
    // Allocate device memory
    size_t qkv_size = batch_size * num_heads * seq_len * d_model * sizeof(float);
    size_t entropy_size = batch_size * num_heads * seq_len * sizeof(float);
    
    float *d_Q, *d_K, *d_V, *d_O;
    float *d_S_state, *d_S_prev;
    
    cudaMalloc(&d_Q, qkv_size);
    cudaMalloc(&d_K, qkv_size);
    cudaMalloc(&d_V, qkv_size);
    cudaMalloc(&d_O, qkv_size);
    cudaMalloc(&d_S_state, entropy_size);
    cudaMalloc(&d_S_prev, entropy_size);
    
    // Initialize S_prev with baseline entropy for first layer
    float baseline_entropy = logf((float)seq_len);
    std::vector<float> h_S_prev(batch_size * num_heads * seq_len, baseline_entropy);
    cudaMemcpy(d_S_prev, h_S_prev.data(), entropy_size, cudaMemcpyHostToDevice);
    
    // Process each layer with WIPER
    for (int layer = 0; layer < num_layers; layer++) {
        launch_wiper_attention(
            d_Q, d_K, d_V, d_O,
            d_S_state, d_S_prev,
            config, layer,
            batch_size, num_heads, seq_len, d_model,
            0  // default stream
        );
        
        // Swap entropy buffers for next layer
        std::swap(d_S_state, d_S_prev);
    }
    
    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_S_state);
    cudaFree(d_S_prev);
    
    return 0;
}
*/
