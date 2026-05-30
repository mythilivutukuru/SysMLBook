#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
using std::max;

// ======================
// START OF STUDENT CODE
// ======================

/**
 * @brief Sliding-window causal self-attention kernel.
 *
 * Each thread is responsible for ONE output token position `t` in one
 * (batch, head) pair.  For token t the only keys/values that are
 * visible are positions  max(0, t-W+1) … t  (causal + window limit).
 *
 * @param Q    Query  [B, H, N, D]
 * @param K    Key    [B, H, N, D]
 * @param V    Value  [B, H, N, D]
 * @param Y    Output [B, H, N, D]  (written by this function)
 * @param B    Batch size
 * @param H    Number of attention heads
 * @param N    Sequence length
 * @param D    Head dimension
 * @param W    Window size (each token attends to at most W previous tokens)
 * @param scale  Pre-computed 1/sqrt(D)
 */
__global__ void sliding_window_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* Y,
    int B,
    int H,
    int N,
    int D,
    int W,
    float scale
) {
    //
    // TODO — Identify which (b, h, t) this thread handles.
    //
    //   The launch grid is:  grid(B, H, ceil(N/BLOCK)),  block(BLOCK)
    //   so the three grid dimensions map directly to b, h, and a tile
    //   of token indices.
    //
    //   Hint:
    //     b  = blockIdx.x
    //     h  = blockIdx.y
    //     t  = threadIdx.x + blockIdx.z * blockDim.x
    //
    //   After computing t, add an early-return guard so threads beyond
    //   the sequence length do nothing.
    //


    //
    // TODO — Determine the causal sliding window for token t.
    //     start_k = max(0, t - W + 1)
    //     end_k   = t
    //


    //
    // TODO — Pass 1: find the maximum attention score (for numerical
    //          stability of the subsequent softmax).
    //
    //   Initialise  max_score = -1e20f  (a very large negative number).
    //


    //
    // TODO — Pass 2: compute the normalisation denominator (sum of
    //          shifted exponentials).
    //
    //   Initialise  sum_exp = 0.0f.


    //
    // TODO — Pass 3: compute the output vector for token t.


}


// ====================
// END OF STUDENT CODE
// ====================

// ======================================================================
// TESTING INFRASTRUCTURE: PLEASE DON'T CHANGE ANYTHING BELOW THIS LINE.
// ======================================================================

/**
 * @brief Allocates device memory, copies data, launches the kernel,
 *        and copies results back.
 *
 * @param h_Q   Host Query  [B, H, N, D]
 * @param h_K   Host Key    [B, H, N, D]
 * @param h_V   Host Value  [B, H, N, D]
 * @param h_Y   Host Output [B, H, N, D]  (written by this function)
 * @param B     Batch size
 * @param H     Number of heads
 * @param N     Sequence length
 * @param D     Head dimension
 * @param W     Window size
 */
void gpu_sliding_window_attention(
    const float* h_Q,
    const float* h_K,
    const float* h_V,
    float*       h_Y,
    int B, int H, int N, int D, int W
) {
    size_t bytes = (size_t)B * H * N * D * sizeof(float);
    float scale  = 1.0f / sqrtf((float)D);

    float *d_Q, *d_K, *d_V, *d_Y;
    cudaMalloc((void**)&d_Q, bytes);
    cudaMalloc((void**)&d_K, bytes);
    cudaMalloc((void**)&d_V, bytes);
    cudaMalloc((void**)&d_Y, bytes);

    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_Y, 0, bytes);

    // Grid: (B, H, ceil(N / BLOCK))
    // Block: (BLOCK,) — one thread per token
    const int BLOCK = 128;
    dim3 grid(B, H, (N + BLOCK - 1) / BLOCK);
    dim3 block(BLOCK);

    sliding_window_attention_kernel<<<grid, block>>>(
        d_Q, d_K, d_V, d_Y,
        B, H, N, D, W, scale
    );

    cudaDeviceSynchronize();

    cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_Y);
}

// ============================================================
// CPU REFERENCE
// ============================================================

/**
 * @brief Pure-C reference implementation of sliding-window causal
 *        self-attention.
 */
void cpu_sliding_window_attention(
    const float* Q,
    const float* K,
    const float* V,
    float*       Y,
    int B, int H, int N, int D, int W
) {
    float scale = 1.0f / sqrtf((float)D);

    float* scores  = (float*)malloc(N * N * sizeof(float));
    float* weights = (float*)malloc(N * N * sizeof(float));

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            int ho = b * (H * N * D) + h * (N * D);

            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d)
                        dot += Q[ho + i * D + d] * K[ho + j * D + d];
                    scores[i * N + j] = dot * scale;
                }
            }

            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    bool causal  = (j <= i);
                    bool in_win  = (j >= i - W + 1);
                    if (!(causal && in_win))
                        scores[i * N + j] = -1e20f;
                }
            }

            for (int i = 0; i < N; ++i) {
                // find row max
                float row_max = -1e20f;
                for (int j = 0; j < N; ++j)
                    if (scores[i * N + j] > row_max)
                        row_max = scores[i * N + j];

                float row_sum = 0.0f;
                for (int j = 0; j < N; ++j) {
                    weights[i * N + j] = expf(scores[i * N + j] - row_max);
                    row_sum += weights[i * N + j];
                }

                for (int j = 0; j < N; ++j)
                    weights[i * N + j] /= row_sum;
            }

            for (int i = 0; i < N; ++i) {
                for (int d = 0; d < D; ++d) {
                    float val = 0.0f;
                    for (int j = 0; j < N; ++j)
                        val += weights[i * N + j] * V[ho + j * D + d];
                    Y[ho + i * D + d] = val;
                }
            }
        }
    }

    free(scores);
    free(weights);
}

// ============================================================
// TESTING INFRASTRUCTURE
// ============================================================

/** Fill tensor with small random floats in [-1, 1]. */
static void fill_random(float* data, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        data[i] = ((float)(rand() % 2001) - 1000.0f) / 1000.0f;
}

/**
 * @brief Element-wise comparison within tolerance.
 * @return 1 if all elements match, 0 otherwise.
 */
static int compare_tensors(const float* ref, const float* gpu,
                           size_t n, float tol)
{
    int ok = 1;
    for (size_t i = 0; i < n; ++i) {
        float diff = fabsf(ref[i] - gpu[i]);
        if (diff > tol) {
            printf("  MISMATCH at index %zu: cpu=%.6f  gpu=%.6f  diff=%.2e\n",
                   i, ref[i], gpu[i], diff);
            ok = 0;
            break;   // report first failure only
        }
    }
    return ok;
}

/**
 * @brief Run one test case.
 *
 * @param B  Batch size
 * @param H  Number of heads
 * @param N  Sequence length
 * @param D  Head dimension
 * @param W  Window size
 */
static void run_test(int B, int H, int N, int D, int W)
{
    printf("=== Test: B=%d  H=%d  N=%d  D=%d  W=%d ===\n",
           B, H, N, D, W);

    size_t n     = (size_t)B * H * N * D;
    size_t bytes = n * sizeof(float);

    float* h_Q     = (float*)malloc(bytes);
    float* h_K     = (float*)malloc(bytes);
    float* h_V     = (float*)malloc(bytes);
    float* h_Y_cpu = (float*)malloc(bytes);
    float* h_Y_gpu = (float*)malloc(bytes);

    fill_random(h_Q, n);
    fill_random(h_K, n);
    fill_random(h_V, n);

    // CPU reference
    cpu_sliding_window_attention(h_Q, h_K, h_V, h_Y_cpu, B, H, N, D, W);

    // GPU result
    gpu_sliding_window_attention(h_Q, h_K, h_V, h_Y_gpu, B, H, N, D, W);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error: %s\n", cudaGetErrorString(err));
        printf("  Result: FAIL\n\n");
    } else {
        int pass = compare_tensors(h_Y_cpu, h_Y_gpu, n, 1e-4f);
        printf("  Result: %s\n\n", pass ? "PASS" : "FAIL");
    }

    free(h_Q); 
    free(h_K); 
    free(h_V);
    free(h_Y_cpu); 
    free(h_Y_gpu);
}

// ============================================================
// MAIN
// ============================================================

int main(void)
{
    srand(42);
    run_test(/*B=*/1, /*H=*/1, /*N=*/8,  /*D=*/4,  /*W=*/3);
    run_test(/*B=*/2, /*H=*/4, /*N=*/32, /*D=*/16, /*W=*/32);
    run_test(/*B=*/1, /*H=*/2, /*N=*/130,/*D=*/8,  /*W=*/5);
    return 0;
}