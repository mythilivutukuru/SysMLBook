#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iostream>

// ======================
// START OF STUDENT CODE
// ======================

/**
 * @brief Standard attention kernel
 *
 * @param Q   Query  [B, NH, N, D]
 * @param K   Key    [B, NH, N, D]
 * @param V   Value  [B, NH, N, D]
 * @param O   Output [B, NH, N, D] 
 * @param B     Batch size
 * @param NH    Number of heads
 * @param N     Sequence length
 * @param D     Head dimension (must be <= 32)
 */
__global__ void standard_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float*       O,
    int B,
    int NH,
    int N,
    int D
) {
    int x = threadIdx.x;
    int i = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;

    int base  = (b * NH + h) * N * D;
    float scale = sqrtf((float)D);

    __shared__ float sS[32];
    __shared__ float s_max;
    __shared__ float s_denom;

    // Phase 1: each thread x computes score(query_i, key_x) from global memory
    if (x < N) {
        float score = 0.f;
        for (int k = 0; k < D; k++)
            score += Q[base + i * D + k] * K[base + x * D + k];
        sS[x] = score / scale;
    }
    __syncthreads();

    // Phase 2: softmax (single thread)
    if (x == 0) {
        float mx = sS[0];
        for (int j = 1; j < N; j++)
            mx = fmaxf(mx, sS[j]);
        s_max = mx;

        float denom = 0.f;
        for (int j = 0; j < N; j++) {
            float e = expf(sS[j] - mx);
            sS[j]   = e;
            denom  += e;
        }
        s_denom = denom;

        for (int j = 0; j < N; j++)
            sS[j] /= denom;
    }
    __syncthreads();

    // Phase 3: weighted sum over value rows (lanes 0..D-1)
    if (x < D) {
        float out = 0.f;
        for (int j = 0; j < N; j++)
            out += sS[j] * V[base + j * D + x];
        O[base + i * D + x] = out;
    }
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
 * @param h_Q   Host Query  [B, NH, N, D]
 * @param h_K   Host Key    [B, NH, N, D]
 * @param h_V   Host Value  [B, NH, N, D]
 * @param h_O   Host Output [B, NH, N, D]  (written by this function)
 * @param B     Batch size
 * @param NH    Number of heads
 * @param N     Sequence length
 * @param D     Head dimension (must be <= 32)
 */
void gpu_standard_attention(
    const float* h_Q,
    const float* h_K,
    const float* h_V,
    float*       h_O,
    int B, int NH, int N, int D
) {
    size_t bytes = (size_t)B * NH * N * D * sizeof(float);

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_O, bytes);

    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_O, 0, bytes);

    // Grid : (N, NH, B) — one block per (batch, head, query-row)
    // Block: (32)       — covers both D <= 32 and N <= 32
    dim3 grid(N, NH, B);
    standard_attention_kernel<<<grid, 32>>>(d_Q, d_K, d_V, d_O, B, NH, N, D);

    cudaDeviceSynchronize();

    cudaMemcpy(h_O, d_O, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
}

// ============================================================
// CPU REFERENCE 
// ============================================================

/**
 * @brief Pure-C reference for full (non-causal) scaled dot-product
 *        attention.  
 */
void cpu_standard_attention(
    const float* Q,
    const float* K,
    const float* V,
    float*       O,
    int B, int NH, int N, int D
) {
    float scale = 1.0f / sqrtf((float)D);
    float* scores = (float*)malloc(N * N * sizeof(float));
    float* weights = (float*)malloc(N * N * sizeof(float));

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < NH; ++h) {
            int base = (b * NH + h) * N * D;

            // Naive matmul: scores = Q * K^T
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    float s = 0.f;
                    for (int x = 0; x < D; ++x)
                        s += Q[base + i*D + x] * K[base + j*D + x];
                    scores[i*N + j] = s * scale;
                }

            // Naive softmax: row-wise over scores -> weights
            for (int i = 0; i < N; ++i) {
                float row_max = -1e30f;
                for (int j = 0; j < N; ++j)
                    if (scores[i*N + j] > row_max)
                        row_max = scores[i*N + j];

                float row_sum = 0.f;
                for (int j = 0; j < N; ++j) {
                    weights[i*N + j] = expf(scores[i*N + j] - row_max);
                    row_sum += weights[i*N + j];
                }

                for (int j = 0; j < N; ++j)
                    weights[i*N + j] /= row_sum;
            }

            // Naive matmul: O = weights * V
            for (int i = 0; i < N; ++i)
                for (int x = 0; x < D; ++x) {
                    float out = 0.f;
                    for (int j = 0; j < N; ++j)
                        out += weights[i*N + j] * V[base + j*D + x];
                    O[base + i*D + x] = out;
                }
        }
    }

    free(scores);
    free(weights);
}
// ============================================================
// TESTING INFRASTRUCTURE
// ============================================================

/** Fill tensor with random floats in [-1, 1]. */
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
            break;
        }
    }
    return ok;
}

/**
 * @brief Run one test case and print PASS / FAIL.
 *
 * @param B   Batch size
 * @param NH  Number of heads
 * @param N   Sequence length  (must be <= 32)
 * @param D   Head dimension   (must be <= 32)
 */
static void run_test(int B, int NH, int N, int D)
{
    printf("=== Test: B=%d  NH=%d  N=%d  D=%d ===\n", B, NH, N, D);

    if (D > 32 || N > 32) {
        printf("  SKIPPED: kernel requires D <= 32 and N <= 32\n\n");
        return;
    }

    size_t n     = (size_t)B * NH * N * D;
    size_t bytes = n * sizeof(float);

    float* h_Q     = (float*)malloc(bytes);
    float* h_K     = (float*)malloc(bytes);
    float* h_V     = (float*)malloc(bytes);
    float* h_O_cpu = (float*)malloc(bytes);
    float* h_O_gpu = (float*)malloc(bytes);

    fill_random(h_Q, n);
    fill_random(h_K, n);
    fill_random(h_V, n);

    // CPU reference
    cpu_standard_attention(h_Q, h_K, h_V, h_O_cpu, B, NH, N, D);

    // GPU result
    gpu_standard_attention(h_Q, h_K, h_V, h_O_gpu, B, NH, N, D);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error: %s\n", cudaGetErrorString(err));
        printf("  Result: FAIL\n\n");
    } else {
        int pass = compare_tensors(h_O_cpu, h_O_gpu, n, 1e-4f);
        printf("  Result: %s\n\n", pass ? "PASS" : "FAIL");
    }

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O_cpu);
    free(h_O_gpu);
}

// ============================================================
// MAIN
// ============================================================

int main(void)
{
    srand(42);

    run_test(/*B=*/1, /*NH=*/1, /*N=*/4,  /*D=*/4);

    run_test(/*B=*/2, /*NH=*/4, /*N=*/16, /*D=*/15);

    run_test(/*B=*/2, /*NH=*/2, /*N=*/31, /*D=*/32);

    return 0;
}