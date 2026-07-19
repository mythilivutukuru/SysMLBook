#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iostream>

// ======================
// START OF STUDENT CODE
// ======================

constexpr int D_MAX = 128;
constexpr int BC_MAX = 32;  
constexpr int BR_MAX = 32; 

/**
 * @brief Flash Attention forward pass kernel.
 *
 * @param Q   Query  [B, NH, N, D]
 * @param K   Key    [B, NH, N, D]
 * @param V   Value  [B, NH, N, D]
 * @param O   Output [B, NH, N, D]  (written by this function)
 * @param B     Batch size
 * @param NH    Number of heads
 * @param N     Sequence length
 * @param D     Head dimension
 * @param Tc    Tile size for keys/values
 * @param Tr    Tile size for queries
 * @param Bc    Block size for keys/values
 * @param Br    Block size for queries
 * @param softmax_scale  Scaling factor for attention scores (usually 1/sqrt(D))
 * @param l     [B, NH, N] (running sum of exponentiated scores per query row, for numerical stability)
 * @param m     [B, NH, N] (running max of scores per query row, for numerical stability)
 * @param O     [B, NH, N, D] (output, updated incrementally)
 */
__global__ void flash_forward_kernel(
    const float* Q,   // [B, NH, N, D]
    const float* K,   // [B, NH, N, D]
    const float* V,   // [B, NH, N, D]
    int B, int NH,
    int N, int D,
    int Tc, int Tr,
    int Bc, int Br,
    float softmax_scale,
    float* l,         // [B, NH, N]
    float* m,         // [B, NH, N]
    float* O          // [B, NH, N, D]
)
{
    // 
    // Basic thread / block indices
    // 
    int tx = threadIdx.x;   // thread index within the block (0 .. Br-1)
    if (tx >= Br) return;

    int b = blockIdx.x;     // batch index
    int h = blockIdx.y;     // head index

    // 
    // Shared memory layout
    //   Qi : current query tile          [Br x D]
    //   Kj : current key tile            [Bc x D]
    //   Vj : current value tile          [Bc x D]
    //   S  : attention scores (Qi x Kj)  [Br x Bc]
    // 
    __shared__ float Qi[BR_MAX * D_MAX];
    __shared__ float Kj[BC_MAX * D_MAX];
    __shared__ float Vj[BC_MAX * D_MAX];
    __shared__ float S[BR_MAX * BC_MAX];

    // 
    // Base offsets into global memory for this (batch, head) pair
    // 
    int qkv_base = ((b * NH + h) * N * D);
    int lm_base  = ((b * NH + h) * N);

    // 
    // Outer loop: iterate over KEY / VALUE tiles  (j = 0 .. Tc-1)
    // 
    for (int j = 0; j < Tc; j++) {

        // 
        // Step 1: Load the j-th Key and Value tiles into shared memory.
        //         Each thread loads one row (tx) across all D columns.
        // 
    

        // 
        // Inner loop: iterate over QUERY tiles  (i = 0 .. Tr-1)
        // 
        for (int i = 0; i < Tr; i++) {

            // 
            // Step 2: Load the i-th Query tile into shared memory.
            // 
            
            // 
            // Running statistics for this query row (from global mem)
            // 
            int row = lm_base + i * Br + tx;

            float row_m_prev = m[row];   // previous running max
            float row_l_prev = l[row];   // previous running sum
            float row_m      = -INFINITY; // local max for this tile

            // 
            // Step 3: Compute raw attention scores S = Qi * Kj^T
            //         and find the local row maximum (row_m).
            // 
            for (int y = 0; y < Bc; y++) {
                float sum = 0.f;
                
                // TODO: apply softmax_scale to sum
                // TODO: store scaled score into S[tx * Bc + y]
                // TODO: update row_m with fmaxf
            }

            // 
            // Step 4: Subtract row_m, exponentiate, and sum (row_l).
            // 
            float row_l = 0.f;
            for (int y = 0; y < Bc; y++) {
                // TODO: set S[tx * Bc + y] = expf(S[tx * Bc + y] - row_m)
                // TODO: accumulate S[tx * Bc + y] into row_l
            }

            // 
            // Step 5: Merge current tile statistics with running stats
            //         using the online-softmax update rule.
            // 
            // TODO: compute row_m_new = max(row_m_prev, row_m)
            // TODO: compute row_l_new using the rescaling formula:
            //         row_l_new = exp(row_m_prev - row_m_new) * row_l_prev
            //                   + exp(row_m      - row_m_new) * row_l
            float row_m_new = /* TODO */ 0.f;
            float row_l_new = /* TODO */ 0.f;

            // 
            // Step 6: Update the output O with the new partial sum.
            //         Rescale the previous O contribution and add the
            //         new weighted V contribution, then normalise.
            // 
            for (int x = 0; x < D; x++) {
                float pv = 0.f;
                for (int y = 0; y < Bc; y++) {
                    // TODO: accumulate S[tx * Bc + y] * Vj[y * D + x] into pv
                }

                int o_idx    = qkv_base + (i * Br + tx) * D + x;
                float prev_o = O[o_idx];

                // TODO: update O[o_idx] with the incremental Flash Attention
                //       output formula (rescale prev_o and add pv, divide by row_l_new)
                O[o_idx] = /* TODO */ prev_o;
            }

            // 
            // Step 7: Write updated running statistics back to global memory.
            // 
            // TODO: store row_m_new into m[row]
            // TODO: store row_l_new into l[row]
        }
        
    }
}

// ====================
// END OF STUDENT CODE
// ====================

// ======================================================================
// TESTING INFRASTRUCTURE: PLEASE DON'T CHANGE ANYTHING BELOW THIS LINE.
// ======================================================================

/**
 * @brief Allocates device memory, copies data, launches the Flash Attention
 *        kernel, and copies results back to the host.
 *
 * @param h_Q   Host Query  [B, NH, N, D]
 * @param h_K   Host Key    [B, NH, N, D]
 * @param h_V   Host Value  [B, NH, N, D]
 * @param h_O   Host Output [B, NH, N, D]  (written by this function)
 * @param B     Batch size
 * @param NH    Number of heads
 * @param N     Sequence length
 * @param D     Head dimension
 */
void gpu_flash_attention(
    const float* h_Q,
    const float* h_K,
    const float* h_V,
    float*       h_O,
    int B, int NH, int N, int D
) {
    const int Bc = 32;
    const int Br = 32;
    const int Tc = (N + Bc - 1) / Bc;
    const int Tr = (N + Br - 1) / Br;
    const float scale = 1.0f / sqrtf((float)D);

    size_t size_qkv = (size_t)B * NH * N * D * sizeof(float);
    size_t size_lm  = (size_t)B * NH * N * sizeof(float);

    float *d_Q, *d_K, *d_V, *d_O, *d_l, *d_m;
    cudaMalloc(&d_Q, size_qkv);
    cudaMalloc(&d_K, size_qkv);
    cudaMalloc(&d_V, size_qkv);
    cudaMalloc(&d_O, size_qkv);
    cudaMalloc(&d_l, size_lm);
    cudaMalloc(&d_m, size_lm);

    cudaMemcpy(d_Q, h_Q, size_qkv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size_qkv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size_qkv, cudaMemcpyHostToDevice);
    cudaMemset(d_O, 0, size_qkv);
    cudaMemset(d_l, 0, size_lm);
    cudaMemset(d_m, 0xFF, size_lm);  // Initialize m to -inf (NaN pattern -> -inf via fmaxf)

    dim3 grid(B, NH);
    dim3 block(Br);

    flash_forward_kernel<<<grid, block>>>(
        d_Q, d_K, d_V,
        B, NH, N, D,
        Tc, Tr, Bc, Br,
        scale, d_l, d_m, d_O
    );

    cudaDeviceSynchronize();

    cudaMemcpy(h_O, d_O, size_qkv, cudaMemcpyDeviceToHost);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_l); cudaFree(d_m);
}

// ========================
// CPU REFERENCE
// ========================

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

// ========================
// TESTING INFRASTRUCTURE
// ========================

static void fill_random(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i)
        data[i] = ((float)(rand() % 2001) - 1000.0f) / 1000.0f;
}

static int compare_tensors(const float* ref, const float* gpu, size_t n, float tol) {
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
 * @brief Run one test case.
 *
 * @param B   Batch size
 * @param NH  Number of heads
 * @param N   Sequence length
 * @param D   Head dimension
 */
static void run_test(int B, int NH, int N, int D) {
    printf("=== Test: B=%d  NH=%d  N=%d  D=%d ===\n", B, NH, N, D);

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
    gpu_flash_attention(h_Q, h_K, h_V, h_O_gpu, B, NH, N, D);

    // Check CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error: %s\n", cudaGetErrorString(err));
        printf("  Result: FAIL\n\n");
    } else {
        int pass = compare_tensors(h_O_cpu, h_O_gpu, n, 1e-3f);
        printf("  Result: %s\n\n", pass ? "PASS" : "FAIL");
    }

    free(h_Q); free(h_K); free(h_V);
    free(h_O_cpu); free(h_O_gpu);
}

// ========================
// MAIN
// ========================

int main(void) {
    srand(42);

    // Test 1: Tiny case — easy to debug
    run_test(/*B=*/1, /*NH=*/1, /*N=*/32, /*D=*/32);

    // Test 2: Moderate batch and heads
    run_test(/*B=*/2, /*NH=*/4, /*N=*/64, /*D=*/64);

    // Test 3: Larger sequence, multiple batches
    run_test(/*B=*/4, /*NH=*/8, /*N=*/128, /*D=*/32);

    return 0;
}
