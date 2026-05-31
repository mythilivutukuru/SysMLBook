#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/* ─────────────────────────────────────────────────────────────────────────────
 * Network dimensions
 * ───────────────────────────────────────────────────────────────────────────── */
#define N       64      /* batch size                  */
#define D_IN    32      /* input feature dimension     */
#define D_H     64      /* hidden layer width          */
#define D_OUT   16      /* output dimension            */

#define NUM_STREAMS 4                  /* number of concurrent streams   */
#define CHUNK       (N / NUM_STREAMS)  /* rows per stream  (= 16)        */

/* ─────────────────────────────────────────────────────────────────────────────
 * Error-checking macro
 * ───────────────────────────────────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d  →  %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* ═════════════════════════════════════════════════════════════════════════════
 * KERNEL 1 — Fused Matrix-Multiply + ReLU
 *
 * C[i][j] = ReLU( sum_k A[i][k] * B[k][j]  +  bias[j] )
 *
 * A      : (rows_A, inner)   — a CHUNK-row slice of X or H
 * B      : (inner,  cols_B)  — full weight matrix
 * bias   : (cols_B,)
 * C      : (rows_A, cols_B)  — output slice
 * ═════════════════════════════════════════════════════════════════════════════*/
__global__ void matmul_relu_kernel(const float *A,
                                   const float *B,
                                   const float *bias,
                                         float *C,
                                   int rows_A,
                                   int inner,
                                   int cols_B)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (row >= rows_A || col >= cols_B) return;

    float acc = 0.0f;
    for (int k = 0; k < inner; ++k)
        acc += A[row * inner + k] * B[k * cols_B + col];

    acc += bias[col];
    C[row * cols_B + col] = fmaxf(acc, 0.0f);
}


/* ═════════════════════════════════════════════════════════════════════════════
 * KERNEL 2 — Plain Matrix-Multiply (no activation)
 *
 * C[i][j] = sum_k A[i][k] * B[k][j]  +  bias[j]
 * ═════════════════════════════════════════════════════════════════════════════*/
__global__ void matmul_kernel(const float *A,
                              const float *B,
                              const float *bias,
                                    float *C,
                              int rows_A,
                              int inner,
                              int cols_B)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (row >= rows_A || col >= cols_B) return;

    float acc = 0.0f;
    for (int k = 0; k < inner; ++k)
        acc += A[row * inner + k] * B[k * cols_B + col];

    C[row * cols_B + col] = acc + bias[col];
}


/* ═════════════════════════════════════════════════════════════════════════════
 * CPU REFERENCE IMPLEMENTATION — do not modify
 * ═════════════════════════════════════════════════════════════════════════════*/
static void mlp_cpu(const float *X,
                    const float *W1, const float *b1,
                    const float *W2, const float *b2,
                          float *H,
                          float *Y)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D_H; ++j) {
            float acc = b1[j];
            for (int k = 0; k < D_IN; ++k)
                acc += X[i * D_IN + k] * W1[k * D_H + j];
            H[i * D_H + j] = acc > 0.0f ? acc : 0.0f;
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D_OUT; ++j) {
            float acc = b2[j];
            for (int k = 0; k < D_H; ++k)
                acc += H[i * D_H + k] * W2[k * D_OUT + j];
            Y[i * D_OUT + j] = acc;
        }
    }
}


/* ═════════════════════════════════════════════════════════════════════════════
 * GPU LAUNCH WRAPPER — 4-stream version
 *
 * Design
 * ──────
 * The batch (N rows) is split into NUM_STREAMS equal chunks of CHUNK rows each.
 * Stream s owns rows  [s*CHUNK , (s+1)*CHUNK).
 *
 * Phase 0  (one-time, blocking):
 *   Copy the shared weight matrices W1, b1, W2, b2 to the device once via a
 *   simple synchronous copy.  These are read-only inputs shared by every
 *   stream, so copying them a single time is correct and avoids redundant
 *   transfers.
 *
 * Phase 1  (all 4 streams in flight simultaneously):
 *   For each stream s:
 *     a) cudaMemcpyAsync  X slice  →  d_X  + s*CHUNK*D_IN       [H2D]
 *     b) matmul_relu_kernel on the CHUNK-row slice                [compute]
 *        reads  d_X slice  +  d_W1 / d_b1  (already on device)
 *        writes d_H slice
 *     c) matmul_kernel on the CHUNK-row slice                    [compute]
 *        reads  d_H slice  +  d_W2 / d_b2
 *        writes d_Y slice
 *     d) cudaMemcpyAsync  d_Y slice  →  h_Y  + s*CHUNK*D_OUT    [D2H]
 *
 * Because operations within a single stream are ordered, steps (a)→(b)→(c)→(d)
 * are guaranteed to run in sequence for that stream.  Across streams the GPU
 * scheduler can overlap H2D copies of stream s+1 with kernel execution of
 * stream s, delivering real throughput gains on hardware with a copy engine.
 *
 * Phase 2:
 *   Synchronise and destroy all four streams.
 * ═════════════════════════════════════════════════════════════════════════════*/
static void mlp_forward_gpu(/* pinned host inputs */
                             const float *h_X,
                             const float *h_W1, const float *h_b1,
                             const float *h_W2, const float *h_b2,
                             /* pinned host output */
                                   float *h_Y,
                             /* pre-allocated device buffers */
                                   float *d_X,
                                   float *d_W1, float *d_b1,
                                   float *d_W2, float *d_b2,
                                   float *d_H,
                                   float *d_Y)
{
    /* ── Phase 0: copy shared weights once (synchronous, tiny) ─────────────── */
    /*
     * W1, b1, W2, b2 are identical for every chunk; copy them once before
     * any stream touches them.  Using blocking cudaMemcpy here is fine because
     * the weight transfers are tiny compared to the compute and they only
     * happen once per forward pass.
     */
    CUDA_CHECK( cudaMemcpy(d_W1, h_W1,
                           D_IN * D_H  * sizeof(float),
                           cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_b1, h_b1,
                           D_H         * sizeof(float),
                           cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_W2, h_W2,
                           D_H  * D_OUT * sizeof(float),
                           cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_b2, h_b2,
                           D_OUT        * sizeof(float),
                           cudaMemcpyHostToDevice) );

    /* ── Phase 1: create 4 streams, each processes one CHUNK of rows ────────── */
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; ++s)
        CUDA_CHECK( cudaStreamCreate(&streams[s]) );

    /* Block/grid dimensions — same shape used for both kernels.
     * The grid Y-dimension is sized to CHUNK rows (not N), since each
     * kernel invocation only touches CHUNK rows of its slice.             */
    dim3 block(16, 16);
    dim3 grid_h(  (D_H  + block.x - 1) / block.x,
                  (CHUNK + block.y - 1) / block.y );
    dim3 grid_out((D_OUT + block.x - 1) / block.x,
                  (CHUNK + block.y - 1) / block.y );

    for (int s = 0; s < NUM_STREAMS; ++s) {
        /* Byte / element offsets for this chunk */
        int  row_off   = s * CHUNK;                      /* row offset in full batch  */
        int  x_off     = row_off * D_IN;                 /* element offset into X     */
        int  h_off     = row_off * D_H;                  /* element offset into H     */
        int  y_off     = row_off * D_OUT;                /* element offset into Y     */

        /* Pointers into the contiguous device buffers for this chunk */
        float *d_X_s   = d_X + x_off;
        float *d_H_s   = d_H + h_off;
        float *d_Y_s   = d_Y + y_off;

        /* Pointers into pinned host buffers for this chunk */
        const float *h_X_s = h_X + x_off;
        float       *h_Y_s = h_Y + y_off;

        /* (a) H2D: transfer this chunk's rows of X */
        CUDA_CHECK( cudaMemcpyAsync(d_X_s, h_X_s,
                                    CHUNK * D_IN * sizeof(float),
                                    cudaMemcpyHostToDevice, streams[s]) );

        /* (b) Kernel 1: hidden layer H_s = ReLU(X_s @ W1 + b1)
         *     Reads d_X_s (just copied) and d_W1/d_b1 (copied in Phase 0).
         *     rows_A = CHUNK; the kernel sees a CHUNK×D_IN input.           */
        matmul_relu_kernel<<<grid_h, block, 0, streams[s]>>>(
            d_X_s, d_W1, d_b1, d_H_s, CHUNK, D_IN, D_H);
        CUDA_CHECK( cudaGetLastError() );

        /* (c) Kernel 2: output layer Y_s = H_s @ W2 + b2
         *     rows_A = CHUNK; the kernel sees a CHUNK×D_H input.            */
        matmul_kernel<<<grid_out, block, 0, streams[s]>>>(
            d_H_s, d_W2, d_b2, d_Y_s, CHUNK, D_H, D_OUT);
        CUDA_CHECK( cudaGetLastError() );

        /* (d) D2H: copy this chunk's Y back to pinned host memory */
        CUDA_CHECK( cudaMemcpyAsync(h_Y_s, d_Y_s,
                                    CHUNK * D_OUT * sizeof(float),
                                    cudaMemcpyDeviceToHost, streams[s]) );
    }

    /* ── Phase 2: wait for all streams, then release them ───────────────────── */
    for (int s = 0; s < NUM_STREAMS; ++s) {
        CUDA_CHECK( cudaStreamSynchronize(streams[s]) );
        CUDA_CHECK( cudaStreamDestroy(streams[s]) );
    }
}


/* ═════════════════════════════════════════════════════════════════════════════
 * MAIN — do not modify
 * ═════════════════════════════════════════════════════════════════════════════*/
int main(void)
{
    size_t sz_X   = N * D_IN  * sizeof(float);
    size_t sz_W1  = D_IN * D_H  * sizeof(float);
    size_t sz_b1  = D_H         * sizeof(float);
    size_t sz_W2  = D_H  * D_OUT * sizeof(float);
    size_t sz_b2  = D_OUT        * sizeof(float);
    size_t sz_H   = N * D_H   * sizeof(float);
    size_t sz_Y   = N * D_OUT  * sizeof(float);

    float *h_X,  *h_W1, *h_b1, *h_W2, *h_b2, *h_H_cpu, *h_Y_cpu, *h_Y_gpu;
    CUDA_CHECK( cudaMallocHost(&h_X,     sz_X)  );
    CUDA_CHECK( cudaMallocHost(&h_W1,    sz_W1) );
    CUDA_CHECK( cudaMallocHost(&h_b1,    sz_b1) );
    CUDA_CHECK( cudaMallocHost(&h_W2,    sz_W2) );
    CUDA_CHECK( cudaMallocHost(&h_b2,    sz_b2) );
    CUDA_CHECK( cudaMallocHost(&h_H_cpu, sz_H)  );
    CUDA_CHECK( cudaMallocHost(&h_Y_cpu, sz_Y)  );
    CUDA_CHECK( cudaMallocHost(&h_Y_gpu, sz_Y)  );

    srand(42);
    for (int i = 0; i < N * D_IN;   ++i) h_X [i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < D_IN * D_H;  ++i) h_W1[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < D_H;          ++i) h_b1[i] = (rand() % 40  -  20) / 100.0f;
    for (int i = 0; i < D_H * D_OUT;  ++i) h_W2[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < D_OUT;         ++i) h_b2[i] = (rand() % 40  -  20) / 100.0f;

    mlp_cpu(h_X, h_W1, h_b1, h_W2, h_b2, h_H_cpu, h_Y_cpu);
    printf("[CPU] MLP forward pass complete.\n");

    float *d_X, *d_W1, *d_b1, *d_W2, *d_b2, *d_H, *d_Y;
    CUDA_CHECK( cudaMalloc(&d_X,  sz_X)  );
    CUDA_CHECK( cudaMalloc(&d_W1, sz_W1) );
    CUDA_CHECK( cudaMalloc(&d_b1, sz_b1) );
    CUDA_CHECK( cudaMalloc(&d_W2, sz_W2) );
    CUDA_CHECK( cudaMalloc(&d_b2, sz_b2) );
    CUDA_CHECK( cudaMalloc(&d_H,  sz_H)  );
    CUDA_CHECK( cudaMalloc(&d_Y,  sz_Y)  );

    mlp_forward_gpu(h_X,  h_W1, h_b1, h_W2, h_b2,
                    h_Y_gpu,
                    d_X,  d_W1, d_b1, d_W2, d_b2, d_H, d_Y);
    printf("[GPU] MLP forward pass complete.\n");

    printf("Checking correctness...\n");
    float max_err = 0.0f;
    int   fail_i  = -1;
    for (int i = 0; i < N * D_OUT; ++i) {
        float err = fabsf(h_Y_cpu[i] - h_Y_gpu[i]);
        if (err > max_err) { max_err = err; }
        if (err > 1e-3f && fail_i < 0) fail_i = i;
    }
    printf("Max absolute error: %f\n", max_err);
    if (fail_i < 0) {
        printf("PASS: GPU output matches CPU reference.\n");
    } else {
        printf("FAIL: first mismatch at index %d  cpu=%.6f  gpu=%.6f\n",
               fail_i, h_Y_cpu[fail_i], h_Y_gpu[fail_i]);
    }

    cudaFree(d_X);  cudaFree(d_W1); cudaFree(d_b1);
    cudaFree(d_W2); cudaFree(d_b2); cudaFree(d_H);  cudaFree(d_Y);
    cudaFreeHost(h_X);  cudaFreeHost(h_W1); cudaFreeHost(h_b1);
    cudaFreeHost(h_W2); cudaFreeHost(h_b2);
    cudaFreeHost(h_H_cpu); cudaFreeHost(h_Y_cpu); cudaFreeHost(h_Y_gpu);
    return 0;
}
