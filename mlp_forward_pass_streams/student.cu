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
 * Convenience macro — wraps every CUDA call and prints file/line on error
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
 * Computes:  C[i][j] = ReLU( sum_k A[i][k] * B[k][j]  +  bias[j] )
 *
 * Arguments:
 *   A      — input slice,   row-major, shape (rows_A, inner)
 *             rows_A = CHUNK  (a contiguous subset of the full batch)
 *   B      — weight matrix, row-major, shape (inner,  cols_B)  [shared, full]
 *   bias   — bias vector,               shape (cols_B,)
 *   C      — output slice,  row-major,  shape (rows_A, cols_B)
 *   rows_A — number of rows in this slice (= CHUNK)
 *   inner  — contracted dimension (= D_IN for layer 1)
 *   cols_B — number of columns in B (= D_H  for layer 1)
 *
 * Thread mapping (suggested):
 *   threadIdx.x + blockIdx.x * blockDim.x  →  column index j
 *   threadIdx.y + blockIdx.y * blockDim.y  →  row    index i  (within slice)
 * ═════════════════════════════════════════════════════════════════════════════*/
__global__ void matmul_relu_kernel(const float *A,
                                   const float *B,
                                   const float *bias,
                                         float *C,
                                   int rows_A,
                                   int inner,
                                   int cols_B)
{
    /* TODO 1 — Compute the row and column index this thread is responsible for.
     *
     *   int col = ...;
     *   int row = ...;
     */

    /* TODO 2 — Guard: return immediately if (row, col) is out of bounds. */

    /* TODO 3 — Accumulate the dot product over the inner dimension. */

    /* TODO 4 — Add the bias term. */

    /* TODO 5 — Apply ReLU activation and write the result to C. */
}


/* ═════════════════════════════════════════════════════════════════════════════
 * KERNEL 2 — Plain Matrix-Multiply (no activation)
 *
 * Computes:  C[i][j] = sum_k A[i][k] * B[k][j]  +  bias[j]
 *
 * Arguments: identical layout to matmul_relu_kernel but WITHOUT ReLU.
 * ═════════════════════════════════════════════════════════════════════════════*/
__global__ void matmul_kernel(const float *A,
                              const float *B,
                              const float *bias,
                                    float *C,
                              int rows_A,
                              int inner,
                              int cols_B)
{
    /* TODO 6 — Compute row and column indices (same as kernel 1). */

    /* TODO 7 — Bounds check (same as kernel 1). */

    /* TODO 8 — Accumulate dot product (same as kernel 1). */

    /* TODO 9 — Add bias and write result (NO ReLU this time). */
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
    /* Layer 1: H = ReLU(X @ W1 + b1) */
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D_H; ++j) {
            float acc = b1[j];
            for (int k = 0; k < D_IN; ++k)
                acc += X[i * D_IN + k] * W1[k * D_H + j];
            H[i * D_H + j] = acc > 0.0f ? acc : 0.0f;
        }
    }
    /* Layer 2: Y = H @ W2 + b2 */
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
 * GPU LAUNCH WRAPPER
 *
 * The batch (N = 64 rows) is divided into NUM_STREAMS = 4 equal chunks of
 * CHUNK = 16 rows.  Stream s owns rows [s*CHUNK, (s+1)*CHUNK).
 *
 * Phase 0 — weight upload (one-time, synchronous):
 *   W1, b1, W2, b2 are shared and read-only; copy them to the device once
 *   before any stream starts, using blocking cudaMemcpy.
 *
 * Phase 1 — per-stream pipeline (all 4 streams in flight simultaneously):
 *   For each stream s:
 *     (a) H2D async copy  →  its CHUNK rows of X
 *     (b) matmul_relu_kernel on the slice  →  CHUNK rows of H
 *     (c) matmul_kernel    on the slice    →  CHUNK rows of Y
 *     (d) D2H async copy  ←  its CHUNK rows of Y back to host
 *
 * Phase 2 — synchronise and destroy all streams.
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
    /* ── Phase 0: upload shared weights once (synchronous) ──────────────────── */
    /*
     * W1, b1, W2, b2 are identical for every chunk.  Copy them once with
     * blocking cudaMemcpy so they are guaranteed to be on the device before
     * any stream's kernel reads them.
     *
     * TODO 10 — Copy W1  : D_IN * D_H  floats
     * TODO 11 — Copy b1  : D_H         floats
     * TODO 12 — Copy W2  : D_H  * D_OUT floats
     * TODO 13 — Copy b2  : D_OUT        floats
     *
     * Use cudaMemcpy (blocking) with direction cudaMemcpyHostToDevice.
     */

    /* ── Phase 1: create 4 streams ───────────────────────────────────────────── */
    /*
     * TODO 14 — Declare an array of NUM_STREAMS stream handles and create each:
     *
     *   cudaStream_t streams[NUM_STREAMS];
     *   for (int s = 0; s < NUM_STREAMS; ++s)
     *       CUDA_CHECK( cudaStreamCreate(&streams[s]) );
     */

    /*
     * Block / grid dimensions.
     * Each kernel launch processes CHUNK rows (not N), so the grid Y-size
     * is computed from CHUNK.
     *
     * TODO 15 — Define block and the two grid shapes:
     *
     *   dim3 block(16, 16);
     *   dim3 grid_h(  (D_H  + block.x - 1) / block.x,
     *                 (CHUNK + block.y - 1) / block.y );   // for hidden layer
     *   dim3 grid_out((D_OUT + block.x - 1) / block.x,
     *                 (CHUNK + block.y - 1) / block.y );   // for output layer
     */

    /* ── Per-stream loop ─────────────────────────────────────────────────────── */
    for (int s = 0; s < NUM_STREAMS; ++s)
    {
        /*
         * TODO 16 — Compute the element offsets for this chunk:
         *
         *   int row_off = s * CHUNK;           // first row of this chunk
         *   int x_off   = row_off * D_IN;      // offset into X / d_X
         *   int h_off   = row_off * D_H;       // offset into H / d_H
         *   int y_off   = row_off * D_OUT;     // offset into Y / d_Y
         *
         * Then derive slice pointers:
         *   float       *d_X_s = d_X + x_off;
         *   float       *d_H_s = d_H + h_off;
         *   float       *d_Y_s = d_Y + y_off;
         *   const float *h_X_s = h_X + x_off;
         *   float       *h_Y_s = h_Y + y_off;
         */

        /* (a) TODO 17 — Async H2D: copy CHUNK rows of X into d_X_s
         *
         *   CUDA_CHECK( cudaMemcpyAsync(d_X_s, h_X_s,
         *                               CHUNK * D_IN * sizeof(float),
         *                               cudaMemcpyHostToDevice, streams[s]) );
         */

        /* (b) TODO 18 — Launch matmul_relu_kernel for this slice.
         *   Input:  d_X_s  (CHUNK × D_IN),  d_W1 (D_IN × D_H),  d_b1
         *   Output: d_H_s  (CHUNK × D_H)
         *   rows_A = CHUNK, inner = D_IN, cols_B = D_H
         *
         *   matmul_relu_kernel<<<grid_h, block, 0, streams[s]>>>(
         *       d_X_s, d_W1, d_b1, d_H_s, CHUNK, D_IN, D_H);
         *   CUDA_CHECK( cudaGetLastError() );
         */

        /* (c) TODO 19 — Launch matmul_kernel for this slice.
         *   Input:  d_H_s  (CHUNK × D_H),   d_W2 (D_H × D_OUT), d_b2
         *   Output: d_Y_s  (CHUNK × D_OUT)
         *   rows_A = CHUNK, inner = D_H, cols_B = D_OUT
         *
         *   matmul_kernel<<<grid_out, block, 0, streams[s]>>>(
         *       d_H_s, d_W2, d_b2, d_Y_s, CHUNK, D_H, D_OUT);
         *   CUDA_CHECK( cudaGetLastError() );
         */

        /* (d) TODO 20 — Async D2H: copy CHUNK rows of Y from d_Y_s back to host
         *
         *   CUDA_CHECK( cudaMemcpyAsync(h_Y_s, d_Y_s,
         *                               CHUNK * D_OUT * sizeof(float),
         *                               cudaMemcpyDeviceToHost, streams[s]) );
         */
    }

    /* ── Phase 2: synchronise and destroy all streams ───────────────────────── */
    /*
     * TODO 21 — For each stream: cudaStreamSynchronize, then cudaStreamDestroy.
     *
     *   for (int s = 0; s < NUM_STREAMS; ++s) {
     *       CUDA_CHECK( cudaStreamSynchronize(streams[s]) );
     *       CUDA_CHECK( cudaStreamDestroy(streams[s]) );
     *   }
     */
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
