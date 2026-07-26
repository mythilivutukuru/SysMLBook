#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>

const int MAX_ELEMS_PER_THREAD = 8;

// -----------------------------------------------------------------------
// TODO: Implement this kernel.
//
// Each thread block computes a TILE x (TILE * elems_per_thread) tile of C.
// Each thread owns one row of that tile (localRow) and is responsible for
// `elems_per_thread` output columns, spaced TILE columns apart:
//
//     col_e = baseGlobalCol + localCol + e * TILE ,  e = 0 .. elems_per_thread-1
//
// Shared memory layout (single dynamic buffer `s`):
//     sA = s                     -> TILE * TILE floats (one K-tile of A)
//     sB = s + TILE*TILE         -> TILE * (TILE*elems_per_thread) floats
//                                    (elems_per_thread column-tiles of B,
//                                     side by side, for the same K-tile)
//
// See README.md for the full algorithm description.
// -----------------------------------------------------------------------
__global__ void matmul_tiled_row_kernel(const float* A, const float* B, float* C,
                                         int N, int TILE, int elems_per_thread) {
    extern __shared__ float s[]; // size must be (TILE*TILE*(1 + elems_per_thread)) floats
    float* sA = s;               // TILE * TILE
    float* sB = s + TILE * TILE; // TILE * (TILE * elems_per_thread)

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int localRow = threadIdx.y; // in [0, TILE)
    int localCol = threadIdx.x; // in [0, TILE)

    // TODO: Compute the global row this thread is responsible for, and the
    // base global column of this thread block's tile (each block covers
    // TILE * elems_per_thread columns).
    int globalRow = 0;     // TODO
    int baseGlobalCol = 0; // TODO

    // TODO: Declare and zero-initialize your per-thread register
    // accumulators. Only the first `elems_per_thread` entries are used.
    float acc[MAX_ELEMS_PER_THREAD];
    // TODO: zero-initialize acc[0..elems_per_thread)

    // TODO: Compute the number of K-dimension tiles.
    int numTiles = 0; // TODO

    // Stride (in floats) between consecutive rows of sB.
    int sB_row_stride = TILE * elems_per_thread;

    for (int t = 0; t < numTiles; ++t) {
        // -----------------------------------------------------------
        // TODO: Cooperatively load one element of the current A-tile
        // into sA[localRow * TILE + localCol]. Zero-pad if the
        // corresponding global (row, col) in A is out of bounds.
        // -----------------------------------------------------------

        // -----------------------------------------------------------
        // TODO: Cooperatively load `elems_per_thread` elements of the
        // current B-tile(s) into sB. For each e in [0, elems_per_thread),
        // load B at row (t*TILE + localRow) and column
        // (baseGlobalCol + localCol + e*TILE) into
        // sB[localRow * sB_row_stride + (localCol + e*TILE)].
        // Zero-pad if out of bounds.
        // -----------------------------------------------------------

        __syncthreads();

        // -----------------------------------------------------------
        // TODO: Compute phase.
        // For each k in [0, TILE):
        //   - read aval = sA[localRow * TILE + k] ONCE
        //   - for each e in [0, elems_per_thread):
        //       read bval from sB[k * sB_row_stride + (localCol + e*TILE)]
        //       acc[e] += aval * bval
        // -----------------------------------------------------------

        __syncthreads();
    }

    // -----------------------------------------------------------------
    // TODO: Write final results back to global memory.
    // For each e in [0, elems_per_thread), write acc[e] to
    // C[globalRow * N + (baseGlobalCol + localCol + e*TILE)],
    // guarding against out-of-bounds row/col.
    // -----------------------------------------------------------------
}

void matmul_gpu_row(int N, int TILE, int elems_per_thread, float* A_h, float* B_h, float* C_h, float &elapsed_ms) {
    float *A_d, *B_d, *C_d;
    size_t bytes = (size_t)N * N * sizeof(float);
    cudaMalloc(&A_d, bytes);
    cudaMalloc(&B_d, bytes);
    cudaMalloc(&C_d, bytes);

    cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);

    // block: TILE x TILE threads
    dim3 block(TILE, TILE);

    // grid.x must account for elems_per_thread (each block covers TILE * elems_per_thread columns)
    int blocks_x = (N + TILE * elems_per_thread - 1) / (TILE * elems_per_thread);
    int blocks_y = (N + TILE - 1) / TILE;
    dim3 grid(blocks_x, blocks_y);

    // shared memory: TILE*TILE for sA + TILE * (TILE*elems) for sB
    size_t sharedFloats = (size_t)TILE * TILE * (1 + elems_per_thread);
    size_t sharedBytes = sharedFloats * sizeof(float);

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);
    matmul_tiled_row_kernel<<<grid, block, sharedBytes>>>(A_d, B_d, C_d, N, TILE, elems_per_thread);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    cudaEventElapsedTime(&elapsed_ms, s, e);

    cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaEventDestroy(s);
    cudaEventDestroy(e);
}

// Simple CPU reference implementation used to validate the GPU result.
void matmul_cpu(int N, const float* A, const float* B, float* C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./matmul_tiled_row <N> <TILE> <elems_per_thread>\n";
        return 1;
    }
    int N = atoi(argv[1]);
    int TILE = atoi(argv[2]);
    int elems = atoi(argv[3]);

    if (elems > MAX_ELEMS_PER_THREAD) {
        std::cerr << "elems_per_thread exceeds MAX_ELEMS_PER_THREAD (" << MAX_ELEMS_PER_THREAD << ")\n";
        return 1;
    }

    std::cout << "Row-per-thread TILED matmul N=" << N << " TILE=" << TILE << " elems_per_thread=" << elems << "\n";

    size_t bytes = (size_t)N * N * sizeof(float);
    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    float* C = (float*)malloc(bytes);
    float* C_ref = (float*)malloc(bytes);

    for (size_t i = 0; i < (size_t)N * N; ++i) {
        A[i] = float(i % 100);
        B[i] = float((i * 7) % 100);
    }

    float ms;
    matmul_gpu_row(N, TILE, elems, A, B, C, ms);
    std::cout << "Kernel time (ms): " << ms << "\n";

    matmul_cpu(N, A, B, C_ref);

    bool pass = true;
    double max_err = 0.0;
    for (size_t i = 0; i < (size_t)N * N; ++i) {
        double err = std::fabs((double)C[i] - (double)C_ref[i]);
        if (err > max_err) max_err = err;
        if (err >= 1e-3) {
            pass = false;
        }
    }

    std::cout << "Max abs error: " << max_err << "\n";
    std::cout << (pass ? "PASS" : "FAIL") << "\n";

    free(A);
    free(B);
    free(C);
    free(C_ref);
    return pass ? 0 : 1;
}
