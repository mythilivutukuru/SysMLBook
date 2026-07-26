#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>

const int MAX_ELEMS_PER_THREAD = 8;

/**
 * Each thread computes elems_per_thread CONTIGUOUS elements down a column.
 * 
 * Thread (ty, tx) in a block computes column: blockCol*TILE + tx
 * and rows: baseRow + ty*elems_per_thread to baseRow + ty*elems_per_thread + elems_per_thread - 1
 * where baseRow = blockRow * (TILE * elems_per_thread)
 */
__global__ void matmul_tiled_col_contiguous_kernel(const float* __restrict__ A,
                                                    const float* __restrict__ B,
                                                    float* __restrict__ C,
                                                    int N, int TILE, int elems_per_thread) {
    extern __shared__ float s[];
    float* sA = s;                                          // size: (TILE * elems_per_thread) * TILE
    float* sB = s + (size_t)TILE * (TILE * elems_per_thread); // size: TILE * TILE

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int localRow = threadIdx.y;  // 0 .. TILE-1
    int localCol = threadIdx.x;  // 0 .. TILE-1

    // Each block covers (TILE * elems_per_thread) rows and TILE columns
    int baseGlobalRow = blockRow * (TILE * elems_per_thread);
    int globalCol = blockCol * TILE + localCol;

    // This thread's starting row for CONTIGUOUS elements
    int threadRowStart = baseGlobalRow + localRow * elems_per_thread;

    // Accumulators for contiguous elements
    float acc[MAX_ELEMS_PER_THREAD];
    for (int e = 0; e < elems_per_thread; ++e) {
        acc[e] = 0.0f;
    }

    int numTiles = (N + TILE - 1) / TILE;
    int sA_row_stride = TILE;
    int sB_row_stride = TILE;

    for (int t = 0; t < numTiles; ++t) {
        // --- Load A tile ---
        // Each thread loads elems_per_thread CONTIGUOUS elements
        int aColBase = t * TILE + localCol;
        for (int e = 0; e < elems_per_thread; ++e) {
            int gRow = threadRowStart + e; 
            float aval = 0.0f;
            if (gRow < N && aColBase < N) {
                aval = A[gRow * N + aColBase];
            }
            // Store into sA at row (localRow * elems_per_thread + e), column localCol
            sA[(localRow * elems_per_thread + e) * sA_row_stride + localCol] = aval;
        }

        // --- Load B tile (TILE x TILE) ---
        int bRow = t * TILE + localRow;
        int bColBase = blockCol * TILE;
        float bval = 0.0f;
        if (bRow < N && (bColBase + localCol) < N) {
            bval = B[bRow * N + (bColBase + localCol)];
        }
        sB[localRow * sB_row_stride + localCol] = bval;

        __syncthreads();

        // --- Compute partial sums ---
        for (int k = 0; k < TILE; ++k) {
            float b_k = sB[k * sB_row_stride + localCol];
            for (int e = 0; e < elems_per_thread; ++e) {
                int sA_row = localRow * elems_per_thread + e;
                float a_k = sA[sA_row * sA_row_stride + k];
                acc[e] += a_k * b_k;
            }
        }

        __syncthreads();
    }

    // --- Write results back (contiguous rows) ---
    for (int e = 0; e < elems_per_thread; ++e) {
        int grow = threadRowStart + e;
        if (grow < N && globalCol < N) {
            C[grow * N + globalCol] = acc[e];
        }
    }
}

void matmul_gpu_col_contiguous(int N, int TILE, int elems_per_thread,
                               float* A_h, float* B_h, float* C_h, float &elapsed_ms) {
    float *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;
    size_t bytes = (size_t)N * N * sizeof(float);
    
    cudaMalloc(&A_d, bytes);
    cudaMalloc(&B_d, bytes);
    cudaMalloc(&C_d, bytes);

    cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);

    // Grid dimensions
    int blocks_x = (N + TILE - 1) / TILE;
    int blocks_y = (N + TILE * elems_per_thread - 1) / (TILE * elems_per_thread);
    dim3 grid(blocks_x, blocks_y);

    // Shared memory: sA (TILE*elems_per_thread x TILE) + sB (TILE x TILE)
    size_t sharedFloats = (size_t)TILE * TILE * (elems_per_thread + 1);
    size_t sharedBytes = sharedFloats * sizeof(float);

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);
    
    matmul_tiled_col_contiguous_kernel<<<grid, block, sharedBytes>>>(
        A_d, B_d, C_d, N, TILE, elems_per_thread);
    
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

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./matmul_tiled_col_contiguous <N> <TILE> <elems_per_thread>\n";
        return 1;
    }
    
    int N = atoi(argv[1]);
    int TILE = atoi(argv[2]);
    int elems = atoi(argv[3]);
    
    std::cout << "========================================\n";
    std::cout << "CONTIGUOUS Col-per-thread TILED matmul\n";
    std::cout << "N=" << N << " TILE=" << TILE << " elems_per_thread=" << elems << "\n";
    std::cout << "========================================\n\n";
    
    size_t bytes = (size_t)N * N * sizeof(float);
    float *A = (float*)malloc(bytes);
    float *B = (float*)malloc(bytes);
    float *C_gpu = (float*)malloc(bytes);
    float *C_cpu = (float*)malloc(bytes);
    
    // Initialize matrices
    for (size_t i = 0; i < (size_t)N * N; ++i) {
        A[i] = float(i % 100);
        B[i] = float((i * 7) % 100);
    }
    
    // GPU computation
    float ms;
    matmul_gpu_col_contiguous(N, TILE, elems, A, B, C_gpu, ms);
    std::cout << "GPU Kernel time: " << ms << " ms\n\n";
    
    free(A);
    free(B);
    free(C_gpu);
    free(C_cpu);
    
    return 0;
}
