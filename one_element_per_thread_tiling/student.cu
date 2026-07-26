#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>

// Kernel: each thread computes one element using shared-memory tiles
//
// TODO: Implement this kernel following the tiling strategy described in
// the README:
//   1. Compute globalRow/globalCol (this thread's output element) and
//      localRow/localCol (this thread's position inside the block).
//   2. Declare and zero-initialize a per-thread accumulator.
//   3. Loop over the K-dimension tiles:
//        a. Cooperatively load one element of A and one element of B
//           into shared memory (sA, sB), zero-padding out-of-bounds
//           indices.
//        b. __syncthreads()
//        c. Accumulate the dot-product contribution from this tile into
//           the accumulator.
//        d. __syncthreads()
//   4. Write the accumulator to C, guarded by a bounds check.
__global__ void matmul_tiled_element_kernel(const float* A, const float* B, float* C, int N, int TILE) {
    extern __shared__ float s[]; // first TILE*TILE for A, next for B
    float* sA = s;
    float* sB = s + TILE * TILE;

    // TODO: compute globalRow, globalCol, localRow, localCol

    // TODO: declare and zero-initialize the accumulator

    // TODO: compute numTiles = ceil(N / TILE)

    // TODO: loop over tiles t = 0 .. numTiles - 1
    // {
    //     TODO: compute the global A/B indices this thread should load
    //           for tile t, and load into sA[...] / sB[...]
    //           (zero-pad if out of bounds)
    //
    //     __syncthreads();
    //
    //     TODO: accumulate over k = 0 .. TILE - 1:
    //           acc += sA[localRow * TILE + k] * sB[k * TILE + localCol];
    //
    //     __syncthreads();
    // }

    // TODO: write acc to C[globalRow * N + globalCol] with a bounds check
}

void matmul_gpu(int N, int TILE, float* A_h, float* B_h, float* C_h, float &elapsed_ms) {
    float *A_d, *B_d, *C_d;
    size_t bytes = (size_t)N * N * sizeof(float);
    cudaMalloc(&A_d, bytes);
    cudaMalloc(&B_d, bytes);
    cudaMalloc(&C_d, bytes);

    cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    size_t sharedBytes = 2 * TILE * TILE * sizeof(float);

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);

    matmul_tiled_element_kernel<<<grid, block, sharedBytes>>>(A_d, B_d, C_d, N, TILE);

    cudaEventRecord(e);
    cudaEventSynchronize(e);
    cudaEventElapsedTime(&elapsed_ms, s, e);

    cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    cudaEventDestroy(s); cudaEventDestroy(e);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./matmul_tiled_element <N> <TILE>\n";
        return 1;
    }
    int N = std::atoi(argv[1]);
    int TILE = std::atoi(argv[2]);

    std::cout << "Element-per-thread TILED matmul N=" << N << " TILE=" << TILE << "\n";

    size_t bytes = (size_t)N * N * sizeof(float);
    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    float* C = (float*)malloc(bytes);

    for (size_t i = 0; i < (size_t)N * N; ++i) {
        A[i] = float(i % 100);
        B[i] = float((i * 7) % 100);
    }

    float ms;
    matmul_gpu(N, TILE, A, B, C, ms);

    std::cout << "Kernel time (ms): " << ms << "\n";

    free(A); free(B); free(C);
    return 0;
}
