#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* -----------------------------------------------------------------------
 * Error-checking helper macro.
 * Wraps any CUDA API call; prints file/line and aborts on failure.
 * ----------------------------------------------------------------------- */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* -----------------------------------------------------------------------
 * CPU reference implementation — do NOT modify.
 * ----------------------------------------------------------------------- */
void cpuVecAdd(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

/* -----------------------------------------------------------------------
 * GPU kernel — fill in the three TODOs below.
 * Each thread is responsible for exactly ONE element of the output array.
 * ----------------------------------------------------------------------- */
__global__ void vecAdd(const float* A, const float* B, float* C, int N) {

    // TODO 1: Compute the global 1-D index for this thread.
    //         Use threadIdx.x, blockDim.x, and blockIdx.x.
    //         (1 line)
    int globalIndex = /* ??? */;

    // TODO 2: Add a bounds check so that threads beyond the array length N
    //         do not read or write out-of-bounds memory.
    //         (1 line — an if-statement that wraps TODO 3)
    /* ??? */ {

        // TODO 3: Perform the vector addition for this element.
        //         Store the result in C[globalIndex].
        //         (1 line)
        /* ??? */;

    } // end bounds check
}

/* -----------------------------------------------------------------------
 * Main — allocates memory, launches the kernel, and validates results.
 * You do NOT need to modify anything below this line.
 * ----------------------------------------------------------------------- */
int main(void) {
    /* ----- Problem size and launch configuration ----- */
    const int N              = 1024;
    const int threadsPerBlock = 256;
    const int numBlocks       = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching kernel: %d blocks x %d threads = %d threads total\n",
           numBlocks, threadsPerBlock, numBlocks * threadsPerBlock);

    /* ----- Host (CPU) memory ----- */
    float* h_A   = (float*)malloc(N * sizeof(float));
    float* h_B   = (float*)malloc(N * sizeof(float));
    float* h_C   = (float*)malloc(N * sizeof(float));   // GPU result
    float* h_ref = (float*)malloc(N * sizeof(float));   // CPU reference

    /* Initialize input arrays with simple values */
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;            /* 0, 1, 2, … N-1  */
        h_B[i] = (float)(N - i);      /* N, N-1, … 1     */
    }

    /* ----- Device (GPU) memory ----- */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * sizeof(float)));

    /* Copy inputs from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    /* ----- Launch the kernel ----- */
    vecAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());   /* catch launch errors        */
    CUDA_CHECK(cudaDeviceSynchronize()); /* wait for kernel to finish */

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* ----- CPU reference ----- */
    cpuVecAdd(h_A, h_B, h_ref, N);

    /* ----- Validate ----- */
    const float TOLERANCE = 1e-5f;
    int passed = 1;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_C[i] - h_ref[i]) > TOLERANCE) {
            fprintf(stderr, "FAILED at index %d: GPU=%.6f  CPU=%.6f\n",
                    i, h_C[i], h_ref[i]);
            passed = 0;
            break;
        }
    }
    if (passed) {
        printf("PASSED: GPU result matches CPU result.\n");
    }

    /* ----- Clean up ----- */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
