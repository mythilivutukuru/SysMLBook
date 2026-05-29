#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <chrono>
#include <immintrin.h>
#include <xmmintrin.h>

static void initialize_matrix(double *M, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        M[i] = (double)rand() / RAND_MAX;
}

static void zero_matrix(double *M, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        M[i] = 0.0;
}

static void reference_mat_mul(const double *A, const double *B,
                               double *C, int size) {
    for (int i = 0; i < size; i++)
        for (int k = 0; k < size; k++) {
            double a_ik = A[i * size + k];
            for (int j = 0; j < size; j++)
                C[i * size + j] += a_ik * B[k * size + j];
        }
}

static int check_correctness(const double *ref, const double *test,
                              int size, double tol) {
    double max_err = 0.0;
    for (int i = 0; i < size * size; i++) {
        double err = fabs(ref[i] - test[i]);
        if (err > max_err) max_err = err;
    }
    printf("Correctness check: %s  (max element-wise error = %e)\n",
           max_err <= tol ? "PASSED" : "FAILED", max_err);
    return max_err <= tol;
}

/**
 * @brief  Matrix multiplication using SIMD.
 * @param  A     Pointer to the first  matrix  (size × size, row-major)
 * @param  B     Pointer to the second matrix  (size × size, row-major)
 * @param  C     Pointer to the result matrix  (size × size, row-major)
 * @param  size  Dimension of the square matrices
 */
void simd_mat_mul(double *A, double *B, double *C, int size) {

    /* ── 128-bit SSE2 path (2 doubles per register) ─────────────────────
     *
     *  for (int i = 0; i < size; i++) {
     *      for (int k = 0; k < size; k++) {
     *          __m128d a_bcast = _mm_set1_pd(A[i * size + k]);
     *          int j;
     *          for (j = 0; j + 2 <= size; j += 2) {
     *              __m128d b_vec = _mm_loadu_pd(&B[k * size + j]);
     *              __m128d c_vec = _mm_loadu_pd(&C[i * size + j]);
     *              c_vec = _mm_add_pd(c_vec, _mm_mul_pd(a_bcast, b_vec));
     *              _mm_storeu_pd(&C[i * size + j], c_vec);
     *          }
     *          for (; j < size; j++)
     *              C[i * size + j] += A[i * size + k] * B[k * size + j];
     *      }
     *  }
     * ──────────────────────────────────────────────────────────────────*/

    /* ── 256-bit AVX2 path (4 doubles per register) ─────────────────────
     *
     *  for (int i = 0; i < size; i++) {
     *      for (int k = 0; k < size; k++) {
     *          __m256d a_bcast = _mm256_broadcast_sd(&A[i * size + k]);
     *          int j;
     *          for (j = 0; j + 4 <= size; j += 4) {
     *              __m256d b_vec = _mm256_loadu_pd(&B[k * size + j]);
     *              __m256d c_vec = _mm256_loadu_pd(&C[i * size + j]);
     *              c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_bcast, b_vec));
     *              _mm256_storeu_pd(&C[i * size + j], c_vec);
     *          }
     *          for (; j < size; j++)
     *              C[i * size + j] += A[i * size + k] * B[k * size + j];
     *      }
     *  }
     * ──────────────────────────────────────────────────────────────────*/


    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {

            // Broadcast the single scalar A[i][k] into all 8 SIMD lanes.
            // Every lane of the subsequent multiply will use this same value.
            __m512d a_bcast = _mm512_set1_pd(A[i * size + k]);

            int j;
            // ── Vectorised inner loop (8 doubles per iteration) ──────────
            for (j = 0; j + 8 <= size; j += 8) {
                // Load 8 elements of B's k-th row starting at column j.
                __m512d b_vec = _mm512_loadu_pd(&B[k * size + j]);

                // Load the current partial sum from C's i-th row.
                __m512d c_vec = _mm512_loadu_pd(&C[i * size + j]);

                // Fused multiply-add: c_vec = a_bcast * b_vec + c_vec
                c_vec = _mm512_fmadd_pd(a_bcast, b_vec, c_vec);

                // Write the updated 8 partial sums back to C.
                _mm512_storeu_pd(&C[i * size + j], c_vec);
            }

            // Handle the remaining columns when size is not a multiple of 8.
            // 'j' continues from where the SIMD loop stopped.
            for (; j < size; j++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}


int main(int argc, char **argv) {

    if (argc <= 1) {
        printf("Usage: %s <matrix_dimension>\n", argv[0]);
        return 0;
    }

    int size = atoi(argv[1]);
    printf("Matrix size: %d x %d\n\n", size, size);

    double *A     = (double *)malloc(size * size * sizeof(double));
    double *B     = (double *)malloc(size * size * sizeof(double));
    double *C_ref = (double *)calloc(size * size, sizeof(double));
    double *C_stu = (double *)calloc(size * size, sizeof(double));

    if (!A || !B || !C_ref || !C_stu) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    srand((unsigned)time(NULL));
    initialize_matrix(A, size, size);
    initialize_matrix(B, size, size);

    zero_matrix(C_ref, size, size);
    auto t0 = std::chrono::high_resolution_clock::now();
    reference_mat_mul(A, B, C_ref, size);
    auto t1 = std::chrono::high_resolution_clock::now();
    long ref_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    printf("Reference multiplication ... done in %5ld ms\n", ref_ms);

    zero_matrix(C_stu, size, size);
    auto t2 = std::chrono::high_resolution_clock::now();
    simd_mat_mul(A, B, C_stu, size);
    auto t3 = std::chrono::high_resolution_clock::now();
    long stu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    printf("SIMD multiplication               ... done in %5ld ms\n", stu_ms);

    if (stu_ms > 0)
        printf("Speedup: %.2fx\n", (double)ref_ms / stu_ms);
    else
        printf("Speedup: N/A (SIMD time < 1 ms)\n");

    printf("\n");

    check_correctness(C_ref, C_stu, size, 1e-6);

    free(A);
    free(B);
    free(C_ref);
    free(C_stu);

    return 0;
}