# SIMD Matrix Multiplication

In this assignment you will implement **matrix multiplication using SIMD (Single Instruction, Multiple Data) intrinsics** on x86-64 processors. You will explore how vectorized instructions can accelerate compute-bound workloads by processing multiple floating-point values in parallel within a single CPU cycle.

Your completed implementation will be verified against a reference CPU implementation for correctness, and its runtime will be compared against the naive baseline.

---

## Background

### Register Widths

| ISA Extension | Register Width | `double` lanes |
|---|---|---|
| SSE2 | 128-bit | 2 |
| AVX / AVX2 | 256-bit | 4 |
| AVX-512 | 512-bit | 8 |

This assignment targets **AVX-512 (512-bit registers)**. The 128-bit and 256-bit paths are provided in the solution file as commented-out reference code.

### Key Intrinsics You Will Use

| Intrinsic | Purpose |
|---|---|
| `_mm512_setzero_pd()` | Zero-initialize an `__m512d` accumulator |
| `_mm512_set1_pd(x)` | Broadcast a scalar `double` to all 8 lanes |
| `_mm512_loadu_pd(ptr)` | Load 8 unaligned `double`s from memory |
| `_mm512_mul_pd(a, b)` | Element-wise multiply |
| `_mm512_add_pd(a, b)` | Element-wise add |
| `_mm512_storeu_pd(ptr, v)` | Store 8 `double`s to unaligned memory |
| `_mm512_fmadd_pd(a, b, c)` | Fused multiply-add: `a*b + c` (optional optimization) |

Header required: `#include <immintrin.h>`

---

## Task

Implement the function:

```c
void simd_mat_mul(double *A, double *B, double *C, int size);
```

which computes `C = A × B` for square matrices of dimension `size × size` stored in **row-major order**.

You must use **AVX-512 intrinsics** (`__m512d`, 512-bit registers) to vectorize the innermost computation. Your implementation must:

1. Process **8 `double` elements at a time** using 512-bit SIMD registers along the `j` (column) dimension of the output matrix.
2. Handle the **remainder** (when `size` is not a multiple of 8) correctly using a scalar fallback loop.
3. Produce results that are **numerically equivalent** to the reference CPU implementation (element-wise absolute difference ≤ `1e-6`).

You are **not** required to implement tiling or loop reordering for this task, though you may if you wish for extra performance.

### Algorithm Sketch

```
for i in [0, size):                        // row of A and C
    for k in [0, size):                    // shared dimension
        a_ik = broadcast A[i][k] to all 8 lanes
        for j in [0, size) step 8:         // columns of B and C, 8 at a time
            C[i][j:j+8] += a_ik * B[k][j:j+8]   // SIMD fused multiply-add
        // scalar tail for remaining columns
```

`A[i][k]` is a **scalar** that is broadcast into all 8 lanes, while `B[k][j:j+8]` is an 8-wide vector loaded from a contiguous row of B.

---

## Testing

A C++ compiler with AVX-512 support is required.

```bash
# Compile with AVX-512 enabled
g++ -mavx512f -std=c++17 -o simd_mat_mul simd_mat_mul.c

# Run with a matrix dimension of your choice (e.g., 1024)
./simd_mat_mul 1024
```

> **Note:** If your machine does not support AVX-512, compile with `-mavx2` and switch to the 256-bit path (uncomment the `__m256d` block and comment out the `__m512d` block).

---