# Register Tiling in Matrix Multiplication

Efficient matrix multiplication on GPUs relies on reducing global memory
traffic and increasing data reuse. Previously, you implemented a tiled
matmul kernel where each thread computed a single element of the output
matrix, with both `A` and `B` tiles cached in shared memory. This
assignment introduces **register blocking**: instead of computing one
output element, each thread will compute several output elements along
one dimension (columns) and hold them in private registers across the
entire K-loop, before writing them back to global memory.

Unlike a fully 2D register-blocked design (where a thread owns an
`R x R` sub-block of the output and both operands are tiled through
shared memory), this assignment uses a simpler, **1D** register-blocking
scheme:

- Only tiles of `B` are staged in shared memory.
- Tiles of `A` are read directly from global memory, one value at a
  time, and that single value is reused across all of the output
  columns a thread is responsible for.
- Each thread keeps its output accumulators in a small private
  register array (`acc[]`) rather than shared memory.

This still improves arithmetic intensity relative to the naive
one-output-per-thread kernel, because every element read from `B`
(via shared memory) is reused across multiple `k` iterations, and every
value read from `A` is reused across `elems_per_thread` multiply–adds
instead of just one.

You are given the file `student.cu` in this directory. It contains:

- `main()` (testing infrastructure),
- `matmul_gpu_registers()` (kernel launch wrapper), and
- The signature (and partial body) of the CUDA kernel that you must
  complete.

---

## Background

### Problem Setup

Given two square matrices `A, B` of size `N x N`, compute:

```
C = A x B,
C[i,j] = sum_{k=0}^{N-1} A[i,k] * B[k,j],   0 <= i, j < N
```

All matrices are stored in **row-major flattened format**.

### Kernel Design Overview

Let:

- `TILE` be the shared-memory tile dimension (and also the thread
  block's row/column dimension — the block is launched as
  `dim3(TILE, TILE)`).
- `elems_per_thread` be the compile-time-bounded (but runtime-valued)
  number of output columns each thread is responsible for.

Each thread block computes a tile of `C` that is:

- `TILE` rows tall, and
- `TILE * elems_per_thread` columns wide.

Each thread within the block:

- Owns exactly **one row** of the output tile (`localRow`), and
- Owns `elems_per_thread` **columns** of that row, strided `TILE`
  apart, starting at `localCol`.

So thread `(localRow, localCol)` is responsible for output columns:

```
localCol, localCol + TILE, localCol + 2*TILE, ..., localCol + (elems_per_thread - 1)*TILE
```

within its block's tile — all in the same output row.

### Shared Memory Layout

A single dynamically allocated shared-memory buffer is used, but
**only `B` is staged through shared memory**:

```cpp
extern __shared__ float s[]; // size = TILE * (TILE * elems_per_thread) floats
float* sB = s;
```

`sB` is treated as a row-major array with `TILE` rows and
`TILE * elems_per_thread` columns (`sB_row_stride`). There is no `sA`
buffer — values of `A` are read directly from global memory inside the
K-loop and reused immediately across the `elems_per_thread` accumulations.

### Thread Responsibilities

Let `(localRow, localCol)` denote a thread's coordinates inside a
block (`threadIdx.y`, `threadIdx.x`).

Each thread computes:

```
globalRow      = blockIdx.y * TILE + localRow
baseGlobalCol  = blockIdx.x * (TILE * elems_per_thread)
```

Each thread must:

- Maintain a register accumulator array `float acc[elems_per_thread]`,
  zero-initialized.
- Cooperatively load a `TILE x (TILE * elems_per_thread)` tile of `B`
  into `sB`.
- Loop over `k` from `0` to `TILE - 1`:
  - Load **one** value of `A` at `(globalRow, t*TILE + k)` from global
    memory into a register.
  - Reuse that single register value to update all `elems_per_thread`
    accumulators, multiplying it against the corresponding `elems_per_thread`
    values of `sB` on row `k`.
- Synchronize correctly between the load phase and the compute phase,
  and again before moving to the next K-tile.
- Write the final `elems_per_thread` accumulated values back to `C`.

### Tiling Along the K Dimension

The multiplication proceeds over tiles along the shared `K` dimension.
The number of tiles is:

```
numTiles = ceil(N / TILE)
```

For each tile index `t`, the block cooperatively loads one tile of `B`
into shared memory (there is nothing to cooperatively load for `A`,
since `A` is read directly by each thread as needed).

#### Loading Tile of B

Each thread loads `elems_per_thread` values of `B` into `sB`. For
`e` in `[0, elems_per_thread)`:

```
bRow = t * TILE + localRow
gCol = baseGlobalCol + localCol + e * TILE
sB[localRow * sB_row_stride + (localCol + e * TILE)] = (in-bounds) ? B[bRow * N + gCol] : 0
```

After loading, call `__syncthreads()` before the compute phase.

#### Compute Phase

For each `k` in `[0, TILE)`:

1. Load one value of `A` at global position `(globalRow, t*TILE + k)`
   into a register (0 if out of bounds).
2. For each `e` in `[0, elems_per_thread)`, read `sB[k * sB_row_stride + (localCol + e*TILE)]`
   and accumulate `acc[e] += aVal * bVal`.

Then call `__syncthreads()` before proceeding to the next tile, so no
thread starts overwriting `sB` while others are still reading it.

### Final Write-Back

After all tiles have been processed, each thread writes its
`elems_per_thread` accumulated values back to `C`, at global row
`globalRow` and columns `baseGlobalCol + localCol + e * TILE`,
guarding against out-of-bounds `N`.

## Task

Implement the kernel:

```cpp
__global__ void matmul_tiled_registers_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    int N, int TILE, int elems_per_thread
)
```

- `A, B` = input matrices in row-major flattened format (`N x N`).
- `C` = output matrix in row-major flattened format (`N x N`).
- `N` = side length of all three matrices.
- `TILE` = tile size; blocks are launched as `dim3(TILE, TILE)` threads.
- `elems_per_thread` = number of output columns each thread accumulates,
  bounded at compile time by `MAX_ELEMS_PER_THREAD`.

**Requirements:**

- Use `extern __shared__` memory for `sB` only — do **not** allocate a
  shared-memory tile for `A`.
- Declare and zero-initialize `float acc[MAX_ELEMS_PER_THREAD]` (using
  only the first `elems_per_thread` entries).
- Compute `globalRow` and `baseGlobalCol`.
- Loop over all K-dimension tiles.
- Cooperatively load the `B` tile into shared memory, guarding
  out-of-bounds accesses with zeros.
- Synchronize correctly (once after loading, once after the inner
  compute loop).
- Inside the compute loop, load each `A` value once from global memory
  and reuse the register across all `elems_per_thread` accumulations.
- Write final results back to global memory, guarding out-of-bounds
  accesses.

## Testing

The provided test:

- Fills `A` and `B` with deterministic values.
- Launches your kernel via `matmul_gpu_registers`.
- Reports the kernel's elapsed time via CUDA events.

You can validate correctness by comparing against a CPU reference
implementation of matrix multiplication (not included here) — a test
case should be considered a PASS only if **all** elements satisfy:

```
|C_CPU - C_GPU| < 1e-3
```

Try varying `TILE` and `elems_per_thread` (subject to
`TILE * TILE <= 1024` and `1 <= elems_per_thread <= MAX_ELEMS_PER_THREAD`)
to see how register blocking along a single dimension affects
performance compared to your earlier one-output-per-thread kernel.
