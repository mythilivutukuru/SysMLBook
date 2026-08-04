# Row-wise Tiling in Matrix Multiplication

Efficient matrix multiplication on GPUs relies on reducing global memory
traffic and increasing data reuse. Previously, you implemented a naive
tiled kernel in which each thread computed a single element of the output
matrix using shared-memory tiles of $A$ and $B$. This assignment extends
that design so that each thread computes **several elements along a
single output row**, amortizing the cost of loading $A$ over multiple
output columns and increasing arithmetic intensity per thread.

This design improves performance by:

1. Loading a tile of $A$ and several column-tiles of $B$ into shared
   memory.
2. Reusing each loaded value of $A$ across `elems_per_thread` output
   columns.
3. Accumulating `elems_per_thread` partial results per thread in
   registers before writing back to global memory.

You are given the file `student.cu` in this directory.
It contains:

- `main()` (testing infrastructure, including a CPU reference check),
- `matmul_gpu_row()` (kernel launch wrapper), and
- The signature of the CUDA kernel that you must implement.

---

## Background

### Problem Setup

Given two square matrices $A, B \in \mathbb{R}^{N \times N}$, compute:

$$
C = A \times B,
$$

where

$$
C[i,j] = \sum_{k=0}^{N-1} A[i,k] \cdot B[k,j],
\qquad 0 \le i,j < N.
$$

All matrices are stored in **row-major flattened format**.

### Kernel Design Overview

Let:

- `TILE` be the shared-memory tile dimension (and the thread-block
  dimension along each axis).
- `elems_per_thread` be the number of output columns each thread is
  responsible for.

Each thread owns exactly **one row** of the output tile, and computes
`elems_per_thread` output elements spaced `TILE` columns apart along
that row, entirely in registers. Therefore:

- Each thread block computes a

  ```text
  TILE × (TILE * elems_per_thread)
  ```

  tile of `C`.

- The block dimension is

  ```text
  TILE × TILE
  ```

  threads (`localCol` indexes columns within a sub-tile, `localRow`
  indexes the row).

- The grid dimension along `x` is scaled down by
  `elems_per_thread` since each block now covers that many times more
  columns than a naive tiled kernel would.

### Shared Memory Layout

A single dynamically allocated shared-memory buffer is used:

```cpp
extern __shared__ float s[];
float* sA = s;                            // TILE * TILE floats
float* sB = s + TILE * TILE;              // TILE * (TILE * elems_per_thread) floats
```

`sA` is a row-major TILEXTILE array holding one
K-tile of $A$. `sB` is a row-major
TILEX(TILE.elems_per_thread)$
array holding `elems_per_thread` side-by-side column-tiles of $B$ for the
same K-tile, laid out contiguously so that row `k` of `sB` starts at
offset `k * (TILE * elems_per_thread)`.

### Thread Responsibilities

Let `(localCol, localRow)` denote `(threadIdx.x, threadIdx.y)`, the
thread coordinates inside a block.

Each thread computes an output row segment beginning at:

```text
globalRow     = blockIdx.y * TILE + localRow
baseGlobalCol = blockIdx.x * (TILE * elems_per_thread)
```

The $e$-th element ($0 \le e < \mathtt{elems\_per\_thread}$) that this
thread owns sits at global column
$\mathtt{baseGlobalCol} + \mathtt{localCol} + e \times \mathtt{TILE}$.

Each thread must:

- Maintain a register accumulator array:

  ```cpp
  float acc[elems_per_thread];
  ```

  initialized to zero (bounded above at compile time by
  `MAX_ELEMS_PER_THREAD`).
- Load **one** element of the current $A$-tile into `sA`.
- Load **`elems_per_thread`** elements of the current $B$-tile(s) into
  `sB` — one per sub-tile.
- Accumulate into `acc`.
- Write the `elems_per_thread` results back to global memory.

### Tiling Along the K Dimension

The multiplication proceeds over tiles along the shared $K$ dimension.
The number of tiles is:

$$
\texttt{numTiles} = \left\lceil \frac{N}{\mathtt{TILE}} \right\rceil.
$$

For each tile index $t$, the block cooperatively loads one tile of $A$
and `elems_per_thread` column-tiles of $B$ into shared memory. Any
access that falls outside the $[0, N)$ bounds (because $N$ is not an
exact multiple of `TILE` or `TILE * elems_per_thread`) must be zero-padded
rather than read out of bounds.

#### Loading the Tile of A

Each thread loads exactly one element of the current A-tile:
row `localRow`, column `t * TILE + localCol`, into
`sA[localRow * TILE + localCol]`.

#### Loading the Tile(s) of B

Each thread loops over `e` in `[0, elems_per_thread)` and loads the
element of $B$ at row `t * TILE + localRow` and column
`baseGlobalCol + localCol + e * TILE` into
`sB[localRow * (TILE * elems_per_thread) + localCol + e * TILE]`.

After loading both `sA` and all sub-tiles of `sB`, call:

```cpp
__syncthreads();
```

#### Compute Phase

For each position `k` in `[0, TILE)` along the shared dimension, each
thread:

1. Reads a single value `aval = sA[localRow * TILE + k]` — this value is
   **reused across all `elems_per_thread` output columns** owned by the
   thread.
2. For each `e` in `[0, elems_per_thread)`, reads `bval` from
   `sB[k * (TILE * elems_per_thread) + localCol + e * TILE]` and
   accumulates `acc[e] += aval * bval`.

Then call `__syncthreads()` before proceeding to the next tile.

### Final Write-Back

After all tiles have been processed, each thread writes its
`elems_per_thread` accumulated values into row `globalRow` of `C`, at
columns `baseGlobalCol + localCol + e * TILE` for each `e`, guarding
against out-of-bounds writes when $N$ is not a multiple of the tile
dimensions.

## Task

Implement the kernel:

```cpp
__global__ void matmul_tiled_row_kernel(
    const float* A,
    const float* B,
          float* C,
    int N,
    int TILE,
    int elems_per_thread
)
```

- `A, B` = input matrices in row-major flattened format ($N \times N$).
- `C` = output matrix in row-major flattened format ($N \times N$).
- `N` = side length of all three matrices.
- `TILE` = shared-memory tile size along one axis; the block dimension is
  $\mathtt{TILE} \times \mathtt{TILE}$.
- `elems_per_thread` = number of output columns each thread computes,
  bounded by the compile-time constant `MAX_ELEMS_PER_THREAD`.

**Requirements:**

- Use `extern __shared__` memory and partition it into `sA` and `sB`.
- Declare and zero-initialize `float acc[MAX_ELEMS_PER_THREAD]` (only the
  first `elems_per_thread` entries are used).
- Compute `globalRow` and `baseGlobalCol`.
- Loop over all $K$-dimension tiles.
- Perform cooperative loading into shared memory, zero-padding
  out-of-bounds reads.
- Synchronize correctly (once after loading, once after computing).
- Accumulate values into registers, reusing each `sA` value across all
  `elems_per_thread` columns.
- Write final results to global memory, guarding against out-of-bounds
  writes.

## Testing

The provided test:

- Computes a CPU reference result.
- Launches your kernel via `matmul_gpu_row`.
- Compares results element-wise.

A test case receives PASS only if **all** elements satisfy the following
criteria:

$$
|C_{\text{CPU}} - C_{\text{GPU}}| < 10^{-3}.
$$
