# Column-Contiguous Tiling in Matrix Multiplication

Efficient matrix multiplication on GPUs relies on reducing global memory traffic and increasing data reuse. Previously, you explored tiling strategies where each thread computed a single element of the output matrix using cooperatively loaded shared-memory tiles. This assignment extends that progression to a **register-blocked, column-contiguous** design, where each thread computes several **contiguous elements down a single column** of the output matrix, increasing arithmetic intensity per thread while keeping the memory-access pattern simple and coalesced.

This design improves performance by:

1. Loading tiles of $A$ and $B$ into shared memory.
2. Having each thread reuse the same shared-memory tile of $B$ across multiple accumulations.
3. Accumulating multiple output elements per thread in registers before writing to global memory.

You are given the file `student.cu` in this directory.
It contains:

- `main()` (testing infrastructure),
- `matmul_gpu_col_contiguous()` (kernel launch wrapper), and
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

- `TILE` be the shared-memory tile dimension (also the block's width and the number of threads per row/column of the block).
- `elems_per_thread` be the number of **contiguous output rows** each thread is responsible for.

Each thread block covers:

- `TILE` columns of $C$, and
- `TILE * elems_per_thread` rows of $C$.

The thread block dimensions are:

$$
\mathtt{TILE} \times \mathtt{TILE}
$$

i.e. `blockDim.x = blockDim.y = TILE`, but each thread produces `elems_per_thread` output values instead of one.

### Shared Memory Layout

A single dynamically allocated shared-memory buffer is used:

```cpp
extern __shared__ float s[];
float* sA = s;                                            // (TILE * elems_per_thread) * TILE floats
float* sB = s + (size_t)TILE * (TILE * elems_per_thread);  // TILE * TILE floats
```

- `sA` holds a tile of $A$ with `TILE * elems_per_thread` rows and `TILE` columns.
- `sB` holds a tile of $B$ with `TILE` rows and `TILE` columns.

Both are treated as row-major arrays with row stride `TILE`.

### Thread Responsibilities

Let `(localRow, localCol)` = `(threadIdx.y, threadIdx.x)` denote a thread's coordinates inside its block, and let `(blockRow, blockCol)` = `(blockIdx.y, blockIdx.x)`.

Each thread is responsible for a **column** of output values:

$$
\begin{aligned}
\mathtt{baseGlobalRow} &= \mathtt{blockRow} \times (\mathtt{TILE} \times \mathtt{elems\_per\_thread}), \\
\mathtt{globalCol} &= \mathtt{blockCol} \times \mathtt{TILE} + \mathtt{localCol}, \\
\mathtt{threadRowStart} &= \mathtt{baseGlobalRow} + \mathtt{localRow} \times \mathtt{elems\_per\_thread}.
\end{aligned}
$$

The thread computes the `elems_per_thread` output elements at rows
`threadRowStart, threadRowStart + 1, ..., threadRowStart + elems_per_thread - 1`, all in column `globalCol`. Note that these output rows are **contiguous**, hence the name of the kernel.

Each thread must:

- Maintain a register accumulator array `acc[elems_per_thread]`, initialized to zero. A fixed-size array of length `MAX_ELEMS_PER_THREAD` is provided for this purpose — only the first `elems_per_thread` entries are used.
- Load `elems_per_thread` contiguous elements of $A$ into `sA`.
- Load one element of $B$ into `sB`.
- Accumulate into `acc`.
- Write the results back to global memory.

### Tiling Along the K Dimension

The multiplication proceeds over tiles along the shared $K$ dimension. The number of tiles is:

$$
\texttt{numTiles} = \left\lceil \frac{N}{\mathtt{TILE}} \right\rceil.
$$

For each tile index $t$, the block cooperatively loads one tile of $A$ and one tile of $B$ into shared memory.

#### Loading Tile of $A$

Each thread loads `elems_per_thread` **contiguous** elements of the current $A$-tile:

- The column being loaded is fixed for the tile: `aColBase = t * TILE + localCol`.
- For each `e` in `[0, elems_per_thread)`, the thread loads global element `A[threadRowStart + e][aColBase]` (with bounds checking against $N$, using `0.0f` for out-of-range accesses) and stores it into `sA` at row `localRow * elems_per_thread + e`, column `localCol`.

#### Loading Tile of $B$

Each thread loads exactly **one** element of the current $B$-tile:

- `bRow = t * TILE + localRow`
- `bCol = blockCol * TILE + localCol`
- The value `B[bRow][bCol]` (with bounds checking, `0.0f` if out of range) is stored into `sB` at row `localRow`, column `localCol`.

After loading both tiles, call:

```cpp
__syncthreads();
```

#### Compute Phase

For each `k` in `[0, TILE)`:

- Load `b_k = sB[k][localCol]` once — this value is shared by all `elems_per_thread` accumulations for this thread.
- For each `e` in `[0, elems_per_thread)`, load `a_k = sA[localRow * elems_per_thread + e][k]` and accumulate `acc[e] += a_k * b_k`.

Then call `__syncthreads()` before proceeding to the next tile.

### Final Write-Back

After all tiles have been processed, each thread writes its `elems_per_thread` accumulated values back to global memory at rows `threadRowStart .. threadRowStart + elems_per_thread - 1`, column `globalCol`, with bounds checking against $N$.

## Task

Implement the kernel:

```cpp
__global__ void matmul_tiled_col_contiguous_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    int N,
    int TILE,
    int elems_per_thread
)
```

- `A, B` = input matrices in row-major flattened format ($N \times N$).
- `C` = output matrix in row-major flattened format ($N \times N$).
- `N` = side length of all three matrices.
- `TILE` = shared-memory tile size and block width/height (`blockDim.x == blockDim.y == TILE`).
- `elems_per_thread` = number of contiguous output rows computed by each thread (at most `MAX_ELEMS_PER_THREAD`).

**Requirements:**

- Use `extern __shared__` memory and partition it into `sA` (size `TILE * elems_per_thread * TILE`) and `sB` (size `TILE * TILE`).
- Compute `blockRow`, `blockCol`, `localRow`, `localCol`, `baseGlobalRow`, `globalCol`, and `threadRowStart`.
- Declare and zero-initialize an accumulator array `acc[elems_per_thread]` (use `MAX_ELEMS_PER_THREAD` as the storage size).
- Loop over all $K$-dimension tiles.
- Perform cooperative loading of `A` (contiguous `elems_per_thread` elements per thread) and `B` (one element per thread) into shared memory, with correct bounds checking.
- Synchronize correctly before and after the compute phase.
- Accumulate values into `acc` by reusing each `sB` value across all `elems_per_thread` multiply-accumulates.
- Write final results back to global memory, with bounds checking.

## Testing

The provided test:

- Initializes deterministic matrices $A$ and $B$.
- Launches your kernel via `matmul_gpu_col_contiguous`.
- Reports the GPU kernel execution time.

You can validate correctness by extending `main()` to also compute a CPU reference result and compare it element-wise against the GPU output. A test case should be considered a PASS only if **all** elements satisfy:

$$
|C_{\text{CPU}} - C_{\text{GPU}}| < 10^{-3}.
$$
