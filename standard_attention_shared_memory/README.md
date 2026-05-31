# Standard Attention Using Shared Memory

Standard self-attention kernels are often written with each thread independently recomputing every dot product, resulting in redundant global memory reads. In this assignment, you will implement an optimized single-block CUDA kernel for self-attention that exploits shared memory and a softmax reduction to eliminate this redundancy.

---

## Background

Given queries $Q \in \mathbb{R}^{B \times H \times N \times D}$, keys $K \in \mathbb{R}^{B \times H \times N \times D}$, and values $V \in \mathbb{R}^{B \times H \times N \times D}$, where $B$ is the batch size, $H$ is the number of heads, $N$ is the sequence length, and $D$ is the head dimension, the output $O \in \mathbb{R}^{B \times H \times N \times D}$ is:

$$O[b,h,i,x] = \sum_{j=0}^{N-1} \alpha_j \cdot V[b,h,j,x]$$

where the attention weights $\alpha_j$ are computed via softmax over the raw scores:

$$s_j = \texttt{scale} \cdot \sum_{k=0}^{D-1} Q[b,h,i,k] \cdot K[b,h,j,k], \qquad \texttt{scale} = \frac{1}{\sqrt{D}}$$

$$m = \max_{j} s_j, \qquad Z = \sum_{j=0}^{N-1} e^{s_j - m}, \qquad \alpha_j = \frac{e^{s_j - m}}{Z}$$

All tensors are stored in a row-major flattened format with layout $[B, H, N, D]$, so element $(b, h, i, x)$ of any tensor `X` resides at:

$$\texttt{X}\bigl[(b \cdot H + h) \cdot N \cdot D + i \cdot D + x\bigr]$$

---

## Task

### Constraints

This kernel targets small problems where both the sequence length and head dimension fit within a single warp: $N \leq 32$ and $D \leq 32$.

The grid and block are configured as follows:

- **Grid:** $(N,\ H,\ B)$ — one block per query token per head per batch item.
- **Block:** $(D,\ 1,\ 1)$ — exactly $D$ threads, one per feature dimension.

Each block handles one full output row $O[b, h, i, :]$.

### Visualization

<!-- Add your visualization image here -->

---



You must implement the body of the following CUDA kernel in `student.cu`:

```cuda
__global__ void standard_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float*       O,
    int B, int NH, int N, int D
)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `Q`, `K`, `V` | Query, key, and value tensors in row-major flattened format with shape $[B, H, T, D]$ |
| `O` | Output tensor of the same shape as `Q`, to be written by the kernel |
| `B` | Batch size |
| `H` | Number of attention heads |
| `T` | Sequence length |
| `D` | Head dimension |

The grid and block configuration, memory allocation, data transfers, and correctness checks are all provided — **do not modify any other code**.

---

## Implementation Requirements

Your kernel must follow this four-step structure:

**Step 1 — Identify indices.**
Derive $b$, $h$, $i$ from the built-in block indices and $x$ from `threadIdx.x`. Here, $i = \texttt{blockIdx.x}$ is the *query token index* — it identifies which row of $Q$ this entire block is responsible for computing the output of, with every thread in the block sharing the same $i$ but each owning a different feature coordinate $x$. Compute `base = (b*NH + h)*N*D`.

**Step 2 — Each thread computes one attention score (no shared Q).**
Thread $x$ (for $x < N$) reads query row $Q[b,h,i,:]$ directly from global memory and computes the dot product with $K[b,h,x,:]$, storing the scaled result:

$$s_x = \frac{1}{\sqrt{D}} \sum_{k} Q[\texttt{base}+i \cdot D+k] \cdot K[\texttt{base}+x \cdot D+k]$$

into `sS[x]`. All $N$ scores are computed simultaneously across threads. Issue a `__syncthreads()` so that thread 0 sees all scores.

**Step 3 — Thread 0 performs the softmax serially.**
Thread 0 alone scans `sS[0..N-1]` to find the maximum $m$, then makes a second pass to compute $e^{s_j - m}$ and the normalization constant $Z = \sum_j e^{s_j - m}$, overwriting `sS[j]` with the final weight $\alpha_j = e^{s_j-m}/Z$. Issue a `__syncthreads()` before the accumulation step so all threads see the updated weights.

**Step 4 — All threads accumulate the output in parallel.**
Thread $x$ (for $x < D$) computes:

$$O[b,h,i,x] = \sum_{j=0}^{N-1} \alpha_j \cdot V[b,h,j,x]$$

and writes the result to `O[base + i*D + x]`. Each thread iterates over all $N$ value rows, reading $\alpha_j$ from shared memory and $V[b,h,j,x]$ from global memory, accumulating the weighted sum into a local register before the final write.

---

## Evaluation and Grading

Your kernel is evaluated against a CPU reference on several configurations (varying $B$, $H$, $N$, and $D$, all satisfying $N \leq 32$ and $D \leq 32$). A test passes if every output element agrees with the reference within $\varepsilon = 10^{-4}$. Partial marks are awarded based on the number of passing tests.

---

## Notes

- You must **only modify** the kernel function `standard_attention_kernel`. Do not change any other code. You only need to submit the file `student.cu`.
- The grid and block configuration, device memory allocation, data transfers, and result comparison are all handled by the provided launcher and test infrastructure.
- Your kernel will be tested on configurations beyond the three provided test cases, including varying batch sizes, head counts, sequence lengths, and head dimensions. $N \leq 32$ and $D \leq 32$ across all test cases.
- You do **not** need to apply the causal mask for this question.
- $N$ and $D$ are not necessarily powers of 2 (or divisible by each other).
