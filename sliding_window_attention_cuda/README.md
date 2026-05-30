# Sliding Window Attention

Transformer-based language models rely on self-attention to capture relationships between tokens. However, standard full self-attention has quadratic complexity in sequence length, which becomes prohibitive for long sequences. **Sliding window attention** addresses this by restricting each token to attend only to a fixed-size causal window of its most recent predecessors. In this assignment, you will implement a CUDA kernel for causal sliding window attention.

## Background

Given queries $Q \in \mathbb{R}^{B \times H \times N \times D}$, keys $K \in \mathbb{R}^{B \times H \times N \times D}$, and values $V \in \mathbb{R}^{B \times H \times N \times D}$ (where $B$ is the batch size, $H$ is the number of attention heads, $N$ is the sequence length, and $D$ is the head dimension), the output $Y \in \mathbb{R}^{B \times H \times N \times D}$ is computed as follows.

For each query position $t$ and head $(b, h)$, the attention window spans key positions $k \in \{k_{\text{start}}, \ldots, t\}$, where:

$$k_{\text{start}} = \max(0,\ t - W + 1)$$

and $W$ is the window size. The attention score for key position $k$ is:

$$s_k = \text{scale} \cdot \sum_{d=0}^{D-1} Q[b, h, t, d] \cdot K[b, h, k, d], \qquad \text{scale} = \frac{1}{\sqrt{D}}$$

The attention weights are computed using a numerically stable softmax over the window:

$$m = \max_{k \in [k_{\text{start}},\, t]} s_k, \qquad Z = \sum_{k=k_{\text{start}}}^{t} e^{s_k - m}, \qquad \alpha_k = \frac{e^{s_k - m}}{Z}$$

The output for each dimension $d$ is:

$$Y[b, h, t, d] = \sum_{k=k_{\text{start}}}^{t} \alpha_k \cdot V[b, h, k, d]$$

### Example

Consider a single batch, single head ($B = H = 1$), sequence length $N = 4$, head dimension $D = 1$, and window size $W = 2$. With $Q = K = V = [1, 2, 3, 4]^{\top}$ and scale $= 1$:

| **Token $t$** | **Window $[k_{\text{start}}, t]$** | **Scores $s_k$** | **Output $Y[t]$** |
|:---:|:---:|:---:|:---:|
| 0 | $[0, 0]$ | $1 \cdot 1 = 1$ | $\alpha_0 \cdot 1 = 1.0$ |
| 1 | $[0, 1]$ | $2, 4$ | $\approx 0.12 \cdot 1 + 0.88 \cdot 2 \approx 1.88$ |
| 2 | $[1, 2]$ | $6, 9$ | $\approx 0.05 \cdot 2 + 0.95 \cdot 3 \approx 2.95$ |
| 3 | $[2, 3]$ | $12, 16$ | $\approx 0.02 \cdot 3 + 0.98 \cdot 4 \approx 3.98$ |

The causal constraint ensures that token $t$ only attends to positions $\leq t$, and the window constraint further limits this to the most recent $W$ positions.

## Task

You must implement the following CUDA kernel:

### Kernel: `sliding_window_attention_kernel`

**Function signature:**

```cuda
__global__ void sliding_window_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* Y,
    int B,
    int H,
    int N,
    int D,
    int W,
    float scale
)
```

**Parameters:**

- `Q`, `K`, `V`: Query, key, and value tensors in row-major flattened format with shape $[B, H, N, D]$.
- `Y`: Output tensor of the same shape as `Q`, to be written by the kernel.
- `B`: Batch size.
- `H`: Number of attention heads.
- `N`: Sequence length.
- `D`: Head dimension.
- `W`: Attention window size.
- `scale`: Scaling factor applied to attention scores (equal to $1/\sqrt{D}$).

**Grid and block configuration (already set in the provided launcher):**

- Grid dimensions: $(B,\ H,\ \lceil N / \texttt{BLOCK} \rceil)$, where `BLOCK` $= 128$.
- Block dimensions: $(\texttt{BLOCK},\ 1,\ 1)$.
- Each thread is responsible for exactly one query token $t$ and performs all the computations related to it without syncing with other threads of the same block.

**Requirements:**

- Identify the batch index $b$, head index $h$, and token index $t$ from the built-in block/thread indices.
- Guard against out-of-bounds token indices (i.e., return early if $t \geq N$).
- Compute the window boundaries: $k_{\text{start}} = \max(0, t - W + 1)$ and $k_{\text{end}} = t$.
- Use a numerically stable three-pass softmax over the window: (1) find the maximum score, (2) compute the sum of exponentials, and (3) accumulate the weighted sum over `V`.
- Access tensors using the row-major offset `head_offset = b * (H * N * D) + h * (N * D)`, so that `Q[head_offset + t * D + d]` gives the $d$-th dimension of the query token $t$.
- Write the result to `Y[head_offset + t * D + d]` for each dimension $d \in \{0, \ldots, D-1\}$.

## Testing

Your implementation will be tested using the provided test cases in `main()`:

- Each test case runs the GPU kernel and compares its output element-wise against the CPU reference implementation.
- A test case passes only if all output elements match within a tolerance of $\epsilon = 10^{-4}$.
- Results are reported as `PASS` or `FAIL` for each configuration.

## Notes

- You must only modify the kernel function `sliding_window_attention_kernel`. Do not change any other code.
- The grid and block configuration, device memory allocation, data transfers, and result comparison are all handled by the provided launcher and test infrastructure. 
- Ensure proper boundary handling: the window must be clamped to $[0, t]$, so tokens near the beginning of the sequence will have a shorter effective window.
- All tensors are stored in row-major flattened format with layout $[B, H, T, D]$.
- You must use standard CUDA math functions such as `expf()`.
- The sequence length $N$ is not required to be a multiple of the block size.