# Convolutional Neural Network — CUDA Implementation

Convolutional Neural Networks (CNNs) are fundamental to modern computer vision tasks. This assignment focuses on implementing two core CUDA kernels:

1. **2D Convolution** (`conv2d_kernel`)
2. **ReLU Activation** (`relu_kernel`)

---

## Background

### 2D Convolution

Given an input matrix $I \in \mathbb{R}^{H \times W}$ and a kernel matrix $K \in \mathbb{R}^{k \times k}$, the convolution produces an output matrix $O \in \mathbb{R}^{H' \times W'}$ where:

$$H' = H - k + 1, \quad W' = W - k + 1$$

Each output element is computed as:

$$O(i, j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I[i+m,\ j+n] \cdot K[m, n]$$

> No padding is applied.

### ReLU Activation

Applied element-wise to each value:

$$\text{ReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$$

---

## Example

**Input Matrix (4×4):**
```
10   5   2   1
 8   4   1   0
 6   3  -1  -1
 4   2  -1  -2
```

**Kernel (3×3):**
```
0   0   0
0  -2   0
0   0   0
```

**Convolution Output (2×2):**
```
-8  -2
-6   2
```

**After ReLU:**
```
0   0
0   2
```

The kernel multiplies the center value of each 3×3 region by −2. For position (0,0), the center value is 4, giving −2 × 4 = −8. ReLU then zeros out all negative values.

---

## Tasks

### Kernel 1: `conv2d_kernel`

```cuda
__global__ void conv2d_kernel(
    const float* input,
    const float* kernel,
    float* output,
    int height,
    int width,
    int kernel_size
)
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `input` | Input matrix, row-major flattened (size: H × W) |
| `kernel` | Convolution kernel, row-major flattened (size: k × k) |
| `output` | Output matrix, row-major flattened (size: H' × W') |
| `height` | Height of the input matrix (H) |
| `width` | Width of the input matrix (W) |
| `kernel_size` | Size of the square kernel (k) |

**Requirements:**
- Use a **2D grid of 2D thread blocks**
- Each thread computes **one output element**
- Handle boundary conditions: threads outside the valid output range must do nothing
- Access data in row-major order: `input[row * width + col]`
- Assume H, W ≥ k

---

### Kernel 2: `relu_kernel`

```cuda
__global__ void relu_kernel(
    float* data,
    int size
)
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `data` | Array to apply ReLU in-place |
| `size` | Total number of elements in the array |

**Requirements:**
- Use a **1D grid of 1D thread blocks**
- Each thread processes **one element**
- Modify in-place: `data[idx] = fmaxf(0.0f, data[idx])`
- Use `fmaxf()` for the max operation

---

## Testing

The provided `main()` in `student.cu` runs automated test cases that:

1. Run the convolution kernel and verify output against expected values
2. Apply the ReLU kernel and verify the final result
3. Report pass/fail for each stage

A test case passes only if all outputs match expected values within a tolerance of **ε = 10⁻⁴**.

---

## Notes

- **Only modify the two kernel functions.** Do not change any other code.
- All matrices are stored in **row-major flattened format**. Matrix dimensions are not required to be powers of 2.
- This follows the deep learning convention of **cross-correlation** (kernel is not flipped).
- Ensure proper memory access patterns to avoid out-of-bounds accesses.
- The main function handles all memory allocation, device transfers, kernel launches, and result comparison — you do not need to modify any of that.