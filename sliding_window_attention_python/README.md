# Sliding Window Attention

Modern transformer-based neural networks rely heavily on the mechanism of **self-attention**. In standard self-attention, every token in a sequence attends to every other token. This enables models to capture long-range dependencies, but it comes at a high computational and memory cost. To overcome this limitation, many efficient transformer architectures restrict attention to only a subset of tokens.

One common approach is **Sliding Window Attention**, where each token can attend only to a fixed number of previous tokens. In this question, you will implement a **Sliding Window Causal Self-Attention Layer**, which combines two constraints:

- **Causality:** A token cannot attend to future tokens.
- **Sliding Window:** A token can attend only to a limited number of previous tokens.

You have been given the following files in the this sub-directory:

- `student.py` - contains the class and it's forward function, which you need to change.
- `evaluate.py` - an evaluation script to test correctness
- `client_test_cases.json` - a few test cases, along with the correct outputs, which in turn are used by the evaluation script to test your implementation

## Background

Given an input sequence represented as a tensor:

$$
X \in \mathbb{R}^{B \times T \times C}
$$

where $B$ = batch size, $T$ = sequence length, and $C$ = embedding dimension (it is denoted as $E$ in the lecture slides).

Self-attention computes:

**(a) Linear projections:**

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

**(b) Scaled dot-product attention scores:**

$$
\text{scores} = \frac{QK^T}{\sqrt{D}}
$$

**(c) Attention weights:**

$$
A = \text{softmax}(\text{scores})
$$

**(d) Final output:**

$$
Y = AV
$$

In this question, attention is further restricted using a **window size** $W$. A token at position $i$ may attend only to tokens in the range $\max(0, i - W + 1) \leq j \leq i$. This enforces: (i) No attention to future tokens (causality) and (ii) No attention to tokens more than $W-1$ steps in the past.

In other words, attention scores are computed as:

$$
\text{scores}_{ij} =
\begin{cases}
\dfrac{Q_i K_j^T}{\sqrt{D}} & \text{if } \max(0, i-W+1) \le j \le i \\
-\infty & \text{otherwise}
\end{cases}
$$

### Example

Let:

- Sequence length $T = 6$
- Window size $W = 3$

**Allowed attention positions**

| Token Index | Can Attend To |
|---|---|
| 0 | [0] |
| 1 | [0,1] |
| 2 | [0,1,2] |
| 3 | [1,2,3] |
| 4 | [2,3,4] |
| 5 | [3,4,5] |

**Corresponding mask matrix**

$$
\begin{array}{c|cccccc}
 & K_1 & K_2 & K_3 & K_4 & K_5 & K_6 \\
\hline
Q_1 & 1 & 0 & 0 & 0 & 0 & 0 \\
Q_2 & 1 & 1 & 0 & 0 & 0 & 0 \\
Q_3 & 1 & 1 & 1 & 0 & 0 & 0 \\
Q_4 & 0 & 1 & 1 & 1 & 0 & 0 \\
Q_5 & 0 & 0 & 1 & 1 & 1 & 0 \\
Q_6 & 0 & 0 & 0 & 1 & 1 & 1
\end{array}
$$

## Task

You are provided with a partially implemented PyTorch module:

```python
class SlidingWindowCausalSelfAttention(nn.Module):
```

Your task is to implement the `forward()` method so that it correctly performs multi-head self-attention with causal masking and the sliding window restriction.

The forward method receives $x \in \mathbb{R}^{B \times T \times C}$, and must return $y \in \mathbb{R}^{B \times T \times C}$ with the same shape as the input.

Your implementation must work for all valid parameter combinations satisfying:

- $B \ge 1$ : batch size
- $T \ge 1$ : sequence length
- $C \ge 1$ : embedding dimension
- $H \ge 1$ : number of attention heads
- $C \bmod H = 0$ : embedding dimension must be divisible by number of heads
- $D = C/H$ : model dimension per head
- $W \ge 1$ : window size might exceed the sequence length, which essentially means that we are just applying causal masking

## Testing

Your submission will be evaluated using the provided evaluation script `evaluate.py`. For each test case, the script first sets a random seed to ensure reproducibility, then constructs your model with specific parameters and generates a fixed random input tensor. A forward pass is executed using this input, and the resulting output is compared against a reference output. The comparison checks include the output shape, the mean value, the standard deviation, the sum of all elements, and the first ten values of the flattened output tensor. A test case is considered to pass only if all of these checks fall within a specified tolerance.

To test your implementation locally using our evaluation script, do the following:

1. Make sure the following files are in the same directory: `student.py`, `evaluate.py`, and `client_test_cases.json`.
2. Execute `python3 evaluate.py`.
3. The grader will display:

```
Running N test cases...

Test 1 PASSED
Test 2 FAILED
...

RESULTS: X/N tests passed
```
