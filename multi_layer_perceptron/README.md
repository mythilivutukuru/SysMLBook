# Multi-Layer Perceptron

A **Multi-Layer Perceptron (MLP)** is one of the foundational architectures in deep learning. It consists of stacked layers of linear transformations along with non-linear activation functions, allowing the network to learn complex mappings from inputs to outputs. In this assignment, you will implement the core components of a two-layer MLP trained with stochastic gradient descent (SGD): a forward pass, a mean-squared-error loss, a backpropagation routine, and a parameter update step.


## Background
The network has the following fixed hyper-parameters:

| Symbol | Meaning |
|---|---|
| $d_{\text{in}} = 8$ | Input dimension |
| $d_{\text{h}} = 16$ | Hidden dimension |
| $d_{\text{out}} = 4$ | Output dimension |
| $B = 32$ | Batch size |
| $\eta = 0.01$ | Learning rate |

Given an input matrix $X \in \mathbb{R}^{B \times d_{\text{in}}}$, the forward pass computes:

**(a) Hidden pre-activation:**

$$
z_1 = X W_1 + b_1, \qquad W_1 \in \mathbb{R}^{d_{\text{in}} \times d_{\text{h}}},\; b_1 \in \mathbb{R}^{d_{\text{h}}}
$$

**(b) ReLU activation:**

$$
a_1 = \mathrm{ReLU}(z_1) = \max(0,\, z_1) \quad \text{(element-wise)}
$$

**(c) Output logits:**

$$
\hat{Y} = a_1 W_2 + b_2, \qquad W_2 \in \mathbb{R}^{d_{\text{h}} \times d_{\text{out}}},\; b_2 \in \mathbb{R}^{d_{\text{out}}}
$$

### Loss Function

Training uses the **Mean Squared Error (MSE)** loss between the network's predictions $\hat{Y}$ and the targets $Y \in \mathbb{R}^{B \times d_{\text{out}}}$:

$$
\mathcal{L} = \frac{1}{B \cdot d_{\text{out}}} \sum_{i=1}^{B} \sum_{k=1}^{d_{\text{out}}} \left(\hat{Y}_{ik} - Y_{ik}\right)^2
= \texttt{mean}\!\left((\hat{Y} - Y)^2\right)
$$

### Backpropagation

Gradients are propagated from the loss back through each layer using the chain rule. The full derivation is as follows:

1. **Gradient of loss w.r.t. logits:**

$$
\frac{\partial \mathcal{L}}{\partial \hat{Y}} = \frac{2(\hat{Y} - Y)}{B}
$$

2. **Output layer gradients:**

$$
\frac{\partial \mathcal{L}}{\partial W_2} = a_1^\top \frac{\partial \mathcal{L}}{\partial \hat{Y}}, \qquad
\frac{\partial \mathcal{L}}{\partial b_2} = \sum_{\text{batch}} \frac{\partial \mathcal{L}}{\partial \hat{Y}}
$$

3. **Gradient through ReLU gate:**

$$
\frac{\partial \mathcal{L}}{\partial a_1} = \frac{\partial \mathcal{L}}{\partial \hat{Y}} W_2^\top, \qquad
\frac{\partial \mathcal{L}}{\partial z_1} = \frac{\partial \mathcal{L}}{\partial a_1} \odot \mathbf{1}[z_1 > 0]
$$

4. **Hidden layer gradients:**

$$
\frac{\partial \mathcal{L}}{\partial W_1} = X^\top \frac{\partial \mathcal{L}}{\partial z_1}, \qquad
\frac{\partial \mathcal{L}}{\partial b_1} = \sum_{\text{batch}} \frac{\partial \mathcal{L}}{\partial z_1}
$$

### Parameter Update (SGD)

After computing all gradients, the parameters are updated with a single SGD step:

$$
\theta \;\leftarrow\; \theta - \eta \cdot \nabla_\theta \mathcal{L}
$$

where $\theta$ denotes the concatenated flat parameter vector $[W_1,\, b_1,\, W_2,\, b_2]$ (in that order).

## Task

You are given the file `student.py` which contains the `MLP` class skeleton with `TODO` markers indicating the sections you must complete.

The `MLP` class exposes the following methods:

| Method | Description |
|---|---|
| `params_flat()` | Returns all parameters as a single 1-D `numpy` array in the order $[W_1, b_1, W_2, b_2]$ (each flattened row-major). Flat vectors will be needed later for inter-process communication. |
| `set_params_flat(flat)` | Loads parameters from a flat array, the inverse of `params_flat()`. |
| `grad_size()` | Returns the total number of scalar parameters (i.e. the length of the flat gradient vector). |
| `forward(X)` | Runs the two-layer forward pass and returns `(logits, cache)` where `cache = (X, z1, a1, logits)` stores intermediate values needed for backprop. |
| `loss(logits, Y)` | Computes and returns the scalar MSE loss. |
| `backward(cache, Y)` | Runs backpropagation and returns a flat gradient vector with the same layout as `params_flat()`. |
| `apply_gradients(flat_grad, lr)` | Applies one SGD step to all parameters. |


Open `student.py` and complete each section marked `# TODO`. Specifically, you must implement: **`forward(X)`**, **`loss(logits, Y)`**, **`backward(cache, Y)`**, and **`apply_gradients(flat_grad, lr)`**.

**Constraints:** Use only `numpy` operations. No Python loops over the batch dimension are permitted; your implementation must be fully vectorised.

## Testing

Seven automated tests are included at the bottom of `student.py`. They are run by executing the script directly:

```
python3 student.py
```
