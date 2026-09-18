import torch
import numpy as np


# ============================================================
# Section 1: Tensor Creation
# ============================================================

def create_zeros(shape):
    """Return a tensor of the given shape filled with zeros."""
    return torch.zeros(shape)


def create_ones(shape):
    """Return a tensor of the given shape filled with ones."""
    return torch.ones(shape)


def create_arange(start, end, step):
    """Return a 1D tensor with values [start, end) spaced by step."""
    return torch.arange(start, end, step)


def create_linspace(start, end, steps):
    """Return a 1D tensor of `steps` values evenly spaced from start to end (inclusive)."""
    return torch.linspace(start, end, steps)


def create_identity(n):
    """Return an n x n identity matrix."""
    return torch.eye(n)


def create_random_uniform(shape, seed):
    """Return a tensor of the given shape with values ~ Uniform[0,1), seeded for reproducibility."""
    torch.manual_seed(seed)
    return torch.rand(shape)


def create_random_normal(shape, seed):
    """Return a tensor of the given shape with values ~ Normal(0,1), seeded for reproducibility."""
    torch.manual_seed(seed)
    return torch.randn(shape)


def create_full(shape, value):
    """Return a tensor of the given shape filled entirely with `value`."""
    return torch.full(shape, value)


# ============================================================
# Section 2: Tensor Properties & Type Conversion
# ============================================================

def get_tensor_info(tensor):
    """Return a dict with keys: shape (tuple), dtype, ndim, numel, device (str)."""
    return {
        "shape": tuple(tensor.shape),
        "dtype": tensor.dtype,
        "ndim": tensor.ndim,
        "numel": tensor.numel(),
        "device": str(tensor.device),
    }


def convert_dtype(tensor, dtype):
    """Return a copy of tensor cast to the given dtype."""
    return tensor.to(dtype)


def tensor_to_numpy(tensor):
    """Convert a CPU tensor to a numpy array."""
    return tensor.numpy()


def numpy_to_tensor(array):
    """Convert a numpy array to a torch tensor (sharing memory where possible)."""
    return torch.from_numpy(array)


# ============================================================
# Section 3: Indexing & Reshaping
# ============================================================

def get_row(tensor, idx):
    """Return row `idx` of a 2D tensor."""
    return tensor[idx]


def get_column(tensor, idx):
    """Return column `idx` of a 2D tensor."""
    return tensor[:, idx]


def slice_tensor(tensor, start, end):
    """Return elements [start:end] along dim 0."""
    return tensor[start:end]


def reshape_tensor(tensor, shape):
    """Return the tensor reshaped to `shape`."""
    return tensor.reshape(shape)


def flatten_tensor(tensor):
    """Return a 1D flattened version of the tensor."""
    return tensor.flatten()


def squeeze_tensor(tensor):
    """Remove all dimensions of size 1."""
    return tensor.squeeze()


def unsqueeze_tensor(tensor, dim):
    """Insert a dimension of size 1 at position `dim`."""
    return tensor.unsqueeze(dim)


def transpose_tensor(tensor, dim0, dim1):
    """Swap dimensions dim0 and dim1."""
    return tensor.transpose(dim0, dim1)


def permute_tensor(tensor, dims):
    """Permute the dimensions of the tensor according to `dims`."""
    return tensor.permute(*dims)


# ============================================================
# Section 4: Combining & Splitting
# ============================================================

def concatenate_tensors(tensors, dim):
    """Concatenate a list of tensors along an existing dimension `dim`."""
    return torch.cat(tensors, dim=dim)


def stack_tensors(tensors, dim):
    """Stack a list of tensors along a NEW dimension `dim`."""
    return torch.stack(tensors, dim=dim)


def split_tensor(tensor, split_size, dim):
    """Split tensor into chunks of `split_size` along `dim`. Returns a tuple of tensors."""
    return torch.split(tensor, split_size, dim=dim)


def chunk_tensor(tensor, chunks, dim):
    """Split tensor into exactly `chunks` (roughly) equal pieces along `dim`."""
    return torch.chunk(tensor, chunks, dim=dim)


# ============================================================
# Section 5: Math Operations
# ============================================================

def elementwise_add(a, b):
    return torch.add(a, b)


def elementwise_multiply(a, b):
    return torch.mul(a, b)


def matrix_multiply(a, b):
    """Standard matrix multiplication (not elementwise)."""
    return torch.matmul(a, b)


def compute_power(tensor, exponent):
    return torch.pow(tensor, exponent)


def compute_sqrt(tensor):
    return torch.sqrt(tensor)


def compute_exp(tensor):
    return torch.exp(tensor)


def compute_log(tensor):
    return torch.log(tensor)


def clamp_tensor(tensor, min_val, max_val):
    return torch.clamp(tensor, min=min_val, max=max_val)


# ============================================================
# Section 6: Reductions
# ============================================================

def sum_tensor(tensor, dim=None):
    if dim is None:
        return torch.sum(tensor)
    return torch.sum(tensor, dim=dim)


def mean_tensor(tensor, dim=None):
    if dim is None:
        return torch.mean(tensor)
    return torch.mean(tensor, dim=dim)


def max_tensor(tensor, dim=None):
    """Return just the max VALUE(S). If dim is given, reduce along that dim (no keepdim)."""
    if dim is None:
        return torch.max(tensor)
    return torch.max(tensor, dim=dim).values


def argmax_tensor(tensor, dim=None):
    if dim is None:
        return torch.argmax(tensor)
    return torch.argmax(tensor, dim=dim)


# ============================================================
# Section 7: Broadcasting & Comparison
# ============================================================

def broadcast_add(a, b):
    """Add two tensors relying on broadcasting rules (e.g. (3,1) + (1,4) -> (3,4))."""
    return a + b


def compare_greater(a, b):
    """Elementwise a > b, returns a boolean tensor."""
    return torch.gt(a, b)


def masked_select_positive(tensor):
    """Return a 1D tensor containing only the strictly positive elements of `tensor`."""
    return tensor[tensor > 0]


# ============================================================
# Section 8: In-place Operations
# ============================================================

def inplace_add(tensor, value):
    """Add `value` to `tensor` IN PLACE and return the same tensor object."""
    tensor.add_(value)
    return tensor


# ============================================================
# Section 9: Autograd
# ============================================================

def compute_gradient(x_value):
    """
    Given a scalar x_value, create a tensor x with requires_grad=True,
    compute y = x**2 + 3*x, backpropagate, and return x.grad (a tensor).
    """
    x = torch.tensor(float(x_value), requires_grad=True)
    y = x ** 2 + 3 * x
    y.backward()
    return x.grad


def linear_regression_step(x, w, b, y_true, lr):
    """
    Perform ONE step of gradient descent for a simple linear model y_pred = w*x + b,
    using mean squared error loss. w and b are 0-dim tensors with requires_grad=True.

    Must NOT track gradients for the parameter update itself, and must zero
    the gradients afterward so repeated calls behave correctly.

    Returns the updated (w, b) as plain tensors (requires_grad=True, grad cleared).
    """
    y_pred = w * x + b
    loss = torch.mean((y_pred - y_true) ** 2)

    if w.grad is not None:
        w.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad = None
    b.grad = None

    return w, b


# ============================================================
# Section 10: Device Management
# ============================================================

def get_device():
    """Return a torch.device: 'cuda' if a GPU is available, else 'cpu'."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_device(tensor, device):
    """Move `tensor` to the given device and return it."""
    return tensor.to(device)


def create_tensor_on_device(shape, device):
    """Create a tensor of zeros with the given shape directly on `device`."""
    return torch.zeros(shape, device=device)


# ============================================================
# Section 11: Reproducibility
# ============================================================

def set_seed(seed):
    """Set the global torch random seed (and cuda seed if available) for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
