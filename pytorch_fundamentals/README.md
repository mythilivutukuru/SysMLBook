# PyTorch Fundamentals Assignment

This assignment is a hands-on tour of the PyTorch functions you'll
use constantly in future assignments: creating tensors, reshaping and
indexing them, doing math on them, computing gradients, and moving things
to a GPU.

Each function you need to implement is a couple of lines — the goal isn't to write a lot of code, it's to make sure you know which PyTorch call to reach for.

---

## Setup

You should do this assignment on **Kaggle Notebooks** (see the
Kaggle Setup Guide if you're new to Kaggle). You need:

- Python 3.9+
- `torch` (PyTorch)
- `numpy`
- `pytest`

Kaggle notebooks already have `torch`, `numpy`, and most data science
packages preinstalled. If `pytest` isn't available, install it in a cell:

```shell
!pip install pytest
```

Upload the three files (`student.py`, `solution.py`,
`test_pytorch_assignment.py`) to your working directory, or clone/unzip the assignment folder into `/kaggle/working/`.

If you want to test GPU-specific behavior, remember to turn on an
**accelerator** (GPU T4 x2 or P100) in the Session Options panel — this
requires phone verification on your Kaggle account. The test suite will
automatically **skip** the CUDA-specific tests if no GPU is available, so
you can also do the whole assignment on CPU only.

---

## Tasks

Open `student.py`. Every function has:
1. A docstring describing exactly what it should return.
2. A `# TODO` comment with a hint about which `torch` call to use.
3. A `raise NotImplementedError` placeholder you should replace.

Work through the sections in order — later sections assume you're
comfortable with tensor basics from earlier ones.

### Sections

| # | Section | What it covers |
|---|---------|-----------------|
| 1 | Tensor Creation | `zeros`, `ones`, `arange`, `linspace`, `eye`, `rand`, `randn`, `full` |
| 2 | Properties & Type Conversion | `.shape`, `.dtype`, `.ndim`, `.numel()`, `.device`, `.to()`, numpy interop |
| 3 | Indexing & Reshaping | row/column access, slicing, `reshape`, `flatten`, `squeeze`, `unsqueeze`, `transpose`, `permute` |
| 4 | Combining & Splitting | `cat`, `stack`, `split`, `chunk` |
| 5 | Math Operations | elementwise add/multiply, `matmul`, `pow`, `sqrt`, `exp`, `log`, `clamp` |
| 6 | Reductions | `sum`, `mean`, `max`, `argmax` (with and without a `dim`) |
| 7 | Broadcasting & Comparison | broadcasting rules, `gt`, boolean masking |
| 8 | In-place Operations | `add_` and why in-place ops matter |
| 9 | Autograd | `requires_grad`, `.backward()`, `.grad`, a manual gradient-descent step |
| 10 | Device Management | detecting CUDA, `.to(device)`, creating tensors directly on a device |
| 11 | Reproducibility | `torch.manual_seed` |

---

## Testing

From the assignment root directory:

```bash
pytest test_pytorch_assignment.py -v
```

This runs the full suite against your `student.py`.

Run only one section while you're working through it (handy for focusing on
one topic at a time):

```bash
pytest test_pytorch_assignment.py -v -k Reshaping
pytest test_pytorch_assignment.py -v -k Autograd
pytest test_pytorch_assignment.py -v -k DeviceManagement
```

Class names you can filter on: `TestTensorCreation`,
`TestPropertiesAndConversion`, `TestIndexingAndReshaping`,
`TestCombiningAndSplitting`, `TestMathOperations`, `TestReductions`,
`TestBroadcastingAndComparison`, `TestInPlace`, `TestAutograd`,
`TestDeviceManagement`, `TestReproducibility`.

A passing run looks like:

```
test_pytorch_assignment.py::TestTensorCreation::test_zeros PASSED
test_pytorch_assignment.py::TestTensorCreation::test_ones PASSED
...
======================== 36 passed in 1.2s =========================
```

### Checking yourself against the reference solution

If you want to confirm the test suite itself is behaving as expected, you can point the tests at
`solution.py` instead of `student.py`:

```bash
IMPL_MODULE=solution pytest test_pytorch_assignment.py -v
```

This should always pass 100%.

---

## Notes

- **Read the docstring before the hint.** The hint tells you which function
  to call; the docstring tells you exactly what shape/values/behavior is
  expected.
- **`torch.max(tensor, dim=...)` returns a named tuple** `(values, indices)`,
  not just a tensor — don't forget to pull out `.values`.
- **In-place ops end with an underscore** (`add_`, `mul_`, `clamp_`, ...).
  They modify the tensor directly and return it, instead of allocating a
  new tensor.
- **Gradients accumulate by default.** Every call to `.backward()` *adds*
  to `.grad` rather than replacing it. That's why `linear_regression_step`
  explicitly zeroes gradients before backpropagating.
- **`torch.no_grad()`** tells autograd "don't track this" — you need it
  when manually updating parameters.

---
