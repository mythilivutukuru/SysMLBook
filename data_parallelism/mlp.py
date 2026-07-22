import numpy as np

INPUT_DIM = 8
HIDDEN_DIM = 16
OUTPUT_DIM = 4
BATCH_SIZE = 32
LR = 0.01


class MLP:

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((INPUT_DIM, HIDDEN_DIM)) * 0.1
        self.b1 = np.zeros(HIDDEN_DIM)
        self.W2 = rng.standard_normal((HIDDEN_DIM, OUTPUT_DIM)) * 0.1
        self.b2 = np.zeros(OUTPUT_DIM)

    def params_flat(self) -> np.ndarray:
        return np.concatenate(
            [self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), self.b2.ravel()]
        )

    def set_params_flat(self, flat: np.ndarray):
        idx = 0

        def _take(shape):
            nonlocal idx
            n = int(np.prod(shape))
            chunk = flat[idx : idx + n].reshape(shape)
            idx += n
            return chunk

        self.W1 = _take(self.W1.shape)
        self.b1 = _take(self.b1.shape)
        self.W2 = _take(self.W2.shape)
        self.b2 = _take(self.b2.shape)

    def grad_size(self) -> int:
        return self.params_flat().size

    def forward(self, X: np.ndarray):
        """Two-layer MLP forward pass.

        Layer 1 : z1 = X @ W1 + b1,  a1 = ReLU(z1)
        Layer 2 : logits = a1 @ W2 + b2

        Returns
        -------
        logits : (batch, OUTPUT_DIM)
        cache  : (X, z1, a1, logits)  — stored for backward
        """
        # TODO: Implement the two-layer forward pass.
        #   Layer 1: z1 = X @ W1 + b1,  a1 = ReLU(z1)
        #   Layer 2: logits = a1 @ W2 + b2
        #   Return (logits, (X, z1, a1, logits))
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        logits = a1 @ self.W2 + self.b2
        cache = (X, z1, a1, logits)
        return logits, cache

    def loss(self, logits: np.ndarray, Y: np.ndarray) -> float:
        # TODO: Compute and return MSE loss.
        return float(np.mean((logits - Y) ** 2))

    def backward(self, cache, Y: np.ndarray) -> np.ndarray:
        """Backprop through MSE → linear → ReLU → linear.

        Returns a flat gradient vector with the same layout as
        params_flat(): [dW1, db1, dW2, db2].
        """
        X, z1, a1, logits = cache
        batch = X.shape[0]

        # TODO: Implement backprop.
        #   dL/dlogits = 2*(logits - Y)/batch
        #   Output layer:  dW2 = a1.T @ dL_dlogits,  db2 = sum over batch
        #   ReLU gate:     dL_dz1 = dL_da1 * (z1 > 0)
        #   Hidden layer:  dW1 = X.T @ dL_dz1,  db1 = sum over batch
        #   Return concatenated flat gradient vector
        dL_dlogits = 2 * (logits - Y) / batch
        dW2 = a1.T @ dL_dlogits
        db2 = dL_dlogits.sum(axis=0)
        dL_da1 = dL_dlogits @ self.W2.T
        dL_dz1 = dL_da1 * (z1 > 0)
        dW1 = X.T @ dL_dz1
        db1 = dL_dz1.sum(axis=0)
        return np.concatenate([dW1.ravel(), db1.ravel(), dW2.ravel(), db2.ravel()])

    def apply_gradients(self, flat_grad: np.ndarray, lr: float = LR):
        # TODO: Apply SGD step: params = params - lr * flat_grad
        self.set_params_flat(self.params_flat() - lr * flat_grad)


# ======================================================================
# TESTING: PLEASE DO NOT MODIFY BELOW THIS LINE
# ======================================================================


def _header(title: str):
    bar = "=" * 55
    print(f"\n{bar}\n  {title}\n{bar}")


def test_param_roundtrip():
    _header("TEST 1")
    model = MLP(seed=0)
    original = model.params_flat().copy()
    model.set_params_flat(original)
    recovered = model.params_flat()
    err = np.max(np.abs(original - recovered))
    status = "PASS" if err < 1e-12 else "FAIL"
    print(f"  Max absolute error : {err:.2e}  [{status}]")
    return status == "PASS"


def test_grad_size():
    _header("TEST 2")
    model = MLP(seed=0)
    expected = (
        INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * OUTPUT_DIM + OUTPUT_DIM
    )
    got = model.grad_size()
    status = "PASS" if got == expected else "FAIL"
    print(f"  Expected : {expected}")
    print(f"  Got      : {got}  [{status}]")
    return status == "PASS"


def test_forward_shapes():
    _header("TEST 3")
    model = MLP(seed=1)
    rng = np.random.default_rng(7)
    X = rng.standard_normal((BATCH_SIZE, INPUT_DIM))
    logits, (cX, z1, a1, cl) = model.forward(X)

    checks = {
        "logits": (logits.shape, (BATCH_SIZE, OUTPUT_DIM)),
        "z1": (z1.shape, (BATCH_SIZE, HIDDEN_DIM)),
        "a1": (a1.shape, (BATCH_SIZE, HIDDEN_DIM)),
    }
    all_ok = True
    for name, (got, exp) in checks.items():
        ok = got == exp
        print(
            f"  {name:8s}: got {str(got):20s} expected {str(exp):20s}  "
            f"[{'PASS' if ok else 'FAIL'}]"
        )
        all_ok = all_ok and ok
    return all_ok


def test_relu_non_negative():
    _header("TEST 4")
    model = MLP(seed=2)
    rng = np.random.default_rng(99)
    X = rng.standard_normal((BATCH_SIZE, INPUT_DIM))
    _, (_, _, a1, _) = model.forward(X)
    neg_count = int((a1 < 0).sum())
    status = "PASS" if neg_count == 0 else "FAIL"
    print(f"  Negative activations : {neg_count}  [{status}]")
    return status == "PASS"


def test_loss_zero_on_perfect_prediction():
    _header("TEST 5")
    model = MLP(seed=3)
    rng = np.random.default_rng(5)
    X = rng.standard_normal((BATCH_SIZE, INPUT_DIM))
    logits, _ = model.forward(X)
    loss_val = model.loss(logits, logits)
    status = "PASS" if loss_val < 1e-12 else "FAIL"
    print(f"  Loss : {loss_val:.2e}  [{status}]")
    return status == "PASS"


def test_backward_shape():
    _header("TEST 6")
    model = MLP(seed=4)
    rng = np.random.default_rng(11)
    X = rng.standard_normal((BATCH_SIZE, INPUT_DIM))
    Y = rng.standard_normal((BATCH_SIZE, OUTPUT_DIM))
    logits, cache = model.forward(X)
    grad = model.backward(cache, Y)
    exp = model.grad_size()
    ok = grad.shape == (exp,)
    print(f"  Expected shape : ({exp},)")
    print(f"  Got shape      : {grad.shape}  [{'PASS' if ok else 'FAIL'}]")
    return ok


def test_apply_gradients_updates_params():
    _header("TEST 7")
    model = MLP(seed=7)
    before = model.params_flat().copy()
    grad = np.ones(model.grad_size())
    model.apply_gradients(grad, lr=0.1)
    after = model.params_flat()
    changed = not np.array_equal(before, after)
    expected_delta = np.linalg.norm(before - after)
    ok = changed and abs(expected_delta - 0.1 * np.linalg.norm(grad)) < 1e-10
    print(f"  Parameters changed : {changed}")
    print(f"  Delta norm correct : {ok}  [{'PASS' if ok else 'FAIL'}]")
    return ok


if __name__ == "__main__":
    tests = [
        test_param_roundtrip,
        test_grad_size,
        test_forward_shapes,
        test_relu_non_negative,
        test_loss_zero_on_perfect_prediction,
        test_backward_shape,
        test_apply_gradients_updates_params,
    ]

    results = [t() for t in tests]

    passed = sum(results)
    total = len(results)
    bar = "=" * 55
    print(f"\n{bar}")
    print(f"  SUMMARY : {passed}/{total} tests passed")
    print(bar + "\n")
