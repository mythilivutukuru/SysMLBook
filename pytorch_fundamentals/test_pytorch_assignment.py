import os
import importlib
import sys
import math

import numpy as np
import pytest
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

IMPL_NAME = os.environ.get("IMPL_MODULE", "student")
mod = importlib.import_module(IMPL_NAME)

HAS_CUDA = torch.cuda.is_available()


# ============================================================
# Section 1: Tensor Creation
# ============================================================
class TestTensorCreation:
    def test_zeros(self):
        t = mod.create_zeros((2, 3))
        assert t.shape == (2, 3)
        assert torch.equal(t, torch.zeros(2, 3))

    def test_ones(self):
        t = mod.create_ones((4,))
        assert t.shape == (4,)
        assert torch.equal(t, torch.ones(4))

    def test_arange(self):
        t = mod.create_arange(0, 10, 2)
        assert torch.equal(t, torch.tensor([0, 2, 4, 6, 8]))

    def test_linspace(self):
        t = mod.create_linspace(0, 1, 5)
        expected = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        assert torch.allclose(t, expected)
        assert t.numel() == 5

    def test_identity(self):
        t = mod.create_identity(3)
        assert torch.equal(t, torch.eye(3))
        assert t.shape == (3, 3)

    def test_random_uniform_reproducible(self):
        t1 = mod.create_random_uniform((3, 3), seed=42)
        t2 = mod.create_random_uniform((3, 3), seed=42)
        assert torch.equal(t1, t2), "Same seed should give identical tensors"
        assert t1.shape == (3, 3)
        assert torch.all((t1 >= 0) & (t1 < 1))

    def test_random_normal_reproducible(self):
        t1 = mod.create_random_normal((5,), seed=7)
        t2 = mod.create_random_normal((5,), seed=7)
        assert torch.equal(t1, t2)
        assert t1.shape == (5,)

    def test_full(self):
        t = mod.create_full((2, 2), 7.0)
        assert torch.equal(t, torch.full((2, 2), 7.0))


# ============================================================
# Section 2: Properties & Type Conversion
# ============================================================
class TestPropertiesAndConversion:
    def test_get_tensor_info(self):
        t = torch.zeros(2, 3, dtype=torch.float32)
        info = mod.get_tensor_info(t)
        assert info["shape"] == (2, 3)
        assert info["dtype"] == torch.float32
        assert info["ndim"] == 2
        assert info["numel"] == 6
        assert info["device"] == "cpu"

    def test_convert_dtype(self):
        t = torch.tensor([1, 2, 3], dtype=torch.int64)
        t2 = mod.convert_dtype(t, torch.float32)
        assert t2.dtype == torch.float32
        assert torch.equal(t2, torch.tensor([1.0, 2.0, 3.0]))

    def test_tensor_to_numpy(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        arr = mod.tensor_to_numpy(t)
        assert isinstance(arr, np.ndarray)
        assert np.allclose(arr, np.array([1.0, 2.0, 3.0]))

    def test_numpy_to_tensor(self):
        arr = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        t = mod.numpy_to_tensor(arr)
        assert isinstance(t, torch.Tensor)
        assert torch.allclose(t, torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32))


# ============================================================
# Section 3: Indexing & Reshaping
# ============================================================
class TestIndexingAndReshaping:
    @pytest.fixture
    def sample_2d(self):
        return torch.arange(12).reshape(3, 4).float()

    def test_get_row(self, sample_2d):
        row = mod.get_row(sample_2d, 1)
        assert torch.equal(row, sample_2d[1])

    def test_get_column(self, sample_2d):
        col = mod.get_column(sample_2d, 2)
        assert torch.equal(col, sample_2d[:, 2])

    def test_slice_tensor(self, sample_2d):
        s = mod.slice_tensor(sample_2d, 0, 2)
        assert torch.equal(s, sample_2d[0:2])

    def test_reshape(self, sample_2d):
        r = mod.reshape_tensor(sample_2d, (2, 6))
        assert r.shape == (2, 6)
        assert torch.equal(r, sample_2d.reshape(2, 6))

    def test_flatten(self, sample_2d):
        f = mod.flatten_tensor(sample_2d)
        assert f.shape == (12,)

    def test_squeeze(self):
        t = torch.zeros(1, 3, 1, 4)
        s = mod.squeeze_tensor(t)
        assert s.shape == (3, 4)

    def test_unsqueeze(self):
        t = torch.zeros(3, 4)
        u = mod.unsqueeze_tensor(t, 0)
        assert u.shape == (1, 3, 4)

    def test_transpose(self, sample_2d):
        tr = mod.transpose_tensor(sample_2d, 0, 1)
        assert tr.shape == (4, 3)
        assert torch.equal(tr, sample_2d.T)

    def test_permute(self):
        t = torch.zeros(2, 3, 4)
        p = mod.permute_tensor(t, (2, 0, 1))
        assert p.shape == (4, 2, 3)


# ============================================================
# Section 4: Combining & Splitting
# ============================================================
class TestCombiningAndSplitting:
    def test_concatenate(self):
        a = torch.zeros(2, 3)
        b = torch.ones(2, 3)
        c = mod.concatenate_tensors([a, b], dim=0)
        assert c.shape == (4, 3)

    def test_concatenate_dim1(self):
        a = torch.zeros(2, 3)
        b = torch.ones(2, 2)
        c = mod.concatenate_tensors([a, b], dim=1)
        assert c.shape == (2, 5)

    def test_stack(self):
        a = torch.zeros(3)
        b = torch.ones(3)
        s = mod.stack_tensors([a, b], dim=0)
        assert s.shape == (2, 3)

    def test_split(self):
        t = torch.arange(10)
        parts = mod.split_tensor(t, 3, dim=0)
        assert len(parts) == 4  # sizes 3,3,3,1
        assert torch.equal(torch.cat(parts), t)

    def test_chunk(self):
        t = torch.arange(9)
        parts = mod.chunk_tensor(t, 3, dim=0)
        assert len(parts) == 3
        for p in parts:
            assert p.shape == (3,)


# ============================================================
# Section 5: Math Operations
# ============================================================
class TestMathOperations:
    def test_add_multiply(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        assert torch.equal(mod.elementwise_add(a, b), torch.tensor([5.0, 7.0, 9.0]))
        assert torch.equal(mod.elementwise_multiply(a, b), torch.tensor([4.0, 10.0, 18.0]))

    def test_matmul(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        result = mod.matrix_multiply(a, b)
        expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
        assert torch.allclose(result, expected)

    def test_power_sqrt(self):
        t = torch.tensor([1.0, 4.0, 9.0])
        assert torch.allclose(mod.compute_power(t, 2), torch.tensor([1.0, 16.0, 81.0]))
        assert torch.allclose(mod.compute_sqrt(t), torch.tensor([1.0, 2.0, 3.0]))

    def test_exp_log(self):
        t = torch.tensor([0.0, 1.0])
        assert torch.allclose(mod.compute_exp(t), torch.tensor([1.0, math.e]), atol=1e-4)
        t2 = torch.tensor([1.0, math.e])
        assert torch.allclose(mod.compute_log(t2), torch.tensor([0.0, 1.0]), atol=1e-4)

    def test_clamp(self):
        t = torch.tensor([-5.0, 0.0, 5.0, 10.0])
        c = mod.clamp_tensor(t, min_val=0.0, max_val=8.0)
        assert torch.equal(c, torch.tensor([0.0, 0.0, 5.0, 8.0]))


# ============================================================
# Section 6: Reductions
# ============================================================
class TestReductions:
    @pytest.fixture
    def sample(self):
        return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_sum(self, sample):
        assert torch.isclose(mod.sum_tensor(sample), torch.tensor(21.0))
        assert torch.allclose(mod.sum_tensor(sample, dim=0), torch.tensor([5.0, 7.0, 9.0]))
        assert torch.allclose(mod.sum_tensor(sample, dim=1), torch.tensor([6.0, 15.0]))

    def test_mean(self, sample):
        assert torch.isclose(mod.mean_tensor(sample), torch.tensor(3.5))
        assert torch.allclose(mod.mean_tensor(sample, dim=1), torch.tensor([2.0, 5.0]))

    def test_max(self, sample):
        assert torch.isclose(mod.max_tensor(sample), torch.tensor(6.0))
        assert torch.allclose(mod.max_tensor(sample, dim=1), torch.tensor([3.0, 6.0]))

    def test_argmax(self, sample):
        assert mod.argmax_tensor(sample) == 5
        assert torch.equal(mod.argmax_tensor(sample, dim=1), torch.tensor([2, 2]))


# ============================================================
# Section 7: Broadcasting & Comparison
# ============================================================
class TestBroadcastingAndComparison:
    def test_broadcast_add(self):
        a = torch.tensor([[1.0], [2.0], [3.0]])  # (3,1)
        b = torch.tensor([10.0, 20.0, 30.0, 40.0])  # (4,)
        result = mod.broadcast_add(a, b)
        assert result.shape == (3, 4)
        assert torch.allclose(result[0], torch.tensor([11.0, 21.0, 31.0, 41.0]))

    def test_compare_greater(self):
        a = torch.tensor([1, 5, 3])
        b = torch.tensor([2, 2, 3])
        result = mod.compare_greater(a, b)
        assert result.dtype == torch.bool
        assert torch.equal(result, torch.tensor([False, True, False]))

    def test_masked_select_positive(self):
        t = torch.tensor([-2.0, 3.0, -1.0, 5.0, 0.0])
        result = mod.masked_select_positive(t)
        assert torch.equal(result, torch.tensor([3.0, 5.0]))


# ============================================================
# Section 8: In-place Operations
# ============================================================
class TestInPlace:
    def test_inplace_add(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        original_ptr = t.data_ptr()
        result = mod.inplace_add(t, 5.0)
        assert torch.equal(result, torch.tensor([6.0, 7.0, 8.0]))
        assert result.data_ptr() == original_ptr, "Should modify the tensor in place, not create a new one"
        assert torch.equal(t, torch.tensor([6.0, 7.0, 8.0])), "Original tensor should also be modified"


# ============================================================
# Section 9: Autograd
# ============================================================
class TestAutograd:
    def test_compute_gradient(self):
        # y = x^2 + 3x  =>  dy/dx = 2x + 3
        grad = mod.compute_gradient(2.0)
        assert torch.isclose(grad, torch.tensor(7.0))  # 2*2 + 3 = 7

        grad2 = mod.compute_gradient(0.0)
        assert torch.isclose(grad2, torch.tensor(3.0))

    def test_linear_regression_step(self):
        torch.manual_seed(0)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_true = torch.tensor([3.0, 5.0, 7.0, 9.0])  # y = 2x + 1
        w = torch.tensor(0.0, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)

        # Manually compute expected first step
        y_pred = w.detach() * x + b.detach()
        loss_grad_w = torch.mean(2 * (y_pred - y_true) * x)
        loss_grad_b = torch.mean(2 * (y_pred - y_true))
        expected_w = 0.0 - 0.01 * loss_grad_w
        expected_b = 0.0 - 0.01 * loss_grad_b

        new_w, new_b = mod.linear_regression_step(x, w, b, y_true, lr=0.01)

        assert torch.isclose(new_w.detach(), expected_w, atol=1e-4)
        assert torch.isclose(new_b.detach(), expected_b, atol=1e-4)
        assert new_w.requires_grad and new_b.requires_grad
        assert new_w.grad is None and new_b.grad is None, "Gradients should be cleared after the step"

    def test_linear_regression_multiple_steps_no_grad_accumulation(self):
        x = torch.tensor([1.0, 2.0])
        y_true = torch.tensor([2.0, 4.0])
        w = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)

        w, b = mod.linear_regression_step(x, w, b, y_true, lr=0.1)
        # Should not raise even when called again (tests that grads were zeroed/cleared)
        w, b = mod.linear_regression_step(x, w, b, y_true, lr=0.1)
        assert w.grad is None and b.grad is None


# ============================================================
# Section 10: Device Management
# ============================================================
class TestDeviceManagement:
    def test_get_device(self):
        device = mod.get_device()
        assert isinstance(device, torch.device)
        expected = "cuda" if HAS_CUDA else "cpu"
        assert device.type == expected

    def test_move_to_device_cpu(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        moved = mod.move_to_device(t, torch.device("cpu"))
        assert moved.device.type == "cpu"
        assert torch.equal(moved, t)

    def test_create_tensor_on_device_cpu(self):
        t = mod.create_tensor_on_device((2, 2), torch.device("cpu"))
        assert t.device.type == "cpu"
        assert t.shape == (2, 2)
        assert torch.equal(t, torch.zeros(2, 2))

    @pytest.mark.skipif(not HAS_CUDA, reason="No CUDA GPU available in this environment")
    def test_move_to_device_cuda(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        moved = mod.move_to_device(t, torch.device("cuda"))
        assert moved.device.type == "cuda"
        assert torch.equal(moved.cpu(), t)

    @pytest.mark.skipif(not HAS_CUDA, reason="No CUDA GPU available in this environment")
    def test_create_tensor_on_device_cuda(self):
        t = mod.create_tensor_on_device((3,), torch.device("cuda"))
        assert t.device.type == "cuda"


# ============================================================
# Section 11: Reproducibility
# ============================================================
class TestReproducibility:
    def test_set_seed_makes_rand_reproducible(self):
        mod.set_seed(123)
        a = torch.rand(5)
        mod.set_seed(123)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_set_seed_changes_output(self):
        mod.set_seed(1)
        a = torch.rand(5)
        mod.set_seed(2)
        b = torch.rand(5)
        assert not torch.equal(a, b)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
