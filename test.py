import pytest
import numpy as np
import torch
from torch.autograd import gradcheck, gradgradcheck
from numba import cuda

from torchlpc import sample_wise_lpc

from .parallel_scan import compute_linear_recurrence
from .recurrence import Recurrence


@pytest.mark.parametrize("decay_requires_grad", [True])
@pytest.mark.parametrize("impulse_requires_grad", [True, False])
@pytest.mark.parametrize("initial_state_requires_grad", [True, False])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_grad(
    decay_requires_grad, impulse_requires_grad, initial_state_requires_grad, dtype
):
    decay = torch.rand(
        4, 50, requires_grad=decay_requires_grad, dtype=dtype, device="cuda"
    )
    impulse = torch.randn(
        4, 50, requires_grad=impulse_requires_grad, dtype=dtype, device="cuda"
    )
    initial_state = torch.randn(
        4, requires_grad=initial_state_requires_grad, dtype=dtype, device="cuda"
    )

    assert gradcheck(Recurrence.apply, (decay, impulse, initial_state))
    assert gradgradcheck(Recurrence.apply, (decay, impulse, initial_state))


@pytest.mark.parametrize("n_dims", [16, 64])
@pytest.mark.parametrize("n_steps", [4096, 2**14 - 1, 2**16 - 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_equivalent(n_dims, n_steps, dtype):
    decays = np.random.rand(n_dims, n_steps).astype(dtype) * 2 - 1
    impulses = np.random.randn(n_dims, n_steps).astype(dtype)
    init_states = np.random.randn(n_dims).astype(dtype)

    target = (
        sample_wise_lpc(
            torch.from_numpy(impulses).cuda(),
            -torch.from_numpy(decays).unsqueeze(-1).cuda(),
            torch.from_numpy(init_states).unsqueeze(-1).cuda(),
        )
        .cpu()
        .numpy()
    )

    out = cuda.device_array((n_dims, n_steps), dtype=dtype)
    compute_linear_recurrence(
        cuda.to_device(decays),
        cuda.to_device(impulses),
        cuda.to_device(init_states),
        out,
        n_dims,
        n_steps,
    )

    kwargs = {"atol": 1e-6} if dtype in (np.float32, np.complex64) else {}

    np.testing.assert_allclose(target, out.copy_to_host(), **kwargs)
