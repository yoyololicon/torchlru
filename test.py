import pytest
import numpy as np
import torch
from numba import cuda

from torchlpc import sample_wise_lpc
from parallel_scan import compute_linear_recurrence


@pytest.mark.parametrize("n_dims", [16, 64])
@pytest.mark.parametrize("n_steps", [4096, 2**14, 2**16])
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
    pred = compute_linear_recurrence(
        cuda.to_device(decays),
        cuda.to_device(impulses),
        cuda.to_device(init_states),
        out,
        n_dims,
        n_steps,
    )

    kwargs = {"atol": 1e-6} if dtype in (np.float32, np.complex64) else {}

    np.testing.assert_allclose(target, out.copy_to_host(), **kwargs)
