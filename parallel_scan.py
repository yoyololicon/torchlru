import numba
from numba import cuda


@cuda.jit(device=True)
def divide_work(n_jobs, n_workers, worker_idx) -> tuple:
    cd = (n_jobs + n_workers - 1) // n_workers
    d, doing_cd = divmod(n_jobs, n_workers)
    if worker_idx < doing_cd:
        x = cd * worker_idx
        y = x + cd
    else:
        x = cd * doing_cd + d * (worker_idx - doing_cd)
        y = x + d
    return x, y


@cuda.jit(device=True)
def compute_warp_start_stop(blockIdx, warp_idx, n_blocks, n_steps):
    block_start, block_stop = divide_work(n_steps, n_blocks, blockIdx)
    block_jobs = block_stop - block_start

    warp_start, warp_stop = divide_work(block_jobs, cuda.warpsize, warp_idx)
    warp_start += block_start
    warp_stop += block_start

    return warp_start, warp_stop


@cuda.jit
def reduction_kernel(
    decay, impulses, initial_state, _decay_storage, _h_storage, n_dims, n_steps
):
    warp = cuda.threadIdx.x // cuda.warpsize
    lane = cuda.threadIdx.x % cuda.warpsize

    decay_storage = _decay_storage[cuda.blockIdx.x * (cuda.warpsize + 1) :]
    h_storage = _h_storage[cuda.blockIdx.x * (cuda.warpsize + 1) :]

    warp_start, warp_stop = compute_warp_start_stop(
        cuda.blockIdx.x, warp, cuda.gridDim.x, n_steps
    )

    # reduce within warp
    for i in range(lane, n_dims, cuda.warpsize):
        cum_decay = 1.0
        h = 0.0
        if (cuda.blockIdx.x == 0) and (warp == 0):
            h = initial_state[i]

        for t in range(warp_start, warp_stop):
            cum_decay *= decay[i, t]
            h = decay[i, t] * h + impulses[i, t]

        decay_storage[warp, i] = cum_decay
        h_storage[warp, i] = h

    cuda.syncthreads()

    # reduce within block
    for i in range(lane + warp * cuda.warpsize, n_dims, cuda.blockDim.x):
        cum_decay = 1.0
        h = 0.0
        for t in range(cuda.warpsize):
            cum_decay *= decay_storage[t, i]
            h = decay_storage[t, i] * h + h_storage[t, i]

        decay_storage[cuda.warpsize, i] = cum_decay
        h_storage[cuda.warpsize, i] = h


@cuda.jit
def block_scan_kernel(decay_storage, h_storage, n_dims, n_blocks):
    for i in range(
        cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x,
        n_dims,
        cuda.blockDim.x * cuda.gridDim.x,
    ):
        for t in range(1, n_blocks):
            cur_idx = t * (cuda.warpsize + 1) + cuda.warpsize
            prev_idx = (t - 1) * (cuda.warpsize + 1) + cuda.warpsize
            h_storage[cur_idx, i] = (
                h_storage[prev_idx, i] * decay_storage[cur_idx, i]
                + h_storage[cur_idx, i]
            )
            decay_storage[cur_idx, i] *= decay_storage[prev_idx, i]


@cuda.jit
def warp_scan_kernel(
    decay, impulses, initial_state, out, decay_storage, h_storage, n_dims, n_steps
):
    warp, lane = divmod(cuda.threadIdx.x, cuda.warpsize)

    for i in range(lane + warp * cuda.warpsize, n_dims, cuda.blockDim.x):
        for t in range(cuda.warpsize):
            if t == 0 and cuda.blockIdx.x == 0:
                continue
            cur_idx = t + cuda.blockIdx.x * (cuda.warpsize + 1)
            prev_idx = cur_idx - 1
            h_storage[cur_idx, i] = (
                h_storage[prev_idx, i] * decay[i, cur_idx] + h_storage[cur_idx, i]
            )
            decay_storage[cur_idx, i] *= decay_storage[prev_idx, i]

    cuda.syncthreads()

    warp_start, warp_stop = compute_warp_start_stop(
        cuda.blockIdx.x, warp, cuda.gridDim.x, n_steps
    )

    # scan within warp
    for i in range(lane, n_dims, cuda.warpsize):
        h = 0.0
        if (cuda.blockIdx.x == 0) and (warp == 0):
            h = initial_state[i]
        else:
            h = h_storage[warp - 1 + cuda.blockIdx.x * (cuda.warpsize + 1), i]

        for t in range(warp_start, warp_stop):
            h = decay[i, t] * h + impulses[i, t]
            out[i, t] = h


def compute_linear_recurrence(
    decays, impulses, init_states, out, n_dims: int, n_steps: int, n_SMs: int = 15
):
    n_blocks = min((n_steps + cuda.warpsize - 1) // cuda.warpsize, n_SMs * 2)

    reduction_mem_shape = (n_blocks * (cuda.warpsize + 1), n_dims)
    decay_storage = cuda.device_array(reduction_mem_shape, dtype=decays.dtype)
    h_storage = cuda.device_array(reduction_mem_shape, dtype=impulses.dtype)

    reduction_kernel[n_blocks, 1024](
        decays, impulses, init_states, decay_storage, h_storage, n_dims, n_steps
    )

    block_scan_kernel[n_blocks, 1024](decay_storage, h_storage, n_dims, n_blocks)

    warp_scan_kernel[n_blocks, 1024](
        decays, impulses, init_states, out, decay_storage, h_storage, n_dims, n_steps
    )

