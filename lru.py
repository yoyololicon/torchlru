import torch
from torch.nn import Module
from torchlpc import sample_wise_lpc
from typing import Optional

from .recurrence import RecurrenceCUDA
from .parallel_scan import WARPSIZE


def linear_recurrence(
    u: torch.Tensor, a: torch.Tensor, zi: torch.Tensor
) -> torch.Tensor:
    if u.is_cuda and (u.size(0) * WARPSIZE < u.size(1)):
        return RecurrenceCUDA.apply(a, u, zi)
    return sample_wise_lpc(u, -a.unsqueeze(-1), zi.unsqueeze(-1))


def init_lambda(module, r_min=0.1, r_max=1.0, max_phase=torch.pi):
    if isinstance(module, LRU):
        u1, u2 = torch.rand(module.hidden_size * 2).chunk(2)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        theta_log = torch.log(max_phase * u2)

        # diag_lambda = torch.exp(-torch.exp(nu_log) + 1j * torch.exp(theta_log))
        # gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))

        module.weight_nu_log.data.copy_(nu_log)
        module.weight_theta_log.data.copy_(theta_log)
        # module.weight_gamma_log.data.copy_(gamma_log)


class LRU(Module):
    input_size: int
    hidden_size: int

    def __init__(self, input_size, hidden_size):
        super(LRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_nu_log = torch.nn.Parameter(torch.empty(hidden_size))
        self.weight_theta_log = torch.nn.Parameter(torch.empty(hidden_size))
        # self.weight_gamma_log = torch.nn.Parameter(torch.empty(hidden_size))

        self.weight_B = torch.nn.Parameter(
            torch.empty(input_size, hidden_size, dtype=torch.complex64)
        )
        self.weight_C = torch.nn.Parameter(
            torch.empty(hidden_size * 2, input_size, dtype=torch.float32)
        )
        self.weight_D = torch.nn.Parameter(torch.empty(input_size, dtype=torch.float32))

        self.reset_parameters()

    def extra_repr(self):
        return f"input_size={self.input_size}, hidden_size={self.hidden_size}"

    def reset_parameters(self):
        self.weight_B.data.normal_(0, 1 / (2 * self.input_size) ** 0.5)
        self.weight_C.data.normal_(0, 1 / self.hidden_size**0.5)
        self.weight_D.data.normal_(0, 1)

        self.apply(init_lambda)

    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None):
        assert x.dim() == 3

        if hx is None:
            hx = x.new_zeros(x.size(0), self.hidden_size, dtype=torch.complex64)

        a = torch.exp(-self.weight_nu_log.exp() + 1j * self.weight_theta_log.exp())
        b = torch.sqrt(1 - torch.abs(a) ** 2)

        B_norm = self.weight_B * b
        C = self.weight_C
        D = self.weight_D

        u = torch.view_as_complex(
            (x @ torch.view_as_real(B_norm).flatten(-2, -1)).view(
                x.shape[0], x.shape[1], -1, 2
            )
        )

        a_lpc = a.broadcast_to(u.shape).transpose(1, 2)
        u_lpc = u.transpose(1, 2)
        zi = hx

        filtered_u = (
            linear_recurrence(u_lpc.flatten(0, 1), a_lpc.flatten(0, 1), zi.flatten())
            .view_as(u_lpc)
            .transpose(1, 2)
        )

        return (
            torch.view_as_real(filtered_u).flatten(-2, -1) @ C + x * D,
            filtered_u[:, -1, :],
        )
