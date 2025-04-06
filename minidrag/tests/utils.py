from tests.kernels.allclose_default import get_default_atol, get_default_rtol

default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float: 1e-5}
default_rtol = {
    torch.float16: 1e-3,
    torch.bfloat16: 1.6e-2,
    torch.float: 1.3e-6
}


def get_default_atol(output) -> float:
    return default_atol[output.dtype]


def get_default_rtol(output) -> float:
    return default_rtol[output.dtype]