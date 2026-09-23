"""Python interface to the shared fastPLS C++ core."""

from .api import (
    PLS,
    cuda_info,
    evaluate,
    fastcor,
    fastsvd,
    has_cuda,
    has_metal,
    plot_permutation,
    pls,
    pls_double_cv,
    pls_single_cv,
    vip,
)
from ._core import backend_info

__all__ = [
    "PLS",
    "backend_info",
    "cuda_info",
    "evaluate",
    "fastcor",
    "fastsvd",
    "has_cuda",
    "has_metal",
    "plot_permutation",
    "pls",
    "pls_double_cv",
    "pls_single_cv",
    "vip",
]
__version__ = "0.3.0"
