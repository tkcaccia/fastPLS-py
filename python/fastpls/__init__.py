"""Python interface to the shared fastPLS C++ core."""

from .api import (
    PLS,
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
__version__ = "0.2.0"
