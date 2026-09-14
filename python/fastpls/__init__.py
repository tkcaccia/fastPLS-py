"""Python interface to the shared fastPLS C++ core."""

from .api import PLS, evaluate, fastcor, fastsvd, has_cuda, has_metal, pls
from ._core import backend_info

__all__ = [
    "PLS",
    "backend_info",
    "evaluate",
    "fastcor",
    "fastsvd",
    "has_cuda",
    "has_metal",
    "pls",
]
__version__ = "0.1.0"
