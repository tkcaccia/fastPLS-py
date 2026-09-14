"""Public Python API for fastPLS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _core


def _matrix(value: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("input must be a vector or two-dimensional matrix")
    if array.dtype not in (np.dtype("float32"), np.dtype("float64")):
        array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError("input contains non-finite values")
    return np.asfortranarray(array)


@dataclass
class PLS:
    """Partial least-squares estimator backed by the fastPLS C++ core."""

    n_components: int = 2
    method: str = "simpls"
    classifier: str | None = None
    scaling: str = "autoscaling"
    backend: str = "cpu"
    oversample: int = 32
    power: int = 5
    seed: int = 1
    orthogonal_components: int = 1
    kernel: str = "linear"
    gamma: float = 1.0
    degree: int = 2
    offset: float = 1.0
    store_scores: bool = False

    def fit(self, X: Any, y: Any) -> "PLS":
        if self.backend != "cpu":
            raise ValueError(
                "fastPLS-py 0.1 provides only the portable CPU backend; "
                "CUDA and Metal are not silently replaced by CPU"
            )
        X_array = _matrix(X)
        y_array = np.asarray(y)
        self.classes_ = None
        classification = self.classifier is not None
        if classification:
            if y_array.ndim != 1:
                y_array = y_array.reshape(-1)
            self.classes_, encoded = np.unique(y_array, return_inverse=True)
            native_y = np.asarray(encoded, dtype=np.int64)
        else:
            native_y = _matrix(y_array, dtype=X_array.dtype)
        if X_array.shape[0] != native_y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        self._native = _core.fit(
            X_array,
            native_y,
            int(self.n_components),
            self.method,
            self.classifier or "regression",
            self.scaling,
            int(self.oversample),
            int(self.power),
            int(self.seed),
            int(self.orthogonal_components),
            self.kernel,
            float(self.gamma),
            int(self.degree),
            float(self.offset),
            bool(self.store_scores),
        )
        self.n_features_in_ = X_array.shape[1]
        self.n_components_ = self._native.n_components
        self.scores_ = self._native.scores if self.store_scores else None
        self.dtype_ = X_array.dtype
        return self

    def predict(self, X: Any, *, top: int | None = None) -> np.ndarray:
        if not hasattr(self, "_native"):
            raise RuntimeError("fit must be called before predict")
        X_array = _matrix(X, dtype=self.dtype_)
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of predictors")
        if self.classes_ is None:
            if top is not None:
                raise ValueError("top is available only for classification")
            return self._native.predict(X_array)
        ranks = 1 if top is None else int(top)
        if ranks < 1 or ranks > len(self.classes_):
            raise ValueError("top must be between 1 and the number of classes")
        encoded = self._native.predict_classes(X_array, ranks)
        decoded = self.classes_[encoded]
        return decoded[:, 0] if ranks == 1 else decoded

    def predict_scores(self, X: Any) -> np.ndarray:
        if not hasattr(self, "_native"):
            raise RuntimeError("fit must be called before predict_scores")
        return self._native.predict_scores(_matrix(X, dtype=self.dtype_))


def pls(Xtrain: Any, Ytrain: Any, Xtest: Any | None = None, **kwargs: Any):
    """Fit a PLS model and optionally return held-out predictions."""
    model = PLS(**kwargs).fit(Xtrain, Ytrain)
    if Xtest is None:
        return model
    return {"model": model, "prediction": model.predict(Xtest)}


def fastsvd(
    x: Any,
    n_components: int,
    *,
    oversample: int = 32,
    power: int = 5,
    seed: int = 1,
) -> dict[str, np.ndarray]:
    """Compute a randomized truncated singular-value decomposition."""
    return _core.fastsvd(
        _matrix(x), int(n_components), int(oversample), int(power), int(seed)
    )


def fastcor(x: Any) -> np.ndarray:
    """Compute a Pearson correlation matrix."""
    matrix = _matrix(x)
    centered = matrix - matrix.mean(axis=0)
    scale = np.sqrt(np.sum(centered.astype(np.float64) ** 2, axis=0))
    if np.any(scale == 0):
        raise ValueError("correlation is undefined for constant columns")
    return (centered.T @ centered) / np.outer(scale, scale)


def has_cuda() -> bool:
    """Return whether this build contains the CUDA runtime adapter."""
    return False


def has_metal() -> bool:
    """Return whether this build contains the Metal runtime adapter."""
    return False


def evaluate(observed: Any, predicted: Any) -> dict[str, Any]:
    """Evaluate classification labels or continuous predictions."""
    observed_array = np.asarray(observed)
    predicted_array = np.asarray(predicted)
    label_dtype = observed_array.dtype.kind in "OUSbiu"
    if label_dtype and observed_array.ndim == 1 and (
        predicted_array.ndim == 1 or
        (predicted_array.ndim == 2 and predicted_array.shape[1] > 1)
    ):
        top = predicted_array.reshape(-1, 1) if predicted_array.ndim == 1 else predicted_array
        if top.shape[0] != observed_array.shape[0]:
            raise ValueError("observed and predicted lengths differ")
        correct = top == observed_array[:, None]
        labels = np.unique(np.concatenate((observed_array, top[:, 0])))
        recalls = []
        for label in labels:
            mask = observed_array == label
            if mask.any():
                recalls.append(np.mean(top[mask, 0] == label))
        return {
            "accuracy": float(np.mean(correct[:, 0])),
            "balanced_accuracy": float(np.mean(recalls)),
            "top_accuracy": float(np.mean(np.any(correct, axis=1))),
            "top": int(top.shape[1]),
        }
    observed_matrix = _matrix(observed_array)
    predicted_matrix = _matrix(predicted_array, dtype=observed_matrix.dtype)
    if observed_matrix.shape != predicted_matrix.shape:
        raise ValueError("observed and predicted dimensions differ")
    residual = observed_matrix - predicted_matrix
    press = float(np.sum(residual.astype(np.float64) ** 2))
    centered = observed_matrix - observed_matrix.mean(axis=0)
    total = float(np.sum(centered.astype(np.float64) ** 2))
    return {
        "R2": float(1.0 - press / total) if total > 0 else np.nan,
        "RMSD": float(np.sqrt(np.mean(residual.astype(np.float64) ** 2))),
        "MAE": float(np.mean(np.abs(residual.astype(np.float64)))),
    }
