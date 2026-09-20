"""Public Python API for fastPLS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

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
    scaling: str = "centering"
    backend: str = "cpu"
    oversample: int = 32
    power: int = 5
    seed: int = 1
    orthogonal_components: int = 1
    kernel: str = "linear"
    gamma: float | None = None
    degree: int = 3
    offset: float = 1.0
    store_scores: bool = False

    def fit(self, X: Any, y: Any) -> "PLS":
        backend = str(self.backend).lower()
        if backend != "cpu":
            raise ValueError(
                "fastPLS-py 0.2 provides only the portable CPU backend; "
                "CUDA and Metal are not silently replaced by CPU"
            )
        X_array = _matrix(X)
        method = _choice(
            self.method, "method", {"simpls", "plssvd", "opls", "kernelpls"}
        )
        classifier = None if self.classifier is None else _choice(
            self.classifier, "classifier", {"argmax", "lda"}
        )
        scaling = _choice(
            self.scaling, "scaling", {"none", "centering", "autoscaling"}
        )
        kernel = _choice(
            self.kernel, "kernel", {"linear", "rbf", "polynomial"}
        )
        components = _positive_integer(self.n_components, "n_components")
        oversample = _positive_integer(self.oversample, "oversample")
        power = _nonnegative_integer(self.power, "power")
        seed = _nonnegative_integer(self.seed, "seed")
        orthogonal_components = _nonnegative_integer(
            self.orthogonal_components, "orthogonal_components"
        )
        degree = _positive_integer(self.degree, "degree")
        gamma = 1.0 / X_array.shape[1] if self.gamma is None else float(self.gamma)
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be finite and positive")
        y_array = np.asarray(y)
        self.classes_ = None
        classification = classifier is not None
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
            components,
            method,
            classifier or "regression",
            scaling,
            oversample,
            power,
            seed,
            orthogonal_components,
            kernel,
            gamma,
            degree,
            float(self.offset),
            bool(self.store_scores),
        )
        self.n_features_in_ = X_array.shape[1]
        self.requested_n_components_ = components
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
        ranks = 1 if top is None else _positive_integer(top, "top")
        if ranks < 1 or ranks > len(self.classes_):
            raise ValueError("top must be between 1 and the number of classes")
        encoded = self._native.predict_classes(X_array, ranks)
        decoded = self.classes_[encoded]
        return decoded[:, 0] if ranks == 1 else decoded

    def predict_scores(self, X: Any) -> np.ndarray:
        if not hasattr(self, "_native"):
            raise RuntimeError("fit must be called before predict_scores")
        return self._native.predict_scores(_matrix(X, dtype=self.dtype_))

    def predict_responses(self, X: Any) -> np.ndarray:
        """Return continuous responses or class-indicator scores."""
        if not hasattr(self, "_native"):
            raise RuntimeError("fit must be called before predict_responses")
        return self._native.predict(_matrix(X, dtype=self.dtype_))

    def vip(self) -> np.ndarray | list[np.ndarray]:
        """Return variable-importance paths for a score-retaining model."""
        if not hasattr(self, "_native"):
            raise RuntimeError("fit must be called before vip")
        return self._native.vip()


def pls(
    Xtrain: Any,
    Ytrain: Any,
    Xtest: Any | None = None,
    Ytest: Any | None = None,
    *,
    top: int | None = None,
    bycol: bool = False,
    **kwargs: Any,
):
    """Fit a PLS model and optionally return held-out predictions."""
    model = PLS(**kwargs).fit(Xtrain, Ytrain)
    if Xtest is None:
        if Ytest is not None:
            raise ValueError("Ytest requires Xtest")
        return model
    prediction = model.predict(Xtest, top=top)
    output = {"model": model, "prediction": prediction}
    if Ytest is not None:
        if model.classes_ is None:
            output["metrics"] = evaluate(
                Ytest, prediction, ytrain=Ytrain, bycol=bycol,
            )
        else:
            output["metrics"] = _evaluate_classification(Ytest, prediction)
    return output


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
        _matrix(x),
        _positive_integer(n_components, "n_components"),
        _positive_integer(oversample, "oversample"),
        _nonnegative_integer(power, "power"),
        _nonnegative_integer(seed, "seed"),
    )


def fastcor(
    a: Any,
    b: Any | None = None,
    *,
    byrow: bool = True,
    diag: bool = True,
) -> np.ndarray:
    """Compute Pearson correlations between rows or columns."""
    left = _matrix(a)
    right = left if b is None else _matrix(b, dtype=left.dtype)
    if byrow:
        if left.shape[1] != right.shape[1]:
            raise ValueError("a and b must have the same number of columns")
        left_centered = left - left.mean(axis=1, keepdims=True)
        right_centered = right - right.mean(axis=1, keepdims=True)
        left_scale = np.linalg.norm(left_centered.astype(np.float64), axis=1)
        right_scale = np.linalg.norm(right_centered.astype(np.float64), axis=1)
        product = left_centered @ right_centered.T
    else:
        if left.shape[0] != right.shape[0]:
            raise ValueError("a and b must have the same number of rows")
        left_centered = left - left.mean(axis=0, keepdims=True)
        right_centered = right - right.mean(axis=0, keepdims=True)
        left_scale = np.linalg.norm(left_centered.astype(np.float64), axis=0)
        right_scale = np.linalg.norm(right_centered.astype(np.float64), axis=0)
        product = left_centered.T @ right_centered
    if np.any(left_scale == 0) or np.any(right_scale == 0):
        raise ValueError("correlation is undefined for constant rows or columns")
    result = product / np.outer(left_scale, right_scale)
    if b is not None and diag:
        if result.shape[0] != result.shape[1]:
            raise ValueError("diag=True requires matching rows or columns")
        return np.diag(result)
    return result


def has_cuda() -> bool:
    """Return whether this build contains the CUDA runtime adapter."""
    return False


def has_metal() -> bool:
    """Return whether this build contains the Metal runtime adapter."""
    return False


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64, copy=False)
    right = right.astype(np.float64, copy=False)
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0 else np.nan


def evaluate(
    observed: Any,
    predicted: Any,
    *,
    ytrain: Any | None = None,
    bycol: bool = True,
    relative_epsilon: float = np.finfo(np.float64).eps,
) -> dict[str, Any]:
    """Evaluate labels, ranked labels, or continuous predictions."""
    observed_array = np.asarray(observed)
    predicted_array = np.asarray(predicted)
    label_dtype = observed_array.dtype.kind in "OUSb"
    if label_dtype and observed_array.ndim == 1 and (
        predicted_array.ndim == 1 or
        (predicted_array.ndim == 2 and predicted_array.shape[1] > 1)
    ):
        top = predicted_array.reshape(-1, 1) if predicted_array.ndim == 1 else predicted_array
        if top.shape[0] != observed_array.shape[0]:
            raise ValueError("observed and predicted lengths differ")
        correct = top == observed_array[:, None]
        labels = np.unique(np.concatenate((observed_array, top[:, 0])))
        confusion = np.zeros((labels.size, labels.size), dtype=np.int64)
        index = {value: position for position, value in enumerate(labels.tolist())}
        for truth, estimate in zip(observed_array, top[:, 0], strict=True):
            confusion[index[truth], index[estimate]] += 1
        recalls = []
        precisions = []
        f1_values = []
        per_class = []
        for label in labels:
            mask = observed_array == label
            predicted_mask = top[:, 0] == label
            true_positive = int(np.sum(mask & predicted_mask))
            recall = true_positive / int(np.sum(mask)) if np.any(mask) else np.nan
            precision = (
                true_positive / int(np.sum(predicted_mask))
                if np.any(predicted_mask) else 0.0
            )
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0 else 0.0
            )
            recalls.append(recall)
            precisions.append(precision)
            f1_values.append(f1)
            per_class.append(
                {"class": label, "precision": precision, "recall": recall, "F1": f1}
            )
        accuracy = float(np.mean(correct[:, 0]))
        no_information = float(np.max(np.bincount(
            np.unique(observed_array, return_inverse=True)[1]
        )) / observed_array.size)
        row_margins = confusion.sum(axis=1)
        column_margins = confusion.sum(axis=0)
        chance = float(np.dot(row_margins, column_margins) / observed_array.size ** 2)
        kappa = (accuracy - chance) / (1.0 - chance) if chance < 1 else np.nan
        metrics = {
            "accuracy": accuracy,
            "no_information_rate": no_information,
            "lift_accuracy": accuracy / no_information if no_information > 0 else np.nan,
            "balanced_accuracy": float(np.nanmean(recalls)),
            "macro_precision": float(np.mean(precisions)),
            "macro_recall": float(np.nanmean(recalls)),
            "macro_f1": float(np.mean(f1_values)),
            "kappa": float(kappa),
            "top_accuracy": float(np.mean(np.any(correct, axis=1))),
            "top": int(top.shape[1]),
        }
        return {
            **metrics,
            "task": "classification",
            "metrics": metrics,
            "classes": labels,
            "per_class": per_class,
            "confusion": confusion,
        }
    observed_matrix = _matrix(observed_array)
    predicted_matrix = _matrix(predicted_array, dtype=observed_matrix.dtype)
    if observed_matrix.shape != predicted_matrix.shape:
        raise ValueError("observed and predicted dimensions differ")
    residual = observed_matrix - predicted_matrix
    press = float(np.sum(residual.astype(np.float64) ** 2))
    observed64 = observed_matrix.astype(np.float64)
    predicted64 = predicted_matrix.astype(np.float64)
    residual64 = observed64 - predicted64
    centered = observed64 - observed64.mean(axis=0)
    total = float(np.sum(centered ** 2))
    rmsd = float(np.sqrt(np.mean(residual64 ** 2)))
    standard_deviation = float(np.sqrt(np.sum(centered ** 2) / max(observed64.size - 1, 1)))
    reference = np.abs(observed64) > relative_epsilon
    relative = np.abs(residual64[reference] / observed64[reference]) if np.any(reference) else np.array([])
    q2 = np.nan
    if ytrain is not None:
        training = _matrix(ytrain, dtype=observed_matrix.dtype).astype(np.float64)
        if training.shape[1] != observed64.shape[1]:
            raise ValueError("ytrain and observed must have the same response columns")
        q2_total = float(np.sum((observed64 - training.mean(axis=0)) ** 2))
        q2 = float(1.0 - press / q2_total) if q2_total > 0 else np.nan
    metrics = {
        "R2": float(1.0 - press / total) if total > 0 else np.nan,
        "Q2": q2,
        "RMSD": rmsd,
        "RMSE": rmsd,
        "MAE": float(np.mean(np.abs(residual64))),
        "bias": float(np.mean(predicted64 - observed64)),
        "median_relative_error_percent": (
            float(100.0 * np.median(relative)) if relative.size else np.nan
        ),
        "MAPE_percent": float(100.0 * np.mean(relative)) if relative.size else np.nan,
        "RPD": standard_deviation / rmsd if rmsd > 0 else np.inf,
        "Pearson_r": _correlation(observed64.ravel(), predicted64.ravel()),
        "Spearman_r": _correlation(_rank(observed64.ravel()), _rank(predicted64.ravel())),
    }
    output: dict[str, Any] = {**metrics, "task": "regression", "metrics": metrics}
    if bycol and observed64.shape[1] > 1:
        output["per_response"] = [
            evaluate(observed64[:, column], predicted64[:, column],
                     ytrain=None if ytrain is None else training[:, column], bycol=False)["metrics"]
            for column in range(observed64.shape[1])
        ]
    return output


def _evaluate_classification(observed: Any, predicted: Any) -> dict[str, Any]:
    """Evaluate labels after an explicit classification fit."""
    return evaluate(
        np.asarray(observed).astype(str),
        np.asarray(predicted).astype(str),
    )


_CLASSIFICATION_SELECTIONS = {
    "accuracy", "balanced_accuracy", "lift_accuracy", "macro_precision",
    "macro_recall", "macro_f1", "kappa", "r2y", "q2y",
}
_REGRESSION_SELECTIONS = {
    "r2y", "q2y", "rmsd", "mae", "mape_percent", "rpd",
    "pearson_r", "spearman_r",
}


def _choice(value: Any, name: str, allowed: set[str]) -> str:
    normalized = str(value).lower()
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {choices}")
    return normalized


def _positive_integer(value: Any, name: str) -> int:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a positive integer")
    numeric = float(array)
    if not np.isfinite(numeric) or numeric < 1 or numeric != np.floor(numeric):
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _nonnegative_integer(value: Any, name: str) -> int:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a non-negative integer")
    numeric = float(array)
    if not np.isfinite(numeric) or numeric < 0 or numeric != np.floor(numeric):
        raise ValueError(f"{name} must be a non-negative integer")
    return int(numeric)


def _component_vector(ncomp: int | Sequence[int]) -> np.ndarray:
    raw = np.asarray([ncomp] if np.isscalar(ncomp) else ncomp)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("ncomp must contain positive integers")
    numeric = raw.astype(np.float64)
    if (not np.isfinite(numeric).all() or np.any(numeric < 1) or
            np.any(numeric != np.floor(numeric))):
        raise ValueError("ncomp must contain positive integers")
    if np.any(numeric > np.iinfo(np.int32).max):
        raise ValueError("ncomp values are too large")
    values = numeric.astype(np.int32)
    return np.unique(values)


def _classification_target(y: Any, classifier: str | None) -> bool:
    values = np.asarray(y)
    return classifier is not None or values.dtype.kind in "OUSb"


def _make_folds(
    sample_count: int,
    kfold: int | str,
    *,
    groups: Any | None,
    labels: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    group_values = np.arange(sample_count) if groups is None else np.asarray(groups)
    if group_values.ndim != 1 or group_values.size != sample_count:
        raise ValueError("constrain must contain one group per sample")
    unique, first, inverse = np.unique(group_values, return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.empty(order.size, dtype=np.int64)
    remap[order] = np.arange(order.size)
    inverse = remap[inverse]
    group_count = unique.size
    if isinstance(kfold, str):
        if kfold.lower() != "loocv":
            raise ValueError("kfold must be an integer or 'loocv'")
        fold_count = group_count
    else:
        requested = _positive_integer(kfold, "kfold")
        fold_count = min(requested, group_count)
    if fold_count < 2:
        raise ValueError("cross-validation requires at least two folds")
    rng = np.random.default_rng(seed)
    group_folds = np.empty(group_count, dtype=np.int32)
    group_labels = None
    if labels is not None:
        label_values = np.asarray(labels)
        group_labels = np.empty(group_count, dtype=label_values.dtype)
        for group in range(group_count):
            member_labels = label_values[inverse == group]
            values, counts = np.unique(member_labels, return_counts=True)
            group_labels[group] = values[np.argmax(counts)]
    if fold_count == group_count:
        group_folds[:] = np.arange(1, group_count + 1)
    elif labels is None:
        shuffled = rng.permutation(group_count)
        group_folds[shuffled] = np.arange(group_count) % fold_count + 1
    else:
        for label in np.unique(group_labels):
            members = np.flatnonzero(group_labels == label)
            shuffled = rng.permutation(members)
            group_folds[shuffled] = np.arange(members.size) % fold_count + 1
    if np.unique(group_folds).size != fold_count:
        raise ValueError(
            "stratified grouped folds cannot populate every fold; "
            "reduce kfold or provide more groups per class"
        )
    folds = group_folds[inverse]
    if labels is not None:
        for label in np.unique(label_values):
            if np.unique(folds[label_values == label]).size < 2:
                raise ValueError(
                    "each class must occur in at least two folds"
                )
    return folds


def _validate_folds(
    folds: Any,
    sample_count: int,
    *,
    groups: Any | None,
    labels: np.ndarray | None,
) -> np.ndarray:
    raw = np.asarray(folds)
    if raw.ndim != 1 or raw.size != sample_count:
        raise ValueError("folds must contain one value per sample")
    numeric = raw.astype(np.float64)
    if (not np.isfinite(numeric).all() or np.any(numeric < 1) or
            np.any(numeric != np.floor(numeric))):
        raise ValueError("folds must contain positive integers")
    _, normalized = np.unique(numeric, return_inverse=True)
    normalized = normalized.astype(np.int32) + 1
    if np.unique(normalized).size < 2:
        raise ValueError("cross-validation requires at least two folds")
    if groups is not None:
        group_values = np.asarray(groups)
        if group_values.ndim != 1 or group_values.size != sample_count:
            raise ValueError("constrain must contain one group per sample")
        for group in np.unique(group_values):
            if np.unique(normalized[group_values == group]).size != 1:
                raise ValueError("each constrained group must remain in one fold")
    if labels is not None:
        label_values = np.asarray(labels)
        for label in np.unique(label_values):
            if np.unique(normalized[label_values == label]).size < 2:
                raise ValueError(
                    "each class must occur in at least two folds"
                )
    return normalized


def _selection_value(
    selection: str,
    evaluation: dict[str, Any],
    *,
    q2: float,
    r2: float,
) -> float:
    if selection == "q2y":
        return float(q2)
    if selection == "r2y":
        return float(r2)
    key = {
        "rmsd": "RMSD", "mae": "MAE", "mape_percent": "MAPE_percent",
        "rpd": "RPD", "pearson_r": "Pearson_r", "spearman_r": "Spearman_r",
    }.get(selection, selection)
    return float(evaluation["metrics"][key])


def pls_single_cv(
    Xdata: Any,
    Ydata: Any,
    ncomp: int | Sequence[int] = 2,
    *,
    constrain: Any | None = None,
    scaling: str = "centering",
    method: str = "simpls",
    backend: str = "cpu",
    seed: int = 1,
    kfold: int | str = 10,
    orthogonal_components: int = 1,
    kernel: str = "linear",
    gamma: float | None = None,
    degree: int = 3,
    offset: float = 1.0,
    classifier: str | None = None,
    fit: bool = True,
    bycol: bool = False,
    selection: str = "auto",
    oversample: int = 32,
    power: int = 5,
    folds: Any | None = None,
) -> dict[str, Any]:
    """Select a component count by compiled grouped cross-validation."""
    if backend != "cpu":
        raise ValueError("fastPLS-py currently provides only the CPU backend")
    X = _matrix(Xdata)
    y = np.asarray(Ydata)
    classification = _classification_target(y, classifier)
    if classification:
        if y.ndim != 1:
            y = y.reshape(-1)
        classes, encoded = np.unique(y, return_inverse=True)
        native_y: Any = np.asarray(encoded, dtype=np.int64)
        classifier = classifier or "argmax"
    else:
        classes = None
        native_y = _matrix(y, dtype=X.dtype)
        classifier = None
    if X.shape[0] != native_y.shape[0]:
        raise ValueError("Xdata and Ydata must contain the same number of samples")
    selection_name = selection.lower()
    if selection_name == "auto":
        selection_name = "accuracy" if classification else "rmsd"
    allowed = _CLASSIFICATION_SELECTIONS if classification else _REGRESSION_SELECTIONS
    if selection_name not in allowed:
        raise ValueError(f"selection '{selection}' is not valid for this task")
    components = _component_vector(ncomp)
    fold_values = _make_folds(
        X.shape[0], kfold, groups=constrain,
        labels=y if classification else None, seed=seed,
    ) if folds is None else _validate_folds(
        folds, X.shape[0], groups=constrain,
        labels=y if classification else None,
    )
    native = _core.cross_validate(
        X, native_y, components, fold_values, method,
        classifier or "regression", scaling, selection_name,
        oversample, power, seed, orthogonal_components, kernel,
        (1.0 / X.shape[1]) if gamma is None else gamma,
        degree, offset, True, selection_name == "q2y",
    )
    cross_validated = []
    predictions = []
    fitted = []
    selection_values = []
    r2_path = []
    for index, component in enumerate(components):
        if classification:
            prediction = classes[np.asarray(native["prediction_index"])[:, index] - 1]
            evaluation = _evaluate_classification(y, prediction)
            q2 = float(native["Q2Y"][index]) if len(native["Q2Y"]) else np.nan
        else:
            prediction = np.asarray(native["predictions"][index])
            evaluation = evaluate(native_y, prediction, bycol=bycol)
            q2 = float(native["Q2Y"][index])
        predictions.append(prediction)
        cross_validated.append(evaluation)
        if fit or selection_name == "r2y":
            model = PLS(
                n_components=int(component), method=method, classifier=classifier,
                scaling=scaling, backend=backend, oversample=oversample,
                power=power, seed=seed, orthogonal_components=orthogonal_components,
                kernel=kernel, gamma=(1.0 / X.shape[1]) if gamma is None else gamma,
                degree=degree, offset=offset,
            ).fit(X, y)
            if classification:
                raw = model._native.predict(X)
                target = np.eye(classes.size, dtype=X.dtype)[native_y]
                fitted_evaluation = evaluate(target, raw, bycol=bycol)
            else:
                raw = model.predict(X)
                fitted_evaluation = evaluate(native_y, raw, bycol=bycol)
            r2 = float(fitted_evaluation["metrics"]["R2"])
            fitted.append(fitted_evaluation)
        else:
            r2 = np.nan
        r2_path.append(r2)
        selection_values.append(_selection_value(
            selection_name, evaluation, q2=q2, r2=r2
        ))
    minimize = selection_name in {"rmsd", "mae", "mape_percent"}
    values = np.asarray(selection_values, dtype=np.float64)
    best_index = int(np.nanargmin(values) if minimize else np.nanargmax(values))
    return {
        "best_ncomp": int(components[best_index]),
        "best_index": best_index,
        "selection_metric": selection_name,
        "best_metric_name": selection_name,
        "best_metric_value": float(values[best_index]),
        "ncomp": components,
        "fold": fold_values,
        "pred": predictions,
        "Ypred": predictions,
        "accuracy": np.asarray(native.get("accuracy", []), dtype=float),
        "balanced_accuracy": np.asarray([
            value["metrics"].get("balanced_accuracy", np.nan)
            for value in cross_validated
        ]),
        "Q2Y": np.asarray(native["Q2Y"], dtype=float),
        "R2Y": np.asarray(r2_path, dtype=float),
        "RMSD": np.asarray(native.get("RMSD", []), dtype=float),
        "selection_metrics": values,
        "metrics": {"cross_validated": cross_validated, "fitted": fitted},
        "status": np.asarray(native["status"], dtype=int),
        "method": method,
        "backend": backend,
    }


def _permutation_indices(groups: Any | None, sample_count: int, rng: np.random.Generator) -> np.ndarray:
    if groups is None:
        return rng.permutation(sample_count)
    values = np.asarray(groups)
    unique, inverse = np.unique(values, return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(unique.size)]
    output = np.arange(sample_count)
    for size in sorted({member.size for member in members}):
        eligible = [member for member in members if member.size == size]
        donors = rng.permutation(len(eligible))
        for target, donor in zip(eligible, donors, strict=True):
            output[target] = eligible[int(donor)]
    return output


def pls_double_cv(
    Xdata: Any,
    Ydata: Any,
    ncomp: int | Sequence[int] = 2,
    *,
    constrain: Any | None = None,
    runn: int = 1,
    kfold_inner: int | str = 10,
    kfold_outer: int | str = 10,
    perm_test: bool = False,
    times: int = 100,
    seed: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run repeated nested cross-validation with fixed grouped folds."""
    runs_requested = _positive_integer(runn, "runn")
    permutations_requested = _positive_integer(times, "times") if perm_test else int(times)
    X = _matrix(Xdata)
    y = np.asarray(Ydata)
    classifier = kwargs.get("classifier")
    classification = _classification_target(y, classifier)
    groups = np.arange(X.shape[0]) if constrain is None else np.asarray(constrain)
    components = _component_vector(ncomp)
    plans = []
    for run in range(runs_requested):
        outer = _make_folds(
            X.shape[0], kfold_outer, groups=groups,
            labels=y if classification else None, seed=seed + 10000 * run,
        )
        inner = {}
        for fold in np.unique(outer):
            train = outer != fold
            inner[int(fold)] = _make_folds(
                int(np.sum(train)), kfold_inner, groups=groups[train],
                labels=y[train] if classification else None,
                seed=seed + 10000 * run + int(fold),
            )
        plans.append((outer, inner))

    def run_nested(response: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
        runs = []
        all_predictions = []
        for run, (outer, inner) in enumerate(plans):
            if classification:
                prediction = np.empty(response.shape[0], dtype=response.dtype)
                response_classes = np.unique(response)
                response_scores = np.zeros(
                    (response.shape[0], response_classes.size), dtype=X.dtype,
                    order="F",
                )
            else:
                response_matrix = _matrix(response, dtype=X.dtype)
                prediction = np.empty_like(response_matrix)
            selected = []
            q2_press = 0.0
            q2_reference = 0.0
            training_r2 = []
            for fold in np.unique(outer):
                train = outer != fold
                test = ~train
                inner_result = pls_single_cv(
                    X[train], response[train], components, constrain=groups[train],
                    seed=seed + 10000 * run + int(fold),
                    kfold=kfold_inner, folds=inner[int(fold)], fit=False, **kwargs,
                )
                selected_component = inner_result["best_ncomp"]
                selected.append(selected_component)
                model = PLS(
                    n_components=selected_component,
                    method=kwargs.get("method", "simpls"),
                    classifier=(kwargs.get("classifier") or "argmax") if classification else None,
                    scaling=kwargs.get("scaling", "centering"),
                    backend=kwargs.get("backend", "cpu"),
                    oversample=kwargs.get("oversample", 32),
                    power=kwargs.get("power", 5), seed=seed + 20000 * run + int(fold),
                    orthogonal_components=kwargs.get("orthogonal_components", 1),
                    kernel=kwargs.get("kernel", "linear"),
                    gamma=(1.0 / X.shape[1]
                           if kwargs.get("gamma") is None else kwargs["gamma"]),
                    degree=kwargs.get("degree", 3), offset=kwargs.get("offset", 1.0),
                ).fit(X[train], response[train])
                prediction[test] = model.predict(X[test])
                test_response = np.asarray(model._native.predict(X[test]))
                train_response = np.asarray(model._native.predict(X[train]))
                if classification:
                    test_target = (
                        response[test, None] == response_classes[None, :]
                    ).astype(X.dtype)
                    train_target = (
                        response[train, None] == response_classes[None, :]
                    ).astype(X.dtype)
                    test_response_global = np.zeros_like(test_target)
                    train_response_global = np.zeros_like(train_target)
                    for active_index, active_class in enumerate(model.classes_):
                        global_index = int(np.flatnonzero(
                            response_classes == active_class
                        )[0])
                        test_response_global[:, global_index] = (
                            test_response[:, active_index]
                        )
                        train_response_global[:, global_index] = (
                            train_response[:, active_index]
                        )
                    response_scores[test] = test_response_global
                    reference_mean = train_target.mean(axis=0)
                    test_response = test_response_global
                    train_response = train_response_global
                else:
                    test_target = response_matrix[test]
                    train_target = response_matrix[train]
                    reference_mean = train_target.mean(axis=0)
                q2_press += float(np.sum(
                    (test_target.astype(np.float64) -
                     test_response.astype(np.float64)) ** 2
                ))
                q2_reference += float(np.sum(
                    (test_target.astype(np.float64) -
                     reference_mean.astype(np.float64)) ** 2
                ))
                training_r2.append(float(evaluate(
                    train_target, train_response, bycol=False
                )["R2"]))
            evaluation = (
                _evaluate_classification(response, prediction)
                if classification else evaluate(response, prediction)
            )
            q2 = 1.0 - q2_press / q2_reference if q2_reference > 0 else np.nan
            r2 = float(np.nanmean(training_r2))
            metric_name = kwargs.get("selection", "auto").lower()
            if metric_name == "auto":
                metric_name = "accuracy" if classification else "rmsd"
            metric_value = _selection_value(
                metric_name, evaluation,
                q2=q2,
                r2=r2,
            )
            runs.append({
                "Ypred": prediction, "fold": outer, "best_ncomp": selected,
                "metric_name": metric_name, "metric_value": metric_value,
                "Q2Y": q2, "R2Y": r2, "metrics": evaluation,
            })
            all_predictions.append(prediction)
        if classification:
            stacked = np.stack(all_predictions, axis=1)
            combined = np.asarray([
                np.unique(row, return_counts=True)[0][
                    np.argmax(np.unique(row, return_counts=True)[1])
                ] for row in stacked
            ])
        else:
            combined = np.mean(np.stack(all_predictions, axis=0), axis=0)
        return runs, combined

    results, combined = run_nested(y)
    output = {
        "results": results,
        "Ypred": combined,
        "metrics": (
            _evaluate_classification(y, combined)
            if classification else evaluate(y, combined)
        ),
        "bcomp": int(np.bincount(np.concatenate([
            np.asarray(run["best_ncomp"], dtype=int) for run in results
        ])).argmax()),
        "selection_metric": results[0]["metric_name"],
        "Q2Y": np.asarray([run["Q2Y"] for run in results], dtype=float),
        "R2Y": np.asarray([run["R2Y"] for run in results], dtype=float),
        "method": kwargs.get("method", "simpls"),
        "backend": kwargs.get("backend", "cpu"),
    }
    if perm_test:
        rng = np.random.default_rng(seed + 900000)
        sampled = []
        errors = []
        for _ in range(permutations_requested):
            permutation = _permutation_indices(constrain, X.shape[0], rng)
            try:
                null_runs, _ = run_nested(y[permutation])
                statistic = float(np.median([
                    run["metric_value"] for run in null_runs
                ]))
                if not np.isfinite(statistic):
                    raise ValueError("permuted statistic is not finite")
                sampled.append(statistic)
            except Exception as error:  # preserve failed null fits in the audit
                errors.append(str(error))
        observed = float(np.median([run["metric_value"] for run in results]))
        sampled_array = np.asarray(sampled, dtype=float)
        minimize = output["selection_metric"] in {"rmsd", "mae", "mape_percent"}
        extreme = np.sum(sampled_array <= observed) if minimize else np.sum(sampled_array >= observed)
        complete = not errors and sampled_array.size == permutations_requested
        p_value = float((extreme + 1) / (sampled_array.size + 1)) if complete else np.nan
        output.update({
            "permutation_metric": output["selection_metric"],
            "permutation_observed": observed,
            "permutation_sampled": sampled_array,
            "p_value": p_value,
            "permutation_valid": complete,
            "permutation_requested": permutations_requested,
            "permutation_completed": int(sampled_array.size),
            "permutation_failed": len(errors),
            "permutation_errors": errors,
            "permutation_unit": "rows" if constrain is None else "exchangeability blocks",
        })
    return output


def vip(model: PLS) -> np.ndarray | list[np.ndarray]:
    """Return variable-importance paths from a fitted model."""
    return model.vip()


def plot_permutation(result: dict[str, Any], ax: Any | None = None) -> Any:
    """Plot the null distribution and observed permutation statistic."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("plot_permutation requires matplotlib") from error
    if "permutation_sampled" not in result:
        raise ValueError("result does not contain a permutation test")
    if ax is None:
        _, ax = plt.subplots()
    sampled = np.asarray(result["permutation_sampled"], dtype=float)
    ax.scatter(np.arange(1, sampled.size + 1), sampled, label="permuted")
    ax.axhline(result["permutation_observed"], color="black", linestyle="--",
               label="observed")
    ax.set(xlabel="Permutation", ylabel=result["permutation_metric"])
    ax.legend()
    return ax
