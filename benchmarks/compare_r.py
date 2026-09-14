#!/usr/bin/env python3
"""Compare Python and R wrappers using identical inputs and controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from fastpls import PLS, evaluate


def make_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(1, 2401, dtype=np.float64)[:, None]
    columns = np.arange(1, 241, dtype=np.float64)[None, :]
    X = np.sin(rows * columns * 0.0017) + np.cos(
        rows * (columns + 2) * 0.0011
    )
    class_index = np.arange(X.shape[0]) % 8
    for value in range(8):
        X[class_index == value, (4 * value):(4 * value + 4)] += 2.0
    coefficients = np.sin(
        np.arange(1, 241, dtype=np.float64)[:, None]
        * np.arange(1, 33, dtype=np.float64)[None, :]
        * 0.013
    )
    Y = X @ coefficients / X.shape[1]
    labels = np.array([f"class-{value}" for value in class_index])
    return X[:1800], X[1800:], Y[:1800], labels[:1800]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r-library", type=Path, default=Path("/tmp/fastPLS-r-lib"))
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    Xtrain, Xtest, Ytrain, labels = make_data()
    records = []
    with tempfile.TemporaryDirectory(prefix="fastpls-parity-") as temporary:
        root = Path(temporary)
        np.savetxt(root / "Xtrain.csv", Xtrain, delimiter=",")
        np.savetxt(root / "Xtest.csv", Xtest, delimiter=",")
        np.savetxt(root / "Ytrain.csv", Ytrain, delimiter=",")
        (root / "labels.txt").write_text("\n".join(labels) + "\n")
        subprocess.run(
            [
                "Rscript",
                str(Path(__file__).with_suffix(".R")),
                str(root),
                str(root),
                str(args.r_library),
                str(args.repetitions),
            ],
            check=True,
        )
        for method in ("simpls", "plssvd"):
            timings = []
            prediction = None
            for _ in range(args.repetitions):
                started = time.perf_counter()
                model = PLS(n_components=8, method=method, seed=17).fit(Xtrain, Ytrain)
                prediction = model.predict(Xtest)
                timings.append(time.perf_counter() - started)
            r_prediction = np.loadtxt(
                root / f"r_{method}_regression.csv", delimiter=",", skiprows=1
            )
            labels_model = PLS(
                n_components=8, method=method, classifier="lda", seed=17
            ).fit(Xtrain, labels)
            python_labels = labels_model.predict(Xtest)
            r_labels = np.loadtxt(root / f"r_{method}_labels.txt", dtype=str)
            records.append(
                {
                    "method": method,
                    "max_abs_prediction_difference": float(
                        np.max(np.abs(prediction - r_prediction))
                    ),
                    "relative_prediction_error": float(
                        np.linalg.norm(prediction - r_prediction) /
                        max(np.linalg.norm(r_prediction), np.finfo(float).eps)
                    ),
                    "classification_agreement": float(np.mean(python_labels == r_labels)),
                    "python_median_seconds": float(np.median(timings)),
                    "r_seconds": float(
                        (root / f"r_{method}_seconds.txt").read_text().strip()
                    ),
                    "python_metrics": evaluate(
                        Xtest @ np.sin(
                            np.arange(1, 241, dtype=np.float64)[:, None]
                            * np.arange(1, 33, dtype=np.float64)[None, :]
                            * 0.013
                        ) / Xtest.shape[1],
                        prediction,
                    ),
                }
            )
    output = {
        "fastpls_r_version": "0.99.66",
        "fastpls_core_commit": "3854e369c1f0cd615e58b968b7721efb4b2a3146",
        "repetitions": args.repetitions,
        "records": records,
    }
    rendered = json.dumps(output, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
