import numpy as np
import pytest

import fastpls


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(42)
    X = np.asfortranarray(rng.normal(size=(120, 12)))
    Y = np.asfortranarray(
        np.column_stack((X[:, 0] + X[:, 1], X[:, 2] - X[:, 3]))
    )
    labels = np.where(X[:, 0] + 0.3 * X[:, 1] > 0, "case", "control")
    return X, Y, labels


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["simpls", "plssvd", "opls", "kernelpls"])
def test_regression_families_are_finite(data, dtype, method):
    X, Y, _ = data
    kwargs = {"kernel": "rbf", "gamma": 0.1} if method == "kernelpls" else {}
    model = fastpls.PLS(n_components=2, method=method, seed=9, **kwargs).fit(
        X[:90].astype(dtype), Y[:90].astype(dtype)
    )
    prediction = model.predict(X[90:].astype(dtype))
    assert prediction.shape == (30, 2)
    assert prediction.dtype == dtype
    assert np.isfinite(prediction).all()
    assert model.scores_ is None


def test_scores_are_opt_in(data):
    X, Y, _ = data
    stored = fastpls.PLS(n_components=2, store_scores=True).fit(X[:90], Y[:90])
    assert stored.scores_.shape == (90, 2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["simpls", "plssvd", "opls", "kernelpls"])
@pytest.mark.parametrize("classifier", ["argmax", "lda"])
def test_classification_families_and_ranks(data, dtype, method, classifier):
    X, _, labels = data
    kwargs = {"kernel": "polynomial", "gamma": 0.1} if method == "kernelpls" else {}
    model = fastpls.PLS(
        n_components=2,
        method=method,
        classifier=classifier,
        seed=9,
        **kwargs,
    ).fit(X[:90].astype(dtype), labels[:90])
    prediction = model.predict(X[90:].astype(dtype))
    ranked = model.predict(X[90:].astype(dtype), top=2)
    assert prediction.shape == (30,)
    assert ranked.shape == (30, 2)
    assert np.array_equal(prediction, ranked[:, 0])
    assert set(np.unique(prediction)).issubset(set(model.classes_))


def test_fixed_seed_is_repeatable(data):
    X, Y, _ = data
    first = fastpls.PLS(n_components=3, seed=17).fit(X[:90], Y[:90])
    second = fastpls.PLS(n_components=3, seed=17).fit(X[:90], Y[:90])
    np.testing.assert_array_equal(first.predict(X[90:]), second.predict(X[90:]))


def test_float32_and_float64_agree(data):
    X, Y, _ = data
    model32 = fastpls.PLS(n_components=3, seed=17).fit(
        X[:90].astype(np.float32), Y[:90].astype(np.float32)
    )
    model64 = fastpls.PLS(n_components=3, seed=17).fit(X[:90], Y[:90])
    np.testing.assert_allclose(
        model32.predict(X[90:].astype(np.float32)),
        model64.predict(X[90:]),
        rtol=2e-4,
        atol=2e-5,
    )


def test_unsupported_accelerator_does_not_fallback(data):
    X, Y, _ = data
    with pytest.raises(ValueError, match="not silently replaced"):
        fastpls.PLS(backend="cuda").fit(X, Y)


def test_evaluate_regression_and_classification():
    regression = fastpls.evaluate([1.0, 2.0], [1.0, 2.0])
    assert regression["R2"] == pytest.approx(1.0)
    assert regression["RMSD"] == pytest.approx(0.0)
    classification = fastpls.evaluate(
        np.array(["a", "b"]), np.array([["a", "b"], ["a", "b"]])
    )
    assert classification["accuracy"] == pytest.approx(0.5)
    assert classification["top_accuracy"] == pytest.approx(1.0)


def test_regression_rejects_top(data):
    X, Y, _ = data
    model = fastpls.PLS(n_components=2).fit(X[:90], Y[:90])
    with pytest.raises(ValueError, match="only for classification"):
        model.predict(X[90:], top=2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fastsvd_reconstructs_low_rank_matrix(dtype):
    left = np.arange(1, 41, dtype=dtype).reshape(20, 2)
    right = np.linspace(-1, 1, 16, dtype=dtype).reshape(2, 8)
    matrix = np.asfortranarray(left @ right)
    result = fastpls.fastsvd(matrix, 2, seed=7)
    reconstructed = (result["u"] * result["d"]) @ result["vt"]
    tolerance = 2e-4 if dtype == np.float32 else 1e-10
    np.testing.assert_allclose(reconstructed, matrix, rtol=tolerance, atol=tolerance)


def test_fastcor_and_capability_flags(data):
    X, _, _ = data
    np.testing.assert_allclose(fastpls.fastcor(X), np.corrcoef(X, rowvar=False))
    assert fastpls.has_cuda() is False
    assert fastpls.has_metal() is False
