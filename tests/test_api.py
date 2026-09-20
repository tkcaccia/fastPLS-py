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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("value", [0.0, 2.5])
def test_constant_response_returns_intercept_only_model(dtype, value):
    rng = np.random.default_rng(81)
    X = np.asfortranarray(rng.normal(size=(41, 30)).astype(dtype))
    y = np.full(41, value, dtype=dtype)
    model = fastpls.PLS(n_components=10, seed=20261542).fit(X, y)
    prediction = model.predict(X[:7])
    assert model.requested_n_components_ == 10
    assert model.n_components_ == 0
    assert prediction.shape == (7, 1)
    np.testing.assert_array_equal(prediction, np.full((7, 1), value, dtype=dtype))


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


def test_model_and_component_selection_share_preprocessing_defaults(data):
    X, Y, _ = data
    selected = fastpls.pls_single_cv(
        X, Y, [1, 2], kfold=3, fit=False, seed=19
    )
    default_model = fastpls.PLS(
        n_components=selected["best_ncomp"], seed=19
    ).fit(X, Y)
    centered_model = fastpls.PLS(
        n_components=selected["best_ncomp"], scaling="centering", seed=19
    ).fit(X, Y)
    np.testing.assert_array_equal(
        default_model.predict(X), centered_model.predict(X)
    )


def test_kernel_default_gamma_is_inverse_predictor_count(data):
    X, Y, _ = data
    default_model = fastpls.PLS(
        n_components=2, method="kernelpls", kernel="rbf", seed=23
    ).fit(X, Y)
    explicit_model = fastpls.PLS(
        n_components=2, method="kernelpls", kernel="rbf",
        gamma=1.0 / X.shape[1], seed=23,
    ).fit(X, Y)
    np.testing.assert_array_equal(
        default_model.predict(X), explicit_model.predict(X)
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
    integer_regression = fastpls.evaluate(
        np.array([1, 2, 3]), np.array([1, 2, 3])
    )
    assert integer_regression["task"] == "regression"


def test_multivariate_rpd_uses_response_wise_centering():
    observed = np.array([[0.0, 100.0], [1.0, 101.0], [2.0, 102.0]])
    predicted = observed + 1.0
    result = fastpls.evaluate(observed, predicted, bycol=False)
    centered = observed - observed.mean(axis=0)
    expected = np.sqrt(np.sum(centered ** 2) / (observed.size - 1))
    assert result["RPD"] == pytest.approx(expected)


@pytest.mark.parametrize("ncomp", [0, 1.5, np.nan, [1, 2.5]])
def test_component_controls_reject_non_positive_integers(data, ncomp):
    X, Y, _ = data
    with pytest.raises(ValueError, match="positive integers|positive integer"):
        if isinstance(ncomp, list):
            fastpls.pls_single_cv(X, Y, ncomp, kfold=3)
        else:
            fastpls.PLS(n_components=ncomp).fit(X, Y)


def test_stratified_folds_reject_empty_validation_folds():
    X = np.arange(30.0).reshape(6, 5)
    labels = np.array(["a", "a", "a", "b", "b", "b"])
    with pytest.raises(ValueError, match="cannot populate every fold"):
        fastpls.pls_single_cv(
            X, labels, [1], kfold=5, classifier="lda", fit=False
        )


def test_regression_rejects_top(data):
    X, Y, _ = data
    model = fastpls.PLS(n_components=2).fit(X[:90], Y[:90])
    with pytest.raises(ValueError, match="only for classification"):
        model.predict(X[90:], top=2)


def test_pls_evaluates_an_independent_test_set(data):
    X, Y, labels = data
    regression = fastpls.pls(X[:90], Y[:90], X[90:], Y[90:], n_components=2)
    assert regression["metrics"]["task"] == "regression"
    assert np.isfinite(regression["metrics"]["Q2"])
    classification = fastpls.pls(
        X[:90], labels[:90], X[90:], labels[90:],
        n_components=2, classifier="lda", top=2,
    )
    assert classification["prediction"].shape == (30, 2)
    assert classification["metrics"]["task"] == "classification"


def test_explicit_classifier_supports_numeric_labels(data):
    X, _, labels = data
    numeric = np.unique(labels, return_inverse=True)[1]
    result = fastpls.pls(
        X[:90], numeric[:90], X[90:], numeric[90:],
        n_components=2, classifier="lda",
    )
    assert result["metrics"]["task"] == "classification"


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("method", "unknown"), ("classifier", "knn"),
     ("scaling", "unit"), ("kernel", "sigmoid")],
)
def test_model_rejects_unknown_algorithm_controls(data, keyword, value):
    X, Y, _ = data
    arguments = {keyword: value}
    with pytest.raises(ValueError, match=keyword):
        fastpls.PLS(**arguments).fit(X, Y)


@pytest.mark.parametrize("value", [0, 1.5, np.nan])
def test_fastsvd_rejects_invalid_component_count(data, value):
    X, _, _ = data
    with pytest.raises(ValueError, match="n_components"):
        fastpls.fastsvd(X, value)


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
    np.testing.assert_allclose(fastpls.fastcor(X), np.corrcoef(X))
    np.testing.assert_allclose(
        fastpls.fastcor(X, byrow=False), np.corrcoef(X, rowvar=False)
    )
    np.testing.assert_allclose(
        fastpls.fastcor(X[:4], X[4:8], diag=False),
        np.corrcoef(X[:8])[:4, 4:8],
    )
    np.testing.assert_allclose(
        fastpls.fastcor(X[:4], X[4:8]),
        np.diag(np.corrcoef(X[:8])[:4, 4:8]),
    )
    assert fastpls.has_cuda() is False
    assert fastpls.has_metal() is False


def test_integer_regression_is_not_misclassified():
    result = fastpls.evaluate([1, 2, 3], [1.1, 2.1, 2.9])
    assert result["task"] == "regression"
    assert result["RMSD"] == pytest.approx(0.1)


@pytest.mark.parametrize("method", ["simpls", "plssvd", "opls", "kernelpls"])
@pytest.mark.parametrize("classifier", ["argmax", "lda"])
def test_single_cv_covers_every_family_and_classifier(data, method, classifier):
    X, _, labels = data
    kwargs = {"kernel": "rbf", "gamma": 0.1} if method == "kernelpls" else {}
    result = fastpls.pls_single_cv(
        X, labels, [1, 2], kfold=3, method=method, classifier=classifier,
        selection="balanced_accuracy", seed=11, fit=False, **kwargs,
    )
    assert result["best_ncomp"] in (1, 2)
    assert result["pred"][0].shape == (X.shape[0],)
    assert np.isfinite(result["best_metric_value"])
    assert np.all(result["status"] == 1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["simpls", "plssvd", "opls", "kernelpls"])
def test_single_cv_multivariate_regression(data, dtype, method):
    X, Y, _ = data
    kwargs = {"kernel": "rbf", "gamma": 0.1} if method == "kernelpls" else {}
    result = fastpls.pls_single_cv(
        X.astype(dtype), Y.astype(dtype), [1, 2], kfold=3,
        selection="RMSD", fit=False, seed=11, method=method, **kwargs,
    )
    assert result["best_ncomp"] in (1, 2)
    assert np.all(np.isfinite(result["RMSD"]))
    assert result["pred"][0].shape == Y.shape


def test_grouped_double_cv_and_corrected_permutation(data):
    X, _, labels = data
    groups = np.repeat(np.arange(40), 3)
    result = fastpls.pls_double_cv(
        X, labels, [1, 2], constrain=groups, runn=1,
        kfold_inner=2, kfold_outer=2, classifier="lda",
        selection="balanced_accuracy", perm_test=True, times=2, seed=11,
    )
    assert result["Ypred"].shape == labels.shape
    assert 0 < result["p_value"] <= 1
    assert result["permutation_requested"] == 2
    assert result["permutation_unit"] == "exchangeability blocks"


@pytest.mark.parametrize("selection", ["Q2Y", "R2Y"])
def test_double_cv_response_metric_selection_is_finite(data, selection):
    X, Y, _ = data
    result = fastpls.pls_double_cv(
        X, Y, [1, 2], runn=1, kfold_inner=2, kfold_outer=2,
        selection=selection, seed=13,
    )
    assert np.isfinite(result[selection]).all()
    assert np.isfinite(result["results"][0][selection])
    assert np.isfinite(result["results"][0]["metric_value"])


def test_vip_requires_scores_and_covers_responses(data):
    X, Y, _ = data
    compact = fastpls.PLS(n_components=2).fit(X, Y)
    with pytest.raises(ValueError, match="store_scores"):
        compact.vip()
    stored = fastpls.PLS(n_components=2, store_scores=True).fit(X, Y)
    values = stored.vip()
    assert isinstance(values, list)
    assert len(values) == Y.shape[1]
    assert values[0].shape == (2, X.shape[1])


def test_plot_permutation_returns_requested_axes():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    returned = fastpls.plot_permutation({
        "permutation_sampled": np.array([0.2, 0.3]),
        "permutation_observed": 0.8,
        "permutation_metric": "accuracy",
    }, ax=axes)
    assert returned is axes
    plt.close(figure)
