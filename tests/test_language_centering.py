import numpy as np
import pytest

from dataset_audit.language_centering import weighted_shrunk_language_centering


def test_weighted_language_centering_shrinks_sparse_means_and_preserves_inputs():
    embeddings = np.asarray([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.7, 0.7],
    ], dtype=np.float32)
    original = embeddings.copy()
    languages = ["aa", "aa", "aa", "bb"]
    weights = np.asarray([1.0, 1.0, 8.0, 1.0])

    centered, metadata = weighted_shrunk_language_centering(
        embeddings, languages, weights, shrinkage_prior_effective_n=2.0
    )

    language_a = next(item for item in metadata["languages"] if item["language"] == "aa")
    assert language_a["effective_sample_size"] == pytest.approx((10.0 ** 2) / (1.0 + 1.0 + 64.0))
    assert language_a["shrinkage_fraction"] == pytest.approx(
        language_a["effective_sample_size"] / (language_a["effective_sample_size"] + 2.0)
    )
    np.testing.assert_array_equal(embeddings, original)
    np.testing.assert_allclose(np.linalg.norm(centered, axis=1), 1.0, atol=2e-6)
    assert centered.dtype == np.float32
    assert metadata["centroid_weighting"].startswith("saved inverse-inclusion weights")


def test_language_centering_rejects_mismatched_or_invalid_weights():
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="one value per embedding"):
        weighted_shrunk_language_centering(vectors, ["en"], np.ones(2))
    with pytest.raises(ValueError, match="must be positive"):
        weighted_shrunk_language_centering(vectors, ["en", "fr"], np.asarray([1.0, 0.0]))
    with pytest.raises(ValueError, match="must be positive"):
        weighted_shrunk_language_centering(vectors, ["en", "fr"], np.ones(2), shrinkage_prior_effective_n=0)
