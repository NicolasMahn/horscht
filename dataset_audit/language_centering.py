#!/usr/bin/env python3
"""Compare frozen embeddings before and after a language-centering diagnostic."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_audit import audit

TRANSFORM_VERSION = "weighted-language-centering-shrinkage-v1"
SHRINKAGE_PRIOR_EFFECTIVE_N = 30.0
MIN_WITHIN_LANGUAGE_SAMPLE_N = 30


def weighted_shrunk_language_centering(
    embeddings: np.ndarray,
    languages: list[str] | np.ndarray,
    weights: np.ndarray,
    shrinkage_prior_effective_n: float = SHRINKAGE_PRIOR_EFFECTIVE_N,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Move each language mean toward the weighted global mean, then unit-normalize."""
    vectors = np.asarray(embeddings, dtype=np.float32)
    language_values = np.asarray(languages, dtype=object)
    row_weights = np.asarray(weights, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError("embeddings must be a nonempty 2D matrix")
    if language_values.shape != (vectors.shape[0],) or row_weights.shape != (vectors.shape[0],):
        raise ValueError("languages and weights must have one value per embedding")
    if not np.isfinite(vectors).all() or not np.isfinite(row_weights).all():
        raise ValueError("embeddings and weights must be finite")
    if np.any(row_weights <= 0) or shrinkage_prior_effective_n <= 0:
        raise ValueError("weights and shrinkage prior must be positive")
    if any(not isinstance(language, str) or not language for language in language_values):
        raise ValueError("language tags must be nonempty strings")

    global_mean = np.average(vectors, axis=0, weights=row_weights).astype(np.float32)
    centered = vectors.copy()
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, language in enumerate(language_values):
        grouped[str(language)].append(index)

    language_stats = []
    for language, indexes in sorted(grouped.items()):
        indices = np.asarray(indexes, dtype=np.int64)
        group_weights = row_weights[indices]
        weight_sum = float(group_weights.sum())
        effective_n = float(weight_sum * weight_sum / np.dot(group_weights, group_weights))
        shrinkage = effective_n / (effective_n + shrinkage_prior_effective_n)
        language_mean = np.average(vectors[indices], axis=0, weights=group_weights).astype(np.float32)
        shift = np.float32(shrinkage) * (language_mean - global_mean)
        adjusted = vectors[indices] - shift
        norms = np.linalg.norm(adjusted, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError(f"centering produced a zero vector for language {language}")
        centered[indices] = adjusted / norms
        language_stats.append({
            "language": language,
            "sample_rows": int(len(indices)),
            "effective_sample_size": effective_n,
            "population_weight": weight_sum,
            "shrinkage_fraction": float(shrinkage),
            "language_mean_norm": float(np.linalg.norm(language_mean)),
            "applied_shift_norm": float(np.linalg.norm(shift)),
        })

    if not np.isfinite(centered).all() or not np.allclose(np.linalg.norm(centered, axis=1), 1.0, atol=2e-5):
        raise RuntimeError("centered vectors are not finite unit vectors")
    return centered, {
        "version": TRANSFORM_VERSION,
        "formula": "z_i = normalize(x_i - alpha_l * (mu_l - mu_global)); alpha_l = n_eff_l / (n_eff_l + tau)",
        "centroid_weighting": "saved inverse-inclusion weights from the stratified sample",
        "effective_sample_size": "(sum w)^2 / sum(w^2), computed per language",
        "shrinkage_prior_effective_n": float(shrinkage_prior_effective_n),
        "global_mean_norm": float(np.linalg.norm(global_mean)),
        "languages": language_stats,
    }


def _nearest_indices(embeddings: np.ndarray, top_k: int = 10, block_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    count = len(embeddings)
    k = min(top_k, count - 1)
    if k < 1:
        raise ValueError("at least two embeddings are required")
    neighbors = np.empty((count, k), dtype=np.int32)
    similarities = np.empty((count, k), dtype=np.float32)
    for start in range(0, count, block_size):
        end = min(start + block_size, count)
        scores = embeddings[start:end] @ embeddings.T
        for local, row_index in enumerate(range(start, end)):
            scores[local, row_index] = -np.inf
            row_scores = scores[local]
            candidates = np.argpartition(row_scores, -k)[-k:]
            candidates = candidates[np.argsort(row_scores[candidates])[::-1]]
            neighbors[row_index] = candidates
            similarities[row_index] = row_scores[candidates]
    return neighbors, similarities


def _between_language_variance_ratio(embeddings: np.ndarray, languages: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    global_mean = np.average(embeddings, axis=0, weights=weights)
    total = float(np.average(np.sum((embeddings - global_mean) ** 2, axis=1), weights=weights))
    between = 0.0
    for language in np.unique(languages):
        indices = np.flatnonzero(languages == language)
        group_weights = weights[indices]
        language_mean = np.average(embeddings[indices], axis=0, weights=group_weights)
        between += float(group_weights.sum() * np.sum((language_mean - global_mean) ** 2) / weights.sum())
    return {
        "between_language_variance": between,
        "total_variance": total,
        "between_language_fraction_of_total": between / total if total else 0.0,
    }


def _neighbor_summary(
    neighbor_indexes: np.ndarray,
    similarities: np.ndarray,
    languages: np.ndarray,
    operations: np.ndarray,
    domains: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    same_language = languages[neighbor_indexes] == languages[:, None]
    same_operation = operations[neighbor_indexes] == operations[:, None]
    same_domain = domains[neighbor_indexes] == domains[:, None]
    cross_language = ~same_language
    cross_counts = cross_language.sum(axis=1)
    cross_op = (same_operation & cross_language).sum(axis=1) / np.maximum(cross_counts, 1)
    cross_domain = (same_domain & cross_language).sum(axis=1) / np.maximum(cross_counts, 1)
    rows_with_cross_language_neighbors = cross_counts > 0
    return {
        "top1_same_language_fraction_weighted": float(np.average(same_language[:, 0], weights=weights)),
        "top10_same_language_fraction_weighted": float(np.average(same_language.mean(axis=1), weights=weights)),
        "top1_same_operation_fraction_weighted": float(np.average(same_operation[:, 0], weights=weights)),
        "top10_same_operation_fraction_weighted": float(np.average(same_operation.mean(axis=1), weights=weights)),
        "top1_same_domain_fraction_weighted": float(np.average(same_domain[:, 0], weights=weights)),
        "top10_same_domain_fraction_weighted": float(np.average(same_domain.mean(axis=1), weights=weights)),
        "cross_language_top10_same_operation_fraction_weighted": float(np.average(cross_op[rows_with_cross_language_neighbors], weights=weights[rows_with_cross_language_neighbors])),
        "cross_language_top10_same_domain_fraction_weighted": float(np.average(cross_domain[rows_with_cross_language_neighbors], weights=weights[rows_with_cross_language_neighbors])),
        "rows_with_cross_language_neighbors_weighted": float(weights[rows_with_cross_language_neighbors].sum()),
        "top1_cosine_mean_weighted": float(np.average(similarities[:, 0], weights=weights)),
        "top1_cosine_p50_weighted": audit._weighted_quantile(similarities[:, 0], weights, 0.5),
        "top1_cosine_p90_weighted": audit._weighted_quantile(similarities[:, 0], weights, 0.9),
    }


def _within_language_neighbor_overlap(
    original: np.ndarray,
    centered: np.ndarray,
    languages: np.ndarray,
    weights: np.ndarray,
    minimum_sample_n: int = MIN_WITHIN_LANGUAGE_SAMPLE_N,
    top_k: int = 10,
) -> dict[str, Any]:
    weighted_overlap = 0.0
    included_weight = 0.0
    included_rows = 0
    included_languages = 0
    per_language = []
    for language in sorted(np.unique(languages)):
        indices = np.flatnonzero(languages == language)
        if len(indices) < minimum_sample_n:
            continue
        k = min(top_k, len(indices) - 1)
        before = original[indices] @ original[indices].T
        after = centered[indices] @ centered[indices].T
        np.fill_diagonal(before, -np.inf)
        np.fill_diagonal(after, -np.inf)
        before_neighbors = np.argpartition(before, -k, axis=1)[:, -k:]
        after_neighbors = np.argpartition(after, -k, axis=1)[:, -k:]
        overlaps = np.fromiter(
            (len(set(first) & set(second)) / k for first, second in zip(before_neighbors, after_neighbors)),
            dtype=np.float32,
            count=len(indices),
        )
        group_weight = float(weights[indices].sum())
        group_overlap = float(np.average(overlaps, weights=weights[indices]))
        weighted_overlap += group_weight * group_overlap
        included_weight += group_weight
        included_rows += len(indices)
        included_languages += 1
        per_language.append({
            "language": str(language),
            "sample_rows": int(len(indices)),
            "population_weight": group_weight,
            "within_language_top10_neighbor_overlap": group_overlap,
        })
    return {
        "minimum_sample_rows_per_language": minimum_sample_n,
        "top_k": top_k,
        "weighted_overlap": weighted_overlap / included_weight if included_weight else None,
        "sample_rows_included": included_rows,
        "sample_population_weight_covered": included_weight,
        "languages_included": included_languages,
        "per_language": per_language,
    }


def run_audit(directory: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in (directory / "sample.jsonl").read_text(encoding="utf-8").splitlines() if line]
    original = np.load(directory / "embeddings.npy", mmap_mode="r")
    if len(rows) != len(original) or not rows:
        raise RuntimeError("frozen sample and original embedding counts do not match")
    weights = np.asarray([float(row["sample_weight"]) for row in rows], dtype=np.float64)
    languages = np.asarray([row["language"] for row in rows], dtype=object)
    operations = np.asarray([row["operation"] for row in rows], dtype=object)
    domains = np.asarray([row["domain"] for row in rows], dtype=object)
    original = np.asarray(original, dtype=np.float32)

    centered, result = weighted_shrunk_language_centering(original, languages, weights)
    centered_path = directory / "language-centered-embeddings-v1.npy"
    np.save(centered_path, centered)
    original_neighbors, original_similarities = _nearest_indices(original)
    centered_neighbors, centered_similarities = _nearest_indices(centered)
    original_metrics = _neighbor_summary(original_neighbors, original_similarities, languages, operations, domains, weights)
    centered_metrics = _neighbor_summary(centered_neighbors, centered_similarities, languages, operations, domains, weights)
    global_overlap = np.fromiter(
        (len(set(before) & set(after)) / len(before) for before, after in zip(original_neighbors, centered_neighbors)),
        dtype=np.float32,
        count=len(rows),
    )
    within_overlap = _within_language_neighbor_overlap(original, centered, languages, weights)
    original_variance = _between_language_variance_ratio(original, languages, weights)
    centered_variance = _between_language_variance_ratio(centered, languages, weights)
    by_language_sample = {
        str(language): int(np.count_nonzero(languages == language)) for language in np.unique(languages)
    }
    result["language_support"] = {
        "sample_language_count": len(by_language_sample),
        "languages_with_at_least_30_sample_rows": sum(count >= 30 for count in by_language_sample.values()),
        "sample_rows_in_languages_with_at_least_30": sum(count for count in by_language_sample.values() if count >= 30),
        "population_weight_in_languages_with_at_least_30": float(sum(
            weights[languages == language].sum() for language, count in by_language_sample.items() if count >= 30
        )),
        "languages_with_effective_n_at_least_30": sum(
            item["effective_sample_size"] >= 30 for item in result["languages"]
        ),
    }
    result["neighbor_retention"] = {
        "global_top10_neighbor_overlap_weighted": float(np.average(global_overlap, weights=weights)),
        "within_language": within_overlap,
    }
    result["language_effect"] = {
        "original": original_variance,
        "centered": centered_variance,
        "between_language_fraction_relative_reduction": (
            1 - centered_variance["between_language_fraction_of_total"] / original_variance["between_language_fraction_of_total"]
            if original_variance["between_language_fraction_of_total"] else 0.0
        ),
    }
    result["nearest_neighbors"] = {
        "original": original_metrics,
        "centered": centered_metrics,
        "weighted_top10_same_operation_change": centered_metrics["top10_same_operation_fraction_weighted"] - original_metrics["top10_same_operation_fraction_weighted"],
        "weighted_top10_same_domain_change": centered_metrics["top10_same_domain_fraction_weighted"] - original_metrics["top10_same_domain_fraction_weighted"],
    }

    snapshot = json.loads((directory / "snapshot.json").read_text(encoding="utf-8"))
    cache = json.loads((directory / "embedding-cache.json").read_text(encoding="utf-8"))
    original_metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    raw_meta = original_metrics["tsne_runs"]
    raw_coords = [np.load(directory / f"tsne-p{item['perplexity']}-seed{item['seed']}.npy") for item in raw_meta]
    centered_metrics_path = directory / "language-centering-v1.json"
    previous = {}
    if centered_metrics_path.is_file():
        try:
            previous = json.loads(centered_metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    expected_runs = [
        {"perplexity": perplexity, "seed": seed}
        for perplexity, seed in audit.TSNE_RUNS
    ]
    reuse_centered_tsne = (
        previous.get("version") == TRANSFORM_VERSION
        and previous.get("shrinkage_prior_effective_n") == SHRINKAGE_PRIOR_EFFECTIVE_N
        and previous.get("input_embedding_cache_key") == cache["cache_key"]
        and [(run.get("perplexity"), run.get("seed")) for run in previous.get("tsne", {}).get("centered_runs", [])]
        == [(run["perplexity"], run["seed"]) for run in expected_runs]
        and all(
            (directory / f"tsne-language-centered-p{run['perplexity']}-seed{run['seed']}.npy").is_file()
            for run in expected_runs
        )
    )
    if reuse_centered_tsne:
        centered_meta = previous["tsne"]["centered_runs"]
    else:
        centered_meta = audit._run_tsne(centered, directory, prefix="language-centered-")
    centered_coords = [
        np.load(directory / f"tsne-language-centered-p{item['perplexity']}-seed{item['seed']}.npy")
        for item in centered_meta
    ]
    centered_checks = []
    projected_neighbors = []
    for meta, coordinates in zip(centered_meta, centered_coords):
        projected = audit._coordinate_neighbors(coordinates)
        projected_neighbors.append(projected)
        overlaps = np.fromiter(
            (len(set(original_row) & set(projected_row)) / len(original_row) for original_row, projected_row in zip(centered_neighbors, projected)),
            dtype=np.float32,
            count=len(rows),
        )
        centered_checks.append({
            "perplexity": meta["perplexity"],
            "seed": meta["seed"],
            "mean_centered_cosine_neighbor_overlap_at_10_weighted": float(np.average(overlaps, weights=weights)),
        })
    layout_overlap = np.fromiter(
        (len(set(first) & set(second)) / len(first) for first, second in zip(projected_neighbors[0], projected_neighbors[1])),
        dtype=np.float32,
        count=len(rows),
    )
    result.update({
        "schema_version": "language-centering-audit-v1",
        "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "input_embedding_cache_key": cache["cache_key"],
        "source_snapshot_sha256": snapshot["source_snapshot_sha256"],
        "sample_size": len(rows),
        "sample_weight_sum": float(weights.sum()),
        "derivative_embedding_path": centered_path.name,
        "normalization": "L2-normalize after the shrunk language-mean shift",
        "tsne": {
            "original_runs": raw_meta,
            "centered_runs": centered_meta,
            "original_cosine_neighbor_checks": original_metrics["tsne_local_neighborhood_checks"],
            "centered_cosine_neighbor_checks": centered_checks,
            "centered_layout_top10_overlap_weighted": float(np.average(layout_overlap, weights=weights)),
            "language_filter_note": "The HTML language selector hides points in the precomputed joint sample layout; it does not refit t-SNE per language.",
        },
    })
    (directory / "language-centering-v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit._write_html(rows, raw_coords, raw_meta, directory, centered_coords=centered_coords, centering_summary=result)
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "audit-2026-09-29")
    args = parser.parse_args()
    result = run_audit(args.output.resolve())
    print(json.dumps({
        "sample_size": result["sample_size"],
        "original_top1_same_language": result["nearest_neighbors"]["original"]["top1_same_language_fraction_weighted"],
        "centered_top1_same_language": result["nearest_neighbors"]["centered"]["top1_same_language_fraction_weighted"],
        "within_language_top10_overlap": result["neighbor_retention"]["within_language"]["weighted_overlap"],
        "report": str(args.output.resolve() / "report.html"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
