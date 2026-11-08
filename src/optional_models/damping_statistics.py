"""Weighted first-embedding statistics for a strictly matched model population.

No simulations, file access, or global random state are used here. Missing model
outcomes are coverage gaps, not non-events. The full input population is always
the fraction denominator; no implicit joint-valid cohort selection is performed.
"""
from __future__ import annotations

from math import fsum

import numpy as np

from .damping_models import OutcomeStatus, REPO_MODEL, XUE_MODEL

__all__ = ["summarize_flux"]

_MODELS = ((REPO_MODEL, "repo"), (XUE_MODEL, "xue"))
_STATUS_ALIASES = {
    "censored": OutcomeStatus.CENSORED.value,
    "invalid": OutcomeStatus.INVALID.value,
    "out_of_range": OutcomeStatus.OUT_OF_RANGE.value,
    "terminated": OutcomeStatus.TERMINATED.value,
}
_STATUSES = {status.value for status in OutcomeStatus}


def _nonnegative_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) and number >= 0 else None


def _order(value):
    """Permit integer or string identifiers without depending on row order."""
    return type(value).__name__, repr(value)


def _matched_rows(rows):
    populations = {model: {} for model, _ in _MODELS}
    for row in rows:
        try:
            model = row["model_id"]
            key = (row["realization"], row["source_id"])
            hash(key)
            if any(item is None for item in key):
                raise ValueError("realization and source_id must not be None")
            if model not in populations:
                raise ValueError(f"unknown model_id: {model!r}")
            if key in populations[model]:
                raise ValueError(f"duplicate source key for {model}: {key!r}")
            weight = _nonnegative_number(row["source_weight"])
            if weight is None:
                raise ValueError("source_weight must be finite and nonnegative")
            status = row["status"]
            status = _STATUS_ALIASES.get(status, status)
            if status not in _STATUSES:
                raise ValueError(f"unknown outcome status: {status!r}")
        except (KeyError, TypeError) as exc:
            raise ValueError("rows require model_id, hashable realization/source_id, "
                             "source_weight, and a recognized status") from exc
        populations[model][key] = (row, weight, status)

    repo, xue = (populations[model] for model, _ in _MODELS)
    if repo.keys() != xue.keys():
        raise ValueError("models must have identical (realization, source_id) keys")
    keys = sorted(repo, key=lambda key: (_order(key[0]), _order(key[1])))
    for key in keys:
        if repo[key][1] != xue[key][1]:
            raise ValueError(f"source_weight mismatch between models for {key!r}")
    return populations, keys


def _coverage(row, status, right_edges):
    """Known first-event history through each upper bin edge, plus event time."""
    if status == OutcomeStatus.EMBEDDED:
        event = _nonnegative_number(row.get("t_embed_myr"))
        if event is not None:
            return np.ones(len(right_edges), dtype=bool), event, None
        reason = "invalid_event_time"
    elif status == OutcomeStatus.TERMINATED:
        # A physical termination is an absorbing competing outcome.
        return np.ones(len(right_edges), dtype=bool), None, None
    elif status in (OutcomeStatus.CENSORED, OutcomeStatus.ALIGNED_ECCENTRIC):
        censor = _nonnegative_number(row.get("censor_time_myr"))
        if censor is not None:
            return right_edges <= censor, None, "censored_before_bin_end"
        reason = "invalid_censor_time"
    else:
        reason = status
    return np.zeros(len(right_edges), dtype=bool), None, reason


def _effective_count(weights):
    """Kish count of contributing events, invariant to a common weight scale."""
    if not weights:
        return 0.0
    scale = max(weights)
    if scale == 0:
        return 0.0
    scaled = [weight / scale for weight in weights]
    return fsum(scaled) ** 2 / fsum(weight * weight for weight in scaled)


def _interval(values):
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def summarize_flux(rows, bin_edges_myr, *, bootstrap_seed=123,
                   bootstrap_draws=1000, min_effective_count=5) -> list[dict]:
    """Summarize matched weighted first events, returning one dictionary per bin.

    Input rows contain ``model_id, realization, source_id, source_weight,
    status, t_embed_myr, censor_time_myr, r_embed_cm``. Only the two imported
    model IDs are accepted. Each (realization, source_id) must occur exactly
    once per model with identical finite, nonnegative weights. OutcomeStatus
    values and the short aliases censored/invalid/out_of_range/terminated are
    accepted. Radius is immaterial to time statistics and is not inspected.
    Invalid/missing event or censor times produce incomplete coverage.

    Bins are [left, right), with the final bin including its right endpoint.
    Cumulative counts start at time zero, including events before the first
    supplied edge, and use the same right endpoint convention as the bins.
    An embedded outcome supplies its entire first-event history. A physical
    termination is an absorbing non-event. Censored/aligned-eccentric outcomes
    establish no event only through their finite censor_time_myr; later bins
    are incomplete. Unavailable/invalid/out-of-range outcomes are unevaluated.

    Schema (``p`` is repo or xue):
      * bin_index, time_left_myr, time_right_myr, right_inclusive;
        n_realizations, total_weight (one model's input weights, summed over
        realizations), mean_total_weight, confidence_level, confidence_method.
      * p_completeness: complete/incomplete/no_population; p_evaluated_weight
        and p_incomplete_reasons describe coverage through the bin end.
      * p_mean_count, p_flux (mean count / bin width, Myr^-1),
        p_cumulative_mean_count, p_cumulative_fraction. Fractions are the
        arithmetic mean of realization fractions, each using ALL its input
        weight, never just evaluated weight. These estimates and their
        corresponding *_ci95 fields are None unless every realization has
        positive input weight and complete coverage of positive-weight rows.
      * p_observed_mean_count, p_observed_cumulative_mean_count,
        p_observed_cumulative_fraction retain measured lower bounds even with
        incomplete coverage (the fraction is None for a zero-weight cohort).
      * p_effective_count: Kish count of positive-weight bin events pooled
        across realizations. ratio = repo_flux / xue_flux only for complete
        cohorts with both fluxes positive and BOTH effective counts at least
        min_effective_count. ratio_status is defined, incomplete, zero, or
        low_count (in that precedence); otherwise ratio and ratio_ci95 are
        None. ratio_ci_status also identifies zero bootstrap means.
      * per_realization: dictionaries with realization, total_weight, and
        p_evaluated_weight, p_completeness, p_count, p_flux,
        p_cumulative_count, p_cumulative_fraction, p_observed_count,
        p_observed_cumulative_count, p_observed_cumulative_fraction.

    Confidence intervals are 95% percentile *realization bootstrap mean
    intervals*, not event/Poisson intervals. Whole realizations are resampled
    with replacement, using identical indices for both models, all bins and
    all metrics; each realization receives equal weight. Ratio intervals are
    ratios of these paired bootstrap means, and are withheld if any draw has
    a zero mean for either model. A single realization gives degenerate mean
    intervals. bootstrap_draws must be a positive integer. Empty rows return
    []; invalid edges, unmatched rows, weights, or controls raise ValueError.
    The input rows are not modified and NumPy's global RNG is not touched.
    """
    try:
        edges = np.asarray(bin_edges_myr, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("bin edges must be finite, nonnegative and increasing") from exc
    if (edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges))
            or np.any(edges < 0) or np.any(np.diff(edges) <= 0)):
        raise ValueError("bin edges must be finite, nonnegative and strictly increasing")
    if (isinstance(bootstrap_draws, (bool, np.bool_))
            or not isinstance(bootstrap_draws, (int, np.integer)) or bootstrap_draws < 1):
        raise ValueError("bootstrap_draws must be a positive integer")
    threshold = _nonnegative_number(min_effective_count)
    if threshold is None:
        raise ValueError("min_effective_count must be finite and nonnegative")
    populations, keys = _matched_rows(rows)
    if not keys:
        return []

    realizations = sorted({key[0] for key in keys}, key=_order)
    r_index = {realization: index for index, realization in enumerate(realizations)}
    n_realizations, n_bins = len(realizations), len(edges) - 1
    shape = (2, n_realizations, n_bins)
    counts, cumulative, evaluated = (np.zeros(shape) for _ in range(3))
    complete = np.ones(shape, dtype=bool)
    reasons = [[set() for _ in range(n_bins)] for _ in _MODELS]
    event_weights = [[[] for _ in range(n_bins)] for _ in _MODELS]
    totals = np.array([fsum(populations[REPO_MODEL][key][1] for key in keys
                            if key[0] == realization) for realization in realizations])
    for model_index, (model, _) in enumerate(_MODELS):
        for key in keys:
            row, weight, status = populations[model][key]
            r = r_index[key[0]]
            known, event, reason = _coverage(row, status, edges[1:])
            evaluated[model_index, r] += weight * known
            if weight > 0:
                complete[model_index, r] &= known
                for b in np.flatnonzero(~known):
                    reasons[model_index][b].add(reason)
            if event is None or weight == 0:
                continue
            before_end = event < edges[1:]
            before_end[-1] = event <= edges[-1]
            cumulative[model_index, r] += weight * before_end
            if edges[0] <= event <= edges[-1]:
                b = min(int(np.searchsorted(edges, event, side="right")) - 1, n_bins - 1)
                counts[model_index, r, b] += weight
                event_weights[model_index][b].append(weight)

    # Equal realization means, with one shared resampling plan for every metric.
    rng = np.random.default_rng(bootstrap_seed)
    sampled = rng.integers(n_realizations, size=(bootstrap_draws, n_realizations))
    widths = np.diff(edges)
    fractions = np.divide(cumulative, totals[None, :, None],
                          out=np.full(shape, np.nan), where=totals[None, :, None] > 0)
    results = []
    for b in range(n_bins):
        result = {
            "bin_index": b, "time_left_myr": float(edges[b]),
            "time_right_myr": float(edges[b + 1]), "right_inclusive": b == n_bins - 1,
            "n_realizations": n_realizations, "total_weight": float(fsum(totals)),
            "mean_total_weight": float(np.mean(totals)), "confidence_level": 0.95,
            "confidence_method": "paired_realization_bootstrap_mean_interval",
            "per_realization": [],
        }
        for r, realization in enumerate(realizations):
            item = {"realization": realization, "total_weight": float(totals[r])}
            for m, (_, prefix) in enumerate(_MODELS):
                covered = bool(complete[m, r, b] and totals[r] > 0)
                item.update({
                    f"{prefix}_completeness": ("no_population" if totals[r] == 0 else
                                                "complete" if covered else "incomplete"),
                    f"{prefix}_evaluated_weight": float(evaluated[m, r, b]),
                    f"{prefix}_observed_count": float(counts[m, r, b]),
                    f"{prefix}_observed_cumulative_count": float(cumulative[m, r, b]),
                    f"{prefix}_observed_cumulative_fraction": (
                        float(fractions[m, r, b]) if totals[r] > 0 else None),
                    f"{prefix}_count": float(counts[m, r, b]) if covered else None,
                    f"{prefix}_flux": float(counts[m, r, b] / widths[b]) if covered else None,
                    f"{prefix}_cumulative_count": float(cumulative[m, r, b]) if covered else None,
                    f"{prefix}_cumulative_fraction": float(fractions[m, r, b]) if covered else None,
                })
            result["per_realization"].append(item)

        for m, (_, prefix) in enumerate(_MODELS):
            covered = bool(np.all(complete[m, :, b]) and np.all(totals > 0))
            result.update({
                f"{prefix}_completeness": ("no_population" if not np.any(totals > 0) else
                                            "complete" if covered else "incomplete"),
                f"{prefix}_evaluated_weight": float(np.sum(evaluated[m, :, b])),
                f"{prefix}_incomplete_reasons": sorted(reasons[m][b]),
                f"{prefix}_observed_mean_count": float(np.mean(counts[m, :, b])),
                f"{prefix}_observed_cumulative_mean_count": float(np.mean(cumulative[m, :, b])),
                f"{prefix}_observed_cumulative_fraction": (
                    float(np.mean(fractions[m, :, b])) if np.all(totals > 0) else None),
                f"{prefix}_effective_count": _effective_count(event_weights[m][b]),
            })
            metrics = {"mean_count": counts[m, :, b],
                       "flux": counts[m, :, b] / widths[b],
                       "cumulative_mean_count": cumulative[m, :, b],
                       "cumulative_fraction": fractions[m, :, b]}
            for name, values in metrics.items():
                result[f"{prefix}_{name}"] = float(np.mean(values)) if covered else None
                result[f"{prefix}_{name}_ci95"] = (
                    _interval(np.mean(values[sampled], axis=1)) if covered else None)

        if any(result[f"{p}_completeness"] != "complete" for _, p in _MODELS):
            ratio_status = "incomplete"
        elif result["repo_flux"] == 0 or result["xue_flux"] == 0:
            ratio_status = "zero"
        elif min(result[f"{p}_effective_count"] for _, p in _MODELS) < threshold:
            ratio_status = "low_count"
        else:
            ratio_status = "defined"
        result.update(ratio_status=ratio_status, ratio=None, ratio_ci95=None,
                      ratio_ci_status=ratio_status)
        if ratio_status == "defined":
            result["ratio"] = result["repo_flux"] / result["xue_flux"]
            repo_means = np.mean(counts[0, :, b][sampled], axis=1)
            xue_means = np.mean(counts[1, :, b][sampled], axis=1)
            if np.all(repo_means > 0) and np.all(xue_means > 0):
                result["ratio_ci95"] = _interval(repo_means / xue_means)
            else:
                result["ratio_ci_status"] = "undefined_zero_bootstrap_mean"
        results.append(result)
    return results
