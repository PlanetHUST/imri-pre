"""Population conservation, coverage and paired realization inference."""
from copy import deepcopy

import numpy as np
import pytest

from physical_process.damping_models import OutcomeStatus, REPO_MODEL, XUE_MODEL
from physical_process.damping_statistics import summarize_flux


def pair(source, weight=1.0, *, realization=0, repo_time=0.5, xue_time=0.5,
         repo_status=OutcomeStatus.EMBEDDED, xue_status=OutcomeStatus.EMBEDDED,
         censor=10.0):
    return [
        dict(model_id=model, realization=realization, source_id=source,
             source_weight=weight, status=status, t_embed_myr=time,
             censor_time_myr=censor, r_embed_cm=1.0e16 if time is not None else None)
        for model, status, time in ((REPO_MODEL, repo_status, repo_time),
                                    (XUE_MODEL, xue_status, xue_time))
    ]


def test_weighted_counts_conservation_boundaries_and_censoring():
    rows = (pair("zero", 2, repo_time=0, xue_time=4)
            + pair("boundary", 3, repo_time=1, xue_time=1)
            + pair("last", 5, repo_time=4, xue_time=0)
            + pair("censored", 7, repo_time=None, xue_time=None,
                   repo_status=OutcomeStatus.CENSORED, xue_status=OutcomeStatus.CENSORED,
                   censor=4)
            + pair("terminated", 11, repo_time=None, xue_time=None,
                   repo_status=OutcomeStatus.TERMINATED, xue_status=OutcomeStatus.TERMINATED))
    result = summarize_flux(rows, [0, 1, 2, 4], bootstrap_draws=40)
    assert [b["repo_mean_count"] for b in result] == [2, 3, 5]
    assert [b["xue_mean_count"] for b in result] == [5, 3, 2]
    assert [b["repo_flux"] for b in result] == [2, 3, 2.5]
    assert [b["repo_cumulative_mean_count"] for b in result] == [2, 5, 10]
    assert [b["right_inclusive"] for b in result] == [False, False, True]
    for prefix in ("repo", "xue"):
        assert sum(b[f"{prefix}_mean_count"] for b in result) == 10
        assert sum(b[f"{prefix}_flux"] * (b["time_right_myr"] - b["time_left_myr"])
                   for b in result) == 10
        assert result[-1][f"{prefix}_cumulative_fraction"] == pytest.approx(10 / 28)
        for b in result:
            assert b["total_weight"] == 28
            assert b[f"{prefix}_evaluated_weight"] == 28
            assert b[f"{prefix}_completeness"] == "complete"
            assert b[f"{prefix}_mean_count_ci95"] == [b[f"{prefix}_mean_count"]] * 2
            assert b["per_realization"][0][f"{prefix}_count"] == b[f"{prefix}_mean_count"]


def test_cumulative_includes_events_before_first_edge():
    result = summarize_flux(pair("early", 2, repo_time=0, xue_time=0)
                            + pair("late", 3, repo_time=3, xue_time=3), [1, 2, 3])
    assert [b["repo_mean_count"] for b in result] == [0, 3]
    assert [b["repo_cumulative_mean_count"] for b in result] == [2, 5]
    assert result[-1]["repo_cumulative_fraction"] == 1


@pytest.mark.parametrize("status", [OutcomeStatus.UNAVAILABLE, OutcomeStatus.INVALID,
                                    OutcomeStatus.OUT_OF_RANGE, "invalid", "out_of_range"])
def test_unevaluated_outcomes_are_never_zero_flux(status):
    rows = pair("event", 2) + pair("missing", 8, xue_status=status, xue_time=None)
    result = summarize_flux(rows, [0, 1], bootstrap_draws=20)[0]
    assert result["repo_completeness"] == "complete"
    assert result["repo_flux"] == 10
    assert result["xue_completeness"] == "incomplete"
    assert result["total_weight"] == 10
    assert result["xue_evaluated_weight"] == 2
    assert result["xue_observed_mean_count"] == 2
    assert result["xue_observed_cumulative_fraction"] == 0.2  # all input weight
    assert result["xue_flux"] is None
    assert result["xue_mean_count"] is None
    assert result["xue_cumulative_fraction"] is None
    assert result["xue_flux_ci95"] is None
    assert result["xue_incomplete_reasons"]
    assert result["ratio_status"] == "incomplete"
    assert result["ratio"] is None
    assert result["ratio_ci95"] is None
    assert result["per_realization"][0]["xue_count"] is None


def test_entirely_unavailable_is_distinguished_from_evaluated_zero():
    result = summarize_flux(pair("missing", repo_time=None, xue_time=None,
                                 repo_status="unavailable", xue_status="censored"), [0, 1])[0]
    assert result["repo_evaluated_weight"] == 0
    assert result["repo_flux"] is None
    assert result["xue_evaluated_weight"] == 1
    assert result["xue_flux"] == 0
    assert result["xue_flux_ci95"] == [0, 0]
    assert result["ratio_status"] == "incomplete"


def test_early_censoring_reduces_coverage_only_after_censor_time():
    rows = pair("event", 2) + pair("early_censor", 8, repo_time=None, xue_time=None,
                                   repo_status="censored", xue_status="censored", censor=1)
    first, second = summarize_flux(rows, [0, 1, 2])
    assert first["repo_completeness"] == "complete"
    assert first["repo_flux"] == 2
    assert first["repo_cumulative_fraction"] == 0.2
    assert second["repo_completeness"] == "incomplete"
    assert second["repo_evaluated_weight"] == 2
    assert second["repo_flux"] is None
    assert second["repo_observed_mean_count"] == 0
    assert second["repo_observed_cumulative_fraction"] == 0.2
    assert second["repo_incomplete_reasons"] == ["censored_before_bin_end"]


@pytest.mark.parametrize("bad_time", [None, -1, np.nan, np.inf, "bad"])
def test_invalid_embedding_times_are_incomplete(bad_time):
    result = summarize_flux(pair("invalid", repo_time=bad_time), [0, 1])[0]
    assert result["repo_completeness"] == "incomplete"
    assert result["repo_evaluated_weight"] == 0
    assert result["repo_flux"] is None
    assert result["repo_incomplete_reasons"] == ["invalid_event_time"]


@pytest.mark.parametrize("status", [OutcomeStatus.CENSORED, OutcomeStatus.ALIGNED_ECCENTRIC])
@pytest.mark.parametrize("censor", [None, -1, np.nan, np.inf])
def test_missing_or_invalid_censor_horizon_cannot_certify_no_event(status, censor):
    result = summarize_flux(pair("unknown_horizon", repo_time=None,
                                 repo_status=status, censor=censor), [0, 1])[0]
    assert result["repo_completeness"] == "incomplete"
    assert result["repo_flux"] is None
    assert result["repo_incomplete_reasons"] == ["invalid_censor_time"]


def test_means_use_equal_realizations_and_all_input_weight_denominators():
    rows = pair("one", 2, realization=1)
    rows += pair("many", 8, realization=2)
    rows += pair("censored", 8, realization=2, repo_status="censored",
                 xue_status="censored", repo_time=None, xue_time=None)
    result = summarize_flux(rows, [0, 2])[0]
    assert result["n_realizations"] == 2
    assert result["total_weight"] == 18
    assert result["mean_total_weight"] == 9
    assert result["repo_mean_count"] == 5
    assert result["repo_flux"] == 2.5
    assert result["repo_cumulative_fraction"] == 0.75  # mean(2/2, 8/16), not 10/18
    assert [r["total_weight"] for r in result["per_realization"]] == [2, 16]
    assert [r["repo_count"] for r in result["per_realization"]] == [2, 8]


def test_bootstrap_matches_paired_resampling_and_repeats_without_mutation():
    rows = []
    for r, count in enumerate([1, 3, 5]):
        for source in range(10):
            rows += pair(source, realization=r, repo_time=0.5 if source < 2 * count else 2,
                         xue_time=0.5 if source < count else 2)
    original = deepcopy(rows)
    seed, draws = 44, 300
    result = summarize_flux(rows, [0, 1, 2], bootstrap_seed=seed, bootstrap_draws=draws)
    assert result == summarize_flux(list(reversed(rows)), [0, 1, 2],
                                    bootstrap_seed=seed, bootstrap_draws=draws)
    assert rows == original
    first = result[0]
    sampled = np.random.default_rng(seed).integers(3, size=(draws, 3))
    repo_means = np.array([2, 6, 10])[sampled].mean(axis=1)
    assert first["confidence_method"] == "paired_realization_bootstrap_mean_interval"
    assert first["repo_flux_ci95"] == pytest.approx(np.quantile(repo_means, [0.025, 0.975]))
    assert first["repo_cumulative_fraction_ci95"] == pytest.approx(
        np.quantile(repo_means / 10, [0.025, 0.975]))
    assert first["repo_effective_count"] == 18
    assert first["xue_effective_count"] == 9
    assert first["ratio_status"] == "defined"
    assert first["ratio"] == 2
    assert first["ratio_ci95"] == [2, 2]  # independent model resampling would spread this


def test_bootstrap_preserves_global_numpy_random_state():
    state = np.random.get_state()
    summarize_flux(pair("event"), [0, 1], bootstrap_seed=88)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_ratio_zero_and_low_effective_event_counts():
    zero = summarize_flux(pair("no_xue_event", xue_time=None, xue_status="censored"), [0, 1])[0]
    assert zero["repo_flux"] == 1
    assert zero["xue_flux"] == 0
    assert zero["ratio_status"] == "zero"
    assert zero["ratio"] is None
    assert zero["ratio_ci95"] is None
    rows = sum((pair(i, 100 if i == 0 else 1) for i in range(6)), [])
    low = summarize_flux(rows, [0, 1])[0]
    assert low["repo_effective_count"] == pytest.approx(105**2 / (100**2 + 5))
    assert low["ratio_status"] == "low_count"
    assert low["ratio"] is None
    assert low["ratio_ci95"] is None
    threshold = summarize_flux(sum((pair(i) for i in range(5)), []), [0, 1])[0]
    assert threshold["repo_effective_count"] == 5
    assert threshold["ratio_status"] == "defined"
    assert threshold["ratio"] == 1
    assert threshold["ratio_ci95"] == [1, 1]
    custom = summarize_flux(pair("one"), [0, 1], min_effective_count=1)[0]
    assert custom["ratio_status"] == "defined"


def test_effective_count_is_for_bin_events_not_censored_population():
    rows = pair("event")
    for i in range(20):
        rows += pair(i, repo_time=None, xue_time=None,
                     repo_status="censored", xue_status="censored")
    result = summarize_flux(rows, [0, 1])[0]
    assert result["total_weight"] == 21
    assert result["repo_effective_count"] == 1
    assert result["ratio_status"] == "low_count"


def test_ratio_interval_does_not_drop_zero_bootstrap_realizations():
    rows = sum((pair(i, realization=0) for i in range(5)), [])
    rows += pair("no_events", realization=1, repo_time=None, xue_time=None,
                 repo_status="censored", xue_status="censored")
    result = summarize_flux(rows, [0, 1], bootstrap_draws=100)[0]
    assert result["ratio_status"] == "defined"
    assert result["ratio"] == 1
    assert result["ratio_ci95"] is None
    assert result["ratio_ci_status"] == "undefined_zero_bootstrap_mean"


def test_zero_weight_unavailable_does_not_remove_positive_weight_coverage():
    rows = pair("valid", 2) + pair("zero", 0, repo_status="unavailable", xue_status="unavailable")
    result = summarize_flux(rows, [0, 1])[0]
    assert result["repo_completeness"] == "complete"
    assert result["repo_flux"] == 2
    assert result["repo_cumulative_fraction"] == 1
    assert result["repo_evaluated_weight"] == result["total_weight"] == 2
    assert result["repo_incomplete_reasons"] == []


def test_empty_and_zero_weight_population():
    assert summarize_flux([], [0, 1]) == []
    result = summarize_flux(pair("zero", 0), [0, 1])[0]
    assert result["total_weight"] == 0
    assert result["repo_completeness"] == "no_population"
    assert result["repo_flux"] is None
    assert result["repo_cumulative_fraction"] is None
    assert result["ratio_status"] == "incomplete"


@pytest.mark.parametrize("mutation, match", [
    (lambda rows: rows.append(dict(rows[0])), "duplicate"),
    (lambda rows: rows.pop(), "identical.*keys"),
    (lambda rows: rows[1].update(source_id="different"), "identical.*keys"),
    (lambda rows: rows[1].update(realization=9), "identical.*keys"),
    (lambda rows: rows[1].update(source_weight=2), "weight mismatch"),
    (lambda rows: rows[1].update(source_weight=1 + 1.e-12), "weight mismatch"),
    (lambda rows: rows[0].update(model_id="unknown"), "unknown model"),
    (lambda rows: rows[0].update(status="typo"), "unknown outcome"),
    (lambda rows: rows[0].pop("source_id"), "rows require"),
    (lambda rows: rows[0].update(source_id=[]), "hashable"),
    (lambda rows: rows[0].update(realization=None), "must not be None"),
])
def test_rejects_unmatched_or_malformed_rows(mutation, match):
    rows = pair("source")
    mutation(rows)
    with pytest.raises(ValueError, match=match):
        summarize_flux(rows, [0, 1])


@pytest.mark.parametrize("weight", [-1, np.inf, np.nan, None, "invalid"])
def test_rejects_invalid_weights(weight):
    with pytest.raises(ValueError, match="source_weight"):
        summarize_flux(pair("source", weight), [0, 1])


@pytest.mark.parametrize("edges", [[0], [0, 0, 1], [0, 2, 1], [-1, 1],
                                   [0, np.nan], [0, np.inf], [[0, 1]], ["bad", 1]])
def test_rejects_invalid_edges(edges):
    with pytest.raises(ValueError, match="bin edges"):
        summarize_flux([], edges)


@pytest.mark.parametrize("draws", [0, -1, 1.5, True])
def test_rejects_invalid_bootstrap_draws(draws):
    with pytest.raises(ValueError, match="bootstrap_draws"):
        summarize_flux([], [0, 1], bootstrap_draws=draws)


@pytest.mark.parametrize("threshold", [-1, np.nan, np.inf, None])
def test_rejects_invalid_effective_count_threshold(threshold):
    with pytest.raises(ValueError, match="min_effective_count"):
        summarize_flux([], [0, 1], min_effective_count=threshold)
