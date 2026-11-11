"""Single-system three-panel rate comparison, with no figure footnotes."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("host_rate_comparison", Path(__file__).resolve().parents[2] / "scripts/compare_host_rates.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_design_has_thirteen_points_and_one_controlled_scan_per_panel():
    points = comparison.unique_points()
    assert len(points) == 13
    assert min(point["z"] for point in points) == 8
    assert max(point["z"] for point in points) == 15
    assert min(point["M"] for point in points) == 1e4
    assert max(point["M"] for point in points) == 1e6
    assert min(point["M_NSC"] for point in points) == 1e6
    assert max(point["M_NSC"] for point in points) == 1e7
    assert len({comparison.point_key(point) for point in points}) == 13
    for name, panel in comparison.comparison_design().items():
        assert len(panel["curves"]) == 1
        curve = panel["curves"][0]
        scanned = comparison.points_for_curve(panel, curve)
        assert len(scanned) == 5
        for point in scanned:
            assert all(point[key] == value for key, value in curve.items())


def _summary_fixture(path):
    pd.DataFrame(columns=["repeat", "event_rate_class", "event_rate_reason"]).to_csv(path / "imri_events.csv", index=False)
    pd.DataFrame({
        "repeat": [0, 1], "n_emb_star": [2, 1], "n_emb_bh": [1, 0],
        "duty_cycle_myr": [10, 10], "n_stellar_unavailable": [0, 0],
        "rate_imri_per_myr": [0.1, 0.0], "rate_det_per_myr": [0.0, 0.0],
        "rate_status": ["complete", "complete"],
        "detector_rate_status": ["complete", "complete"],
    }).to_csv(path / "imri_repeat_results.csv", index=False)
    pd.DataFrame({
        "repeat": [0, 0, 0, 0, 0, 1],
        "stage": ["damping", "damping", "damping", "entry_detection", "damping", "damping"],
        "status": ["embedded_star", "embedded_star", "embedded_bh", "embedded_bh", "damping_incomplete", "embedded_star"],
        "task_kind": ["primary", "replacement", "primary", "primary", "primary", "primary"],
    }).to_csv(path / "imri_source_transitions.csv", index=False)


def test_damping_rate_counts_successes_not_attempts_and_keeps_zero_detects():
    with TemporaryDirectory() as directory:
        path = Path(directory)
        _summary_fixture(path)
        result, repeats = comparison.summarize_run(path)
        assert result["damping_rate"] == pytest.approx(0.2)
        assert result["damping_sem"] == pytest.approx(0.1)
        assert result["primary_damping_rate"] == pytest.approx(0.15)
        assert result["event_rate"] == pytest.approx(0.05)
        assert result["detect_rate"] == 0
        assert result["repeats"] == 2
        assert repeats.rate_damping_per_myr.tolist() == [0.3, 0.1]


def test_numerical_damping_failure_is_not_zero():
    with TemporaryDirectory() as directory:
        path = Path(directory)
        _summary_fixture(path)
        transitions = pd.read_csv(path / "imri_source_transitions.csv")
        transitions.loc[transitions.status == "damping_incomplete", "status"] = "calculation_error"
        transitions.to_csv(path / "imri_source_transitions.csv", index=False)
        result, _ = comparison.summarize_run(path)
        assert np.isnan(result["damping_rate"])
        assert result["detect_rate"] == 0


def test_unevaluated_events_override_misleading_complete_aggregate():
    with TemporaryDirectory() as directory:
        path = Path(directory)
        _summary_fixture(path)
        pd.DataFrame({"repeat": [0], "event_rate_class": ["not evaluated"],
                      "event_rate_reason": ["isco_before_entry"]}).to_csv(path / "imri_events.csv", index=False)
        result, repeats = comparison.summarize_run(path)
        assert np.isnan(result["event_rate"])
        assert np.isnan(result["detect_rate"])
        assert result["damping_rate"] == pytest.approx(0.2)
        assert result["event_status"] == "incomplete"
        assert result["event_unavailable_reasons"] == "isco_before_entry"
        assert (repeats.core_rate_status == "complete").all()


def test_clean_figure_has_only_three_panels_one_legend_and_shared_limits():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for index, point in enumerate(comparison.unique_points()):
        row = dict(point, repeats=3)
        for rate, value in (("damping", 2.0), ("event", 0.2), ("detect", 0.0)):
            row[f"{rate}_rate"] = value
            row[f"{rate}_sem"] = 0.0 if rate == "detect" else 0.05
        if point["M"] == 1e6:
            for rate in ("event", "detect"):
                row[f"{rate}_rate"] = np.nan
                row[f"{rate}_sem"] = np.nan
        rows.append(row)
    frame = pd.DataFrame(rows)
    closed = []
    original_close = plt.close
    def check_figure(fig=None):
        if hasattr(fig, "axes"):
            assert len(fig.axes) == 3
            assert len(fig.legends) == 1
            assert len(fig.texts) == 0
            assert all(ax.get_ylim() == fig.axes[0].get_ylim() for ax in fig.axes)
            assert "host" not in fig.axes[0].get_ylabel()
            assert [text.get_text() for text in fig.legends[0].get_texts()] == ["Damping", "IMRI", "Detectable"]
            closed.append(fig)
        original_close(fig)
    with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
        path = Path(directory)
        patch.setattr(plt, "close", check_figure)
        limits = comparison.plot_comparisons(frame, path)
        assert limits[0] == 0
        assert len(closed) == 1
        for suffix in ("png", "pdf", "svg"):
            assert (path / f"rate_dependence.{suffix}").stat().st_size > 1000
        manifest = {"seed": 1, "repeats": 3, "reference": {"R_NSC": 1.3, "gamma_n": 1.75, "alpha": 0.1, "Mdot": 1.44e23, "r_out": 1e16}}
        comparison.write_ppt_notes(frame, path, manifest)
        notes = (path / "ppt_notes.md").read_text()
        assert "no host-number or volume normalization" in notes
        assert "unavailable" in notes
        assert "uncertain scenario range" in notes


def test_plot_rejects_missing_measured_points():
    frame = pd.DataFrame(comparison.unique_points()).iloc[1:]
    with pytest.raises(ValueError, match="Expected one measured point"):
        comparison.select_panel(frame, comparison.comparison_design()["redshift"])
