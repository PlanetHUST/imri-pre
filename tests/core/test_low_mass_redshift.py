"""The second host benchmark reports measured SNRs, including empty samples."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def experiment(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("low_mass_redshift", scripts / "compare_low_mass_redshift.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_snr_and_delay_accounting(tmp_path, experiment):
    pd.DataFrame([
        dict(is_imri_candidate=True, snr_status="evaluated", lisa_snr=25.,
             is_lisa_detectable=True, band_entry_time_myr=11., mass_msun=100.),
        dict(is_imri_candidate=True, snr_status="evaluated", lisa_snr=5.,
             is_lisa_detectable=False, band_entry_time_myr=2., mass_msun=50.),
        dict(is_imri_candidate=False, snr_status="not_evaluated", lisa_snr=np.nan,
             is_lisa_detectable=False, band_entry_time_myr=1., mass_msun=20.),
    ]).to_csv(tmp_path / "imri_events.csv", index=False)
    summary, events = experiment.snr_records(tmp_path, 10., 1e4)
    assert summary["candidates"] == summary["evaluated"] == 2
    assert summary["detections"] == summary["entries_after_agn"] == summary["detections_after_agn"] == 1
    assert summary["snr_median"] == 15.
    assert summary["snr_max"] == 25.
    assert len(events) == 2
    assert (events.initial_z == 10).all()
    assert (events.M == 1e4).all()


def test_no_candidates_have_missing_snr_not_zero(tmp_path, experiment):
    pd.DataFrame([dict(is_imri_candidate=False, snr_status="not_evaluated", lisa_snr=np.nan,
                      is_lisa_detectable=False, band_entry_time_myr=1.)]).to_csv(
                          tmp_path / "imri_events.csv", index=False)
    summary, events = experiment.snr_records(tmp_path, 8., 1e4)
    assert summary["candidates"] == summary["evaluated"] == summary["detections"] == 0
    assert np.isnan(summary["snr_max"])
    assert events.empty


def test_high_mass_unavailable_snr_is_not_a_zero_detection_claim(tmp_path, experiment):
    pd.DataFrame([dict(is_imri_candidate=False, snr_status="not_evaluated", lisa_snr=np.nan,
                      event_rate_class="not_evaluated", event_rate_reason="isco_before_entry",
                      is_lisa_detectable=False, band_entry_time_myr=np.nan)]).to_csv(
                          tmp_path / "imri_events.csv", index=False)
    summary, events = experiment.snr_records(tmp_path, 10., 1e6)
    assert summary["unavailable_events"] == 1
    assert summary["evaluated"] == 0
    assert np.isnan(summary["snr_median"])
    assert np.isnan(summary["snr_max"])


def test_scatter_uses_all_evaluated_candidates_and_actual_redshift(tmp_path, experiment):
    frame = pd.DataFrame([
        dict(M=1e4, mass_msun=50., snr_observation_redshift=9.9, initial_z=10.,
             lisa_snr=5., snr_status="evaluated", repeat=0, source_id=1),
        dict(M=1e5, mass_msun=100., snr_observation_redshift=7.9, initial_z=8.,
             lisa_snr=25., snr_status="evaluated", repeat=0, source_id=2),
        dict(M=1e6, mass_msun=100., snr_observation_redshift=7.9, initial_z=8.,
             lisa_snr=25., snr_status="evaluated", repeat=0, source_id=3),
        dict(M=1e4, mass_msun=50., snr_observation_redshift=9.9, initial_z=10.,
             lisa_snr=np.nan, snr_status="not_evaluated", repeat=0, source_id=4),
    ])
    selected = experiment.scatter_population(frame)
    assert selected.source_id.tolist() == [1, 2]
    assert selected.snr_observation_redshift.tolist() == [9.9, 7.9]
    frame.to_csv(tmp_path / "candidate_snr.csv", index=False)
    plotted = experiment.plot_candidate_properties(tmp_path)
    pd.testing.assert_frame_equal(plotted.reset_index(drop=True), selected.reset_index(drop=True))
    for suffix in ("png", "pdf", "svg"):
        assert (tmp_path / f"snr_population_scatter.{suffix}").stat().st_size > 0
    summary = json.loads((tmp_path / "snr_population_scatter_summary.json").read_text())
    assert [item["above_snr_20"] for item in summary["benchmarks"]] == [0, 1]


def test_scatter_rejects_missing_columns(experiment):
    with pytest.raises(ValueError, match="Missing scatter columns"):
        experiment.scatter_population(pd.DataFrame({"M": [1e4]}))


def test_three_mass_plot_and_report_preserve_unavailable_rates(tmp_path, experiment):
    rates, snrs = [], []
    for mass in (1e4, 1e5, 1e6):
        for z in experiment.REDSHIFTS:
            unavailable = mass == 1e6
            row = dict(M=mass, z=z, event_status="incomplete" if unavailable else "complete")
            for rate in ("damping", "event", "detect"):
                row[rate + "_rate"] = np.nan if unavailable and rate != "damping" else 1.
                row[rate + "_sem"] = np.nan if unavailable and rate != "damping" else .1
            rates.append(row)
            value = np.nan if unavailable else 10.
            snrs.append(dict(M=mass, z=z, snr_median=value, snr_max=value,
                             snr_p10=value, snr_p90=value, entries_after_agn=0,
                             detections_after_agn=0))
    pd.DataFrame(rates).to_csv(tmp_path / "host_rates.csv", index=False)
    pd.DataFrame(snrs).to_csv(tmp_path / "snr_summary.csv", index=False)
    (tmp_path / "experiment.json").write_text(json.dumps(dict(central_mass_msun=1e6,
        new_disk=dict(Mdot_g_s=1.44e24, outer_radius_cm=1e16))))
    experiment.plot(tmp_path)
    experiment.write_report(tmp_path)
    for name in ("rate_dependence", "snr_vs_redshift"):
        for suffix in ("png", "pdf", "svg"):
            assert (tmp_path / f"{name}.{suffix}").stat().st_size > 0
    report = (tmp_path / "report.md").read_text()
    assert "| 1 ± 0.1 | N/A | N/A | N/A | N/A | N/A |" in report
    assert "isco_before_entry" in report
