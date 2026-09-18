"""Interface validation checkpoints A-H for the merger-tree batch layer.

Each checkpoint isolates one interface, uses a small controlled input, writes
compact machine-readable results under ``output/checkpoints/<checkpoint>/`` and
reports PASS / WARN / FAIL.  None of them audits the full merger tree and none
computes a cosmic rate.

    python scripts/interface_checkpoints.py            # run everything
    python scripts/interface_checkpoints.py --only B C  # run a subset

Checkpoints
-----------
A  original single-halo baseline (the unmodified ``scripts/main.py`` path)
B  merger-tree reader: columns, parsing, identity, filtering, chunk invariance
C  host-to-run mapping: raw tree values versus resolved local inputs
D  original CLI versus refactored ``run_single_halo`` equivalence
E  manual single-halo run versus tree-derived run equivalence
F  serial versus parallel batch reproducibility
G  batch failure isolation and status semantics
H  production reader + executor on a subset of the real merger tree (CLI path)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from workflows.cli import build_config_from_arguments, build_parser, run_cli  # noqa: E402
from workflows.config import (  # noqa: E402
    IMRI_AVERAGED_RESULTS_FILENAME,
    IMRI_EVENTS_FILENAME,
    IMRI_REPEAT_RESULTS_FILENAME,
)
from workflows.host_mapping import (  # noqa: E402
    HostMappingError,
    HostMappingPolicy,
    resolve_host_parameters,
)
from workflows.merger_tree import (  # noqa: E402
    HaloHost,
    HostSelection,
    MergerTreeError,
    MergerTreeReader,
    REQUIRED_TREE_COLUMNS,
    hosts_to_frame,
)
from workflows.single_halo import (  # noqa: E402
    STATUS_OUTPUT_FAILURE,
    STATUS_SUCCESS,
    run_single_halo,
)
from workflows.tree_batch import (  # noqa: E402
    BatchSettings,
    HaloSimulationJob,
    build_job,
    derive_host_seed,
    run_jobs,
    run_tree_batch,
)
from physical_process.sampling import SOLAR_METALLICITY  # noqa: E402

TREE_PATH = PROJECT_ROOT / "galaxy_1e12.csv"
CHECKPOINT_ROOT = PROJECT_ROOT / "output" / "checkpoints"
MAIN_SCRIPT = PROJECT_ROOT / "scripts" / "main.py"
PYTHON = sys.executable

# Controlled single-halo configuration shared by A, D and E (fast Pop III host).
BASE_SMBH_MSUN = 1.0e5
BASE_NSC_MSUN = 5.0e4
BASE_REPEATS = 2
BASE_SEED = 777
BASE_DAMPING_CPUS = 2
COMPARED_FILES = (
    IMRI_AVERAGED_RESULTS_FILENAME,
    IMRI_REPEAT_RESULTS_FILENAME,
    IMRI_EVENTS_FILENAME,
)
KEY_RESULT_COLUMNS = (
    "n_sampled",
    "n_disk_cross",
    "n_emb_star",
    "n_emb_bh",
    "n_mfbh",
    "n_bh_candi",
    "n_imri",
    "rate_imri_per_myr",
    "n_det",
    "rate_det_per_myr",
    "n_mig_error",
    "n_profile_error",
    "task_seed",
)

SCALING_POLICY = HostMappingPolicy(
    smbh_mass_source="stellar-scaling",
    nsc_mass_source="stellar-scaling",
)


class Report:
    """Collect pass/warn/fail conditions for one checkpoint."""

    def __init__(self, name: str, directory: Path) -> None:
        self.name = name
        self.directory = directory
        self.checks: list[dict[str, Any]] = []
        self.data: dict[str, Any] = {}
        directory.mkdir(parents=True, exist_ok=True)

    def check(self, label: str, passed: bool, detail: Any = None, warn: bool = False) -> bool:
        level = "PASS" if passed else ("WARN" if warn else "FAIL")
        self.checks.append({"check": label, "result": level, "detail": detail})
        print(f"  [{level}] {label}" + (f" -- {detail}" if detail is not None and not passed else ""))
        return passed

    @property
    def status(self) -> str:
        if any(item["result"] == "FAIL" for item in self.checks):
            return "FAIL"
        if any(item["result"] == "WARN" for item in self.checks):
            return "WARN"
        return "PASS"

    def save(self, filename: str = "checkpoint_summary.json") -> Path:
        payload = {
            "checkpoint": self.name,
            "status": self.status,
            "checks": self.checks,
            "data": _json_safe(self.data),
        }
        path = self.directory / filename
        with path.open("w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"  => {self.name}: {self.status} ({path})")
        return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fresh(directory: Path) -> Path:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files_identical(left: Path, right: Path, names=COMPARED_FILES) -> dict[str, bool]:
    return {name: _sha256(left / name) == _sha256(right / name) for name in names}


def _tables_equal(left: Path, right: Path, names=COMPARED_FILES) -> dict[str, bool]:
    """Value-level comparison tolerant to integer-versus-float text formatting."""
    outcome = {}
    for name in names:
        first = pd.read_csv(left / name)
        second = pd.read_csv(right / name)
        if first.shape != second.shape or list(first.columns) != list(second.columns):
            outcome[name] = False
            continue
        same = True
        for column in first.columns:
            a, b = first[column], second[column]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                same &= bool(np.array_equal(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True))
            else:
                same &= bool(a.fillna("<na>").astype(str).equals(b.fillna("<na>").astype(str)))
        outcome[name] = same
    return outcome


def _key_quantities(run_dir: Path) -> dict[str, Any]:
    repeats = pd.read_csv(run_dir / IMRI_REPEAT_RESULTS_FILENAME)
    averaged = pd.read_csv(run_dir / IMRI_AVERAGED_RESULTS_FILENAME)
    events = pd.read_csv(run_dir / IMRI_EVENTS_FILENAME)
    return {
        "repeat_rows": {
            column: repeats[column].tolist() for column in KEY_RESULT_COLUMNS
        },
        "averaged_n_imri": float(averaged["n_imri"].iloc[0]),
        "averaged_rate_imri_per_myr": float(averaged["rate_imri_per_myr"].iloc[0]),
        "averaged_rate_status": str(averaged["rate_status"].iloc[0]),
        "event_rows": int(len(events.index)),
        "event_classes": events["event_rate_class"].value_counts().to_dict()
        if len(events.index)
        else {},
    }


def _base_argv(output_dir: Path, **overrides: Any) -> list[str]:
    values = {
        "smbh": BASE_SMBH_MSUN,
        "nsc": BASE_NSC_MSUN,
        "seed": BASE_SEED,
        "redshift": None,
        "metallicity": None,
    }
    values.update(overrides)
    argv = [
        str(BASE_REPEATS),
        str(output_dir),
        str(BASE_DAMPING_CPUS),
        "--smbh-masses",
        repr(float(values["smbh"])),
        "--nsc-masses",
        repr(float(values["nsc"])),
        "--base-seed",
        str(values["seed"]),
    ]
    if values["redshift"] is not None:
        argv += ["--redshift", repr(float(values["redshift"]))]
    if values["metallicity"] is not None:
        argv += ["--metallicity", repr(float(values["metallicity"]))]
    return argv


def _run_original_cli(argv: list[str]) -> float:
    """Run the unmodified command-line entry point as a subprocess."""
    started = time.perf_counter()
    completed = subprocess.run(
        [PYTHON, str(MAIN_SCRIPT), *argv],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"original CLI failed:\n{completed.stderr[-2000:]}")
    return time.perf_counter() - started


# --------------------------------------------------------------------------- A
def checkpoint_a() -> Report:
    report = Report("A_original_baseline", CHECKPOINT_ROOT / "A_original_baseline")
    run_dir = _fresh(report.directory / "original")
    runtime = _run_original_cli(_base_argv(run_dir))
    quantities = _key_quantities(run_dir)
    report.data.update(
        {
            "argv": _base_argv(run_dir),
            "runtime_s": runtime,
            "digests": {name: _sha256(run_dir / name) for name in COMPARED_FILES},
            "key_quantities": quantities,
        }
    )
    report.check("original CLI produced all result files", all((run_dir / n).is_file() for n in COMPARED_FILES))
    report.check(
        "baseline repeat count matches configuration",
        len(quantities["repeat_rows"]["task_seed"]) == BASE_REPEATS,
    )
    report.check(
        "baseline task seeds are recorded",
        all(seed is not None for seed in quantities["repeat_rows"]["task_seed"]),
    )
    report.save()
    return report


# --------------------------------------------------------------------------- B
FIXTURE_HEADER = (
    ",plc,z,m,inode,br,br_b4,fc,sf,bh,brf,Mhot,Mcold,Mstar,Mout,mZ,Z,dM,Q,mlist,dt,gen,"
    "MSF,MBH,MBH2,Mgas,Mgas2,Mtde,Mtde2,Macc2"
)


def _fixture_row(index: int, z: str, m: str, inode: str, br: str, mstar: str, metal: str, mbh: str, junk: str = "0") -> str:
    # Unused columns get deliberately non-numeric junk to prove they are not parsed.
    return ",".join(
        [
            str(index), "-1", z, m, inode, br, "1", "0", "-1", "-1", "-1",
            junk, junk, mstar, junk, junk, metal, "9", "0", "336631", "747",
            junk, "6740", mbh, "0", "0", "0", "0", "0", "",
        ]
    )


def write_reader_fixture(path: Path) -> Path:
    rows = [
        FIXTURE_HEADER,
        _fixture_row(1, "6.0", "1e12", "1", "1", "3.0e9", "2.0e-4", "0.0", junk="not-a-number"),
        _fixture_row(2, "6.5", "9e11", "1", "1", "2.0e9", "1.0e-4", "1.0e5", junk="x"),
        _fixture_row(3, "nan", "9e11", "2", "1", "2.0e9", "1.0e-4", "1.0e5"),      # missing z
        _fixture_row(4, "-0.5", "9e11", "2", "1", "2.0e9", "1.0e-4", "1.0e5"),     # invalid z
        _fixture_row(5, "7.0", "8e11", "2", "1", "garbage", "1.0e-4", "2.0e5"),   # bad Mstar
        _fixture_row(6, "7.5", "8e11", "2", "1", "1.5e9", "", "-3.0"),             # negative MBH, empty Z
        _fixture_row(7, "8.0", "5e11", "3", "2", "1.0e9", "5.0e-5", "5.0e4"),
        _fixture_row(8, "8.5", "4e11", "3", "2", "0.8e9", "5.0e-5", "3.0e4"),
        _fixture_row(9, "9.0", "3e11", "4", "2", "0.5e9", "2.0e-5", "1.0e4"),
    ]
    path.write_text("\n".join(rows) + "\n")
    return path


def checkpoint_b() -> Report:
    report = Report("B_tree_reader", CHECKPOINT_ROOT / "B_tree_reader")
    fixture = write_reader_fixture(report.directory / "fixture_tree.csv")

    # Required-column validation.
    bad = report.directory / "fixture_missing_column.csv"
    bad.write_text("z,m,inode\n6.0,1e12,1\n")
    try:
        MergerTreeReader(bad)
        report.check("missing required column is rejected", False)
    except MergerTreeError as error:
        report.check("missing required column is rejected", "MBH" in str(error), str(error))
    try:
        MergerTreeReader(report.directory / "does_not_exist.csv")
        report.check("missing file is rejected", False)
    except MergerTreeError:
        report.check("missing file is rejected", True)

    # Parsing, identity and counters on the synthetic fixture.
    reader = MergerTreeReader(fixture, chunk_size=4)
    hosts = list(reader.iter_hosts())
    counters = reader.counters.to_dict()
    report.data["fixture_counters"] = counters
    report.data["fixture_hosts"] = [host.to_dict() for host in hosts]
    report.check("unused junk columns do not break parsing", counters["rows_seen"] == 9, counters)
    report.check("NaN redshift counted as missing_required_field", counters["missing_required_field"] == 1, counters)
    report.check("negative redshift counted as invalid_redshift", counters["invalid_redshift"] == 1, counters)
    report.check("negative MBH counted as invalid_bh_mass", counters["invalid_bh_mass"] == 1, counters)
    report.check("valid rows become hosts", counters["rows_selected"] == 6 and len(hosts) == 6, len(hosts))
    by_id = {host.host_id: host for host in hosts}
    report.check(
        "host identities are unique and deterministic",
        len(by_id) == len(hosts) and hosts[0].host_id == "n000001_b000001_r0000000",
        [host.host_id for host in hosts],
    )
    garbage = by_id.get("n000002_b000001_r0000004")
    report.check(
        "unparseable Mstar becomes None rather than crashing",
        garbage is not None and garbage.stellar_mass_msun is None,
    )
    report.check(
        "numeric fields parse to floats",
        isinstance(hosts[0].redshift, float) and hosts[0].halo_mass_msun == 1e12 and hosts[0].metallicity_mass_fraction == 2.0e-4,
    )

    # Filtering.
    filtered = list(MergerTreeReader(fixture, HostSelection(nodes=(3.0,))).iter_hosts())
    report.check("node filter", [h.record_index for h in filtered] == [6, 7])
    filtered = list(MergerTreeReader(fixture, HostSelection(branches=(2.0,), redshift_max=8.6)).iter_hosts())
    report.check("branch + redshift filter", [h.record_index for h in filtered] == [6, 7])
    filtered = list(MergerTreeReader(fixture, HostSelection(smbh_mass_min_msun=4e4)).iter_hosts())
    report.check("SMBH-mass filter", [h.record_index for h in filtered] == [1, 4, 6], [h.record_index for h in filtered])
    filtered = list(MergerTreeReader(fixture, HostSelection(halo_mass_max_msun=5e11)).iter_hosts())
    report.check("halo-mass filter", [h.record_index for h in filtered] == [6, 7, 8])
    filtered = list(MergerTreeReader(fixture, HostSelection(max_hosts=2)).iter_hosts())
    report.check("max_hosts truncation", [h.record_index for h in filtered] == [0, 1])

    # Deterministic repeated reading and chunk-size invariance (fixture + real file).
    def signature(path: Path, selection: HostSelection, chunk_size: int) -> str:
        frame = hosts_to_frame(list(MergerTreeReader(path, selection, chunk_size).iter_hosts()))
        return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()

    fixture_signatures = {size: signature(fixture, HostSelection(), size) for size in (1, 2, 3, 4, 7, 1000)}
    report.check("fixture: chunk-size invariance", len(set(fixture_signatures.values())) == 1, fixture_signatures)
    report.check("fixture: repeated reading is deterministic", signature(fixture, HostSelection(), 3) == fixture_signatures[3])

    real_selection = HostSelection(nodes=(1.0, 100.0, 320.0), max_hosts=40)
    started = time.perf_counter()
    real_signatures = {size: signature(TREE_PATH, real_selection, size) for size in (50, 333, 1000, 50_000)}
    report.data["real_read_time_s_total"] = time.perf_counter() - started
    report.check("real tree: chunk-size invariance", len(set(real_signatures.values())) == 1, real_signatures)
    real_reader = MergerTreeReader(TREE_PATH, real_selection, chunk_size=1000)
    real_hosts = list(real_reader.iter_hosts())
    hosts_to_frame(real_hosts).to_csv(report.directory / "selected_hosts.csv", index=False)
    report.data["real_counters"] = real_reader.counters.to_dict()
    report.check("real tree: only required columns requested", list(REQUIRED_TREE_COLUMNS) == ["z", "m", "Mstar", "Z", "MBH", "inode", "br"])
    report.check("real tree: MBH column is identically zero in selected rows (documented limitation)", all(h.smbh_mass_msun == 0.0 for h in real_hosts), warn=True)

    # Memory boundedness: the streaming pass over the full file never holds more than one chunk.
    full = MergerTreeReader(TREE_PATH, HostSelection(), chunk_size=500)
    started = time.perf_counter()
    count = sum(1 for _ in full.iter_hosts())
    report.data["full_stream_rows"] = count
    report.data["full_stream_time_s"] = time.perf_counter() - started
    report.check("full streaming pass sees every row once", count == full.counters.rows_seen == 9999, count)

    with (report.directory / "reader_summary.json").open("w") as handle:
        json.dump(_json_safe({"fixture": counters, "real": real_reader.counters.to_dict(), "chunk_signatures": {"fixture": fixture_signatures, "real": real_signatures}}), handle, indent=2)
    report.save()
    return report


# --------------------------------------------------------------------------- C
def checkpoint_c() -> Report:
    report = Report("C_host_mapping", CHECKPOINT_ROOT / "C_host_mapping")
    parser = build_parser()
    base = build_config_from_arguments(parser.parse_args(["1", str(report.directory / "unused"), "1", "--smbh-masses", "1e5", "--nsc-masses", "2e5"]))
    hosts = list(MergerTreeReader(TREE_PATH, HostSelection(nodes=(1.0, 100.0, 320.0), max_hosts=6)).iter_hosts())
    policies = {
        "tree_default": HostMappingPolicy(),
        "config_masses": HostMappingPolicy(smbh_mass_source="config", nsc_mass_source="config"),
        "stellar_scaling": SCALING_POLICY,
        "stellar_scaling_tree_metallicity": HostMappingPolicy(smbh_mass_source="stellar-scaling", nsc_mass_source="stellar-scaling", metallicity_source="tree"),
        "narrow_domain": HostMappingPolicy(smbh_mass_source="stellar-scaling", domain_smbh_mass_max_msun=1.0e5),
    }
    rows = []
    for host in hosts:
        for name, policy in policies.items():
            row = {"policy": name, **{f"raw_{k}": v for k, v in host.to_dict().items()}}
            try:
                resolved = resolve_host_parameters(host, base, policy)
                row.update({f"resolved_{k}": v for k, v in resolved.to_dict().items() if k != "host_id"})
                row["mapping_status"] = "resolved"
            except HostMappingError as error:
                row["mapping_status"] = error.status
                row["mapping_error"] = error.message
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(report.directory / "mapping_table.csv", index=False)

    def rows_for(name: str) -> pd.DataFrame:
        return table[table["policy"] == name]

    report.check("tree MBH source rejects the empty MBH column as invalid_host_input", (rows_for("tree_default")["mapping_status"] == "invalid_host_input").all())
    cfg = rows_for("config_masses")
    report.check("config source: redshift comes from tree z", np.allclose(cfg["resolved_redshift"], cfg["raw_redshift"]) and (cfg["resolved_redshift_source"] == "tree:z").all())
    report.check("config source: masses are host independent", (cfg["resolved_smbh_mass_msun"] == 1e5).all() and (cfg["resolved_nsc_mass_msun"] == 2e5).all() and (cfg["resolved_smbh_mass_msun_source"] == "config").all())
    sca = rows_for("stellar_scaling")
    report.check("stellar scaling: M_smbh = f * Mstar", np.allclose(sca["resolved_smbh_mass_msun"], SCALING_POLICY.smbh_stellar_mass_fraction * sca["raw_stellar_mass_msun"]))
    report.check("stellar scaling: M_nsc = f * Mstar", np.allclose(sca["resolved_nsc_mass_msun"], SCALING_POLICY.nsc_stellar_mass_fraction * sca["raw_stellar_mass_msun"]))
    report.check("stellar scaling: provenance labelled as derived", sca["resolved_smbh_mass_msun_source"].str.startswith("derived:").all())
    report.check("stellar scaling: metallicity stays config (1e-5 Zsun)", (sca["resolved_metallicity_zsun"] == base.metallicity).all())
    met = rows_for("stellar_scaling_tree_metallicity")
    report.check("tree metallicity: Z_zsun = Z_massfraction / 0.0145", np.allclose(met["resolved_metallicity_zsun"], met["raw_metallicity_mass_fraction"] / SOLAR_METALLICITY) and (met["resolved_metallicity_zsun_source"] == "tree:Z/Zsun").all())
    report.check("tree metallicity: selects the Pop II regime (>= 1e-4 Zsun)", (met["resolved_metallicity_zsun"] >= 1e-4).all(), float(met["resolved_metallicity_zsun"].iloc[0]))
    report.check("domain policy: hosts above the SMBH bound are outside_local_model_domain", (rows_for("narrow_domain")["mapping_status"] == "outside_local_model_domain").all())

    # Explicit synthetic edge cases.
    edge = HaloHost(host_id="edge", redshift=6.0, record_index=0, stellar_mass_msun=None)
    try:
        resolve_host_parameters(edge, base, SCALING_POLICY)
        report.check("missing Mstar -> missing_required_input", False)
    except HostMappingError as error:
        report.check("missing Mstar -> missing_required_input", error.status == "missing_required_input")
    edge = HaloHost(host_id="edge", redshift=6.0, record_index=0, smbh_mass_msun=5e4)
    resolved = resolve_host_parameters(edge, base, HostMappingPolicy())
    report.check("tree MBH used verbatim when positive", resolved.smbh_mass_msun == 5e4 and resolved.sources["smbh_mass_msun"] == "tree:MBH")
    multi = build_config_from_arguments(parser.parse_args(["1", str(report.directory / "unused"), "1", "--smbh-masses", "1e4,1e5"]))
    try:
        resolve_host_parameters(edge, multi, HostMappingPolicy(smbh_mass_source="config"))
        report.check("config source with a mass vector -> mapping_unavailable", False)
    except HostMappingError as error:
        report.check("config source with a mass vector -> mapping_unavailable", error.status == "mapping_unavailable")

    _plot_mapping(table, report.directory / "figure2_tree_to_local_mapping.png")
    report.data["hosts"] = [host.to_dict() for host in hosts]
    report.data["policies"] = {name: policy.to_dict() for name, policy in policies.items()}
    report.save()
    return report


def _plot_mapping(table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    sca = table[table["policy"] == "stellar_scaling"].sort_values("raw_stellar_mass_msun")
    met = table[table["policy"] == "stellar_scaling_tree_metallicity"].sort_values("raw_metallicity_mass_fraction")
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=150)
    mstar = sca["raw_stellar_mass_msun"].to_numpy(dtype=float)
    axes[0].plot(mstar, sca["resolved_smbh_mass_msun"], color="#2a78d6", linewidth=1.5, label="M_smbh = 1e-4 x Mstar (derived)")
    axes[0].scatter(mstar, sca["resolved_smbh_mass_msun"], s=70, facecolor="none", edgecolor="#2a78d6", linewidth=1.5)
    axes[0].scatter(mstar, sca["resolved_nsc_mass_msun"], s=22, color="#eb6834", marker="s", label="M_nsc = 1e-4 x Mstar (derived)")
    axes[0].set_xlabel("tree Mstar [Msun]")
    axes[0].set_ylabel("resolved local mass [Msun]")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    metal = met["raw_metallicity_mass_fraction"].to_numpy(dtype=float)
    axes[1].plot(metal, met["resolved_metallicity_zsun"], color="#1baf7a", linewidth=1.5, label="Z_zsun = Z / 0.0145 (tree)")
    axes[1].scatter(metal, met["resolved_metallicity_zsun"], s=30, color="#1baf7a")
    axes[1].axhline(1e-4, color="#9a9891", linewidth=1, linestyle="--")
    axes[1].axhline(1e-5, color="#eb6834", linewidth=1, linestyle=":")
    axes[1].text(metal.min(), 1.25e-4, "Pop III threshold (1e-4 Zsun)", fontsize=7, color="#52514e")
    axes[1].text(metal.min(), 1.25e-5, "config default (1e-5 Zsun)", fontsize=7, color="#eb6834")
    axes[1].set_yscale("log")
    axes[1].set_ylim(3e-6, 5e-2)
    axes[1].set_xlabel("tree Z (metal mass fraction)")
    axes[1].set_ylabel("resolved metallicity [Zsun]")
    axes[1].legend(frameon=False, fontsize=8, loc="center right")
    for axis in axes:
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.ticklabel_format(axis="x", useOffset=False, style="sci", scilimits=(0, 0))
        axis.tick_params(axis="x", labelsize=8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(True, color="#e4e2dc", linewidth=0.6)
    axes[0].yaxis.set_major_locator(MaxNLocator(5))
    axes[0].ticklabel_format(axis="y", useOffset=False, style="sci", scilimits=(0, 0))
    figure.suptitle("Figure 2: tree host fields -> resolved local simulation inputs", fontsize=10, x=0.02, ha="left")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


# --------------------------------------------------------------------------- D
def checkpoint_d() -> Report:
    report = Report("D_single_halo_equivalence", CHECKPOINT_ROOT / "D_single_halo_equivalence")
    original_dir = CHECKPOINT_ROOT / "A_original_baseline" / "original"
    if not (original_dir / IMRI_REPEAT_RESULTS_FILENAME).is_file():
        _fresh(original_dir)
        _run_original_cli(_base_argv(original_dir))
    parser = build_parser()
    base = build_config_from_arguments(parser.parse_args(_base_argv(report.directory / "unused")))

    # (1) Programmatic call with no host override: same config, explicit seed/output.
    plain_dir = _fresh(report.directory / "refactored_plain")
    result = run_single_halo(base, host=None, seed=BASE_SEED, output_dir=plain_dir)
    report.check("run_single_halo succeeded without a host", result.status == STATUS_SUCCESS, result.error)
    identical = _files_identical(original_dir, plain_dir)
    report.check("refactored (no host) == original CLI outputs, byte for byte", all(identical.values()), identical)

    # (2) Same physics reached through an explicit resolved host record.
    host = HaloHost(host_id="manual_match", redshift=base.redshift, record_index=0, smbh_mass_msun=BASE_SMBH_MSUN)
    policy = HostMappingPolicy(smbh_mass_source="tree", nsc_mass_source="config", metallicity_source="config")
    host_dir = _fresh(report.directory / "refactored_host")
    result_host = run_single_halo(base, host, seed=BASE_SEED, output_dir=host_dir, policy=policy)
    report.check("run_single_halo succeeded with an explicit host", result_host.status == STATUS_SUCCESS, result_host.error)
    identical_host = _files_identical(original_dir, host_dir)
    equal_host = _tables_equal(original_dir, host_dir)
    report.check("refactored (explicit host) == original CLI outputs, value for value", all(equal_host.values()), equal_host)
    report.check(
        "refactored (explicit host) byte-identical to original (only int/float text formatting of z may differ)",
        all(identical_host.values()),
        identical_host,
        warn=True,
    )
    report.check("structured summary reports the kernel counters", result_host.summary.get("n_repeat") == BASE_REPEATS and "n_imri" in result_host.summary)

    # (3) Seed/output overrides are honoured and different seeds differ.
    other_dir = _fresh(report.directory / "refactored_other_seed")
    result_other = run_single_halo(base, host, seed=BASE_SEED + 1, output_dir=other_dir, policy=policy)
    report.check("a different seed changes the realisation", result_other.summary.get("task_seeds") != result_host.summary.get("task_seeds"))
    report.data.update({"original": _key_quantities(original_dir), "refactored_plain": _key_quantities(plain_dir), "refactored_host": result_host.to_dict(), "identical": identical, "identical_host": identical_host})
    report.save()
    return report


# --------------------------------------------------------------------------- E
def checkpoint_e() -> Report:
    report = Report("E_tree_to_single_halo_equivalence", CHECKPOINT_ROOT / "E_tree_to_single_halo_equivalence")
    parser = build_parser()
    base = build_config_from_arguments(parser.parse_args(_base_argv(report.directory / "unused")))
    host = next(iter(MergerTreeReader(TREE_PATH, HostSelection(nodes=(3.0,), max_hosts=1)).iter_hosts()))
    seed = derive_host_seed(0, host)
    resolved = resolve_host_parameters(host, base, SCALING_POLICY)
    report.data["host"] = host.to_dict()
    report.data["resolved"] = resolved.to_dict()
    report.data["seed"] = seed

    # Manual run: the same four numbers typed on the original command line.
    manual_dir = _fresh(report.directory / "manual_cli")
    _run_original_cli(_base_argv(manual_dir, smbh=resolved.smbh_mass_msun, nsc=resolved.nsc_mass_msun, seed=seed, redshift=resolved.redshift, metallicity=resolved.metallicity_zsun))
    # Tree-derived run: the same host through the adapter and the programmatic kernel call.
    tree_dir = _fresh(report.directory / "tree_derived")
    result = run_single_halo(base, host, seed=seed, output_dir=tree_dir, policy=SCALING_POLICY)
    report.check("tree-derived run succeeded", result.status == STATUS_SUCCESS, result.error)
    identical = _files_identical(manual_dir, tree_dir)
    report.check("tree-derived == manual single-halo outputs, byte for byte", all(identical.values()), identical)
    manual_meta = json.loads((manual_dir / "simulation_metadata.json").read_text())["parameters"]
    tree_meta = json.loads((tree_dir / "simulation_metadata.json").read_text())["parameters"]
    same = {k: manual_meta[k] == tree_meta[k] for k in ("smbh_masses_msun", "nsc_masses_msun", "redshift", "metallicity", "base_seed", "num_repeats")}
    report.check("resolved physical parameters agree in metadata", all(same.values()), same)
    report.data.update({"identical": identical, "manual": _key_quantities(manual_dir), "tree_derived": _key_quantities(tree_dir)})
    report.save()
    return report


# --------------------------------------------------------------------------- F
def _batch_signature(root: Path) -> dict[str, Any]:
    summary = pd.read_csv(root / "jobs" / "job_summary.csv")
    per_job = {}
    for job_id in summary["job_id"]:
        job_dir = root / "simulations" / job_id
        per_job[job_id] = {name: _sha256(job_dir / name) for name in COMPARED_FILES if (job_dir / name).is_file()}
    return {
        "job_ids": summary["job_id"].tolist(),
        "seeds": summary["seed"].tolist(),
        "statuses": summary["status"].tolist(),
        "n_imri": summary["n_imri"].tolist(),
        "rate_imri_per_myr": summary["rate_imri_per_myr"].tolist(),
        "task_seeds": summary["task_seeds"].astype(str).tolist(),
        "per_job_digests": per_job,
    }


def checkpoint_f() -> Report:
    report = Report("F_parallel_reproducibility", CHECKPOINT_ROOT / "F_parallel_reproducibility")
    parser = build_parser()
    base = build_config_from_arguments(parser.parse_args(["1", str(report.directory / "unused"), "1", "--base-seed", "1"]))
    selection = HostSelection(nodes=(3.0, 200.0), max_hosts=4)
    signatures = {}
    runtimes = {}
    for workers in (1, 2, 4):
        root = _fresh(report.directory / f"workers_{workers}")
        outcome = run_tree_batch(TREE_PATH, base, root, selection=selection, policy=SCALING_POLICY, settings=BatchSettings(workers=workers, master_seed=42, make_figures=False), progress=None)
        runtimes[workers] = outcome.runtime_s
        signatures[workers] = _batch_signature(root)
        print(f"  workers={workers}: {outcome.status_counts.get(STATUS_SUCCESS)} succeeded in {outcome.runtime_s:.1f} s")
    reference = signatures[1]
    for workers in (2, 4):
        other = signatures[workers]
        report.check(f"workers={workers}: identical job set and order", other["job_ids"] == reference["job_ids"])
        report.check(f"workers={workers}: identical per-job seeds", other["seeds"] == reference["seeds"])
        report.check(f"workers={workers}: identical statuses", other["statuses"] == reference["statuses"])
        report.check(f"workers={workers}: identical per-repeat task seeds", other["task_seeds"] == reference["task_seeds"])
        report.check(f"workers={workers}: identical per-job result files (byte for byte)", other["per_job_digests"] == reference["per_job_digests"])
    report.check("all jobs succeeded", all(s == STATUS_SUCCESS for s in reference["statuses"]), reference["statuses"])
    report.check("no duplicate job identities", len(set(reference["job_ids"])) == len(reference["job_ids"]))
    report.check("no duplicate seeds", len(set(reference["seeds"])) == len(reference["seeds"]))
    report.check("no output collisions (one directory per job)", len({Path(p).name for p in pd.read_csv(report.directory / "workers_1" / "jobs" / "job_summary.csv")["output_path"]}) == len(reference["job_ids"]))
    report.data.update({"runtimes_s": runtimes, "signatures": {k: {kk: vv for kk, vv in v.items() if kk != "per_job_digests"} for k, v in signatures.items()}})
    _plot_runtime(runtimes, report.directory / "figure4_runtime_vs_workers.png")
    report.save()
    return report


def _plot_runtime(runtimes: dict[int, float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    workers = sorted(runtimes)
    figure, axis = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    axis.plot(workers, [runtimes[w] for w in workers], marker="o", color="#2a78d6", linewidth=2, label="measured")
    axis.plot(workers, [runtimes[workers[0]] / w for w in workers], linestyle="--", color="#9a9891", linewidth=1.2, label="ideal 1/N")
    axis.set_xlabel("batch workers")
    axis.set_ylabel("batch runtime [s]")
    axis.set_xticks(workers)
    axis.set_ylim(bottom=0)
    axis.set_title("Figure 4: runtime versus worker count (4 hosts, identical results)", fontsize=9, loc="left")
    axis.legend(frameon=False, fontsize=8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, color="#e4e2dc", linewidth=0.6)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


# --------------------------------------------------------------------------- G
def write_failure_fixture(path: Path) -> Path:
    rows = [
        FIXTURE_HEADER,
        _fixture_row(1, "6.0", "1e12", "1", "1", "3.0e9", "2.0e-4", "0.0"),          # valid (scaling)
        _fixture_row(2, "6.2", "1e12", "1", "1", "nan", "2.0e-4", "0.0"),            # missing Mstar
        _fixture_row(3, "6.4", "1e12", "1", "1", "5.0e11", "2.0e-4", "0.0"),         # M_smbh = 5e7 > domain max
        _fixture_row(4, "-1.0", "1e12", "1", "1", "3.0e9", "2.0e-4", "0.0"),         # invalid z (reader rejects)
        _fixture_row(5, "6.6", "1e12", "2", "1", "2.5e9", "2.0e-4", "0.0"),          # valid (scaling)
    ]
    path.write_text("\n".join(rows) + "\n")
    return path


def checkpoint_g() -> Report:
    report = Report("G_failure_isolation", CHECKPOINT_ROOT / "G_failure_isolation")
    fixture = write_failure_fixture(report.directory / "fixture_failures.csv")
    parser = build_parser()
    base = build_config_from_arguments(parser.parse_args(["1", str(report.directory / "unused"), "1", "--base-seed", "3"]))
    root = _fresh(report.directory / "batch")
    outcome = run_tree_batch(fixture, base, root, policy=SCALING_POLICY, settings=BatchSettings(workers=2, master_seed=7, make_figures=True), progress=None)
    statuses = {r.job_id: r.status for r in outcome.results}
    report.data["statuses"] = statuses
    report.data["status_counts"] = outcome.status_counts
    report.data["reader_counters"] = outcome.reader_counters
    report.check("reader rejected the negative-redshift row before job construction", outcome.reader_counters["invalid_redshift"] == 1 and outcome.reader_counters["jobs_created"] == 4)
    report.check("valid hosts completed", statuses.get("n000001_b000001_r0000000") == STATUS_SUCCESS and statuses.get("n000002_b000001_r0000004") == STATUS_SUCCESS, statuses)
    report.check("missing Mstar -> missing_required_input", statuses.get("n000001_b000001_r0000001") == "missing_required_input", statuses)
    report.check("over-massive host -> outside_local_model_domain", statuses.get("n000001_b000001_r0000002") == "outside_local_model_domain", statuses)
    summary = pd.read_csv(root / "jobs" / "job_summary.csv")
    failed = summary[summary["status"] != STATUS_SUCCESS]
    succeeded = summary[summary["status"] == STATUS_SUCCESS]
    report.check("failed jobs carry an explicit error message", failed["error"].notna().all())
    report.check("failed jobs carry no physical counters (NaN, not zero)", failed["n_imri"].isna().all())
    report.check("succeeded jobs carry physical counters (a zero here is a physical zero)", succeeded["n_imri"].notna().all())
    report.check("failed jobs created no result files", all(not (root / "simulations" / job_id / IMRI_AVERAGED_RESULTS_FILENAME).exists() for job_id in failed["job_id"]))

    # Runtime failure isolation: an unwritable output namespace for one job among valid ones.
    hosts = list(MergerTreeReader(fixture, HostSelection(nodes=(1.0, 2.0))).iter_hosts())
    valid = [h for h in hosts if h.record_index in (0, 4)]
    sim_dir = _fresh(report.directory / "runtime_failure")
    blocker = sim_dir / "blocked"
    blocker.write_text("this file blocks a job output directory\n")
    jobs = [build_job(valid[0], 7, sim_dir), HaloSimulationJob(job_id="blocked_job", host=valid[1], seed=1, output_dir=str(blocker / "nested")), build_job(valid[1], 7, sim_dir)]
    results = run_jobs(jobs, base, SCALING_POLICY, workers=2)
    by_id = {r.job_id: r for r in results}
    report.check("runtime output failure is reported as output_failure", by_id["blocked_job"].status == STATUS_OUTPUT_FAILURE, by_id["blocked_job"].error)
    report.check("jobs around the failure still succeed", all(by_id[j.job_id].status == STATUS_SUCCESS for j in (jobs[0], jobs[2])))
    report.check("results are returned in submission order", [r.job_id for r in results] == [j.job_id for j in jobs])
    report.data["runtime_failure_results"] = [r.to_dict() for r in results]
    report.save()
    return report


# --------------------------------------------------------------------------- H
def checkpoint_h() -> Report:
    report = Report("H_real_tree_batch", CHECKPOINT_ROOT / "H_real_tree_batch")
    root = _fresh(report.directory / "tree_run")
    argv = [
        "1", str(report.directory / "unused"), "1",
        "--tree", str(TREE_PATH), "--output", str(root), "--workers", "4",
        "--master-seed", "2026", "--tree-chunk-size", "2000",
        "--z-min", "6.005", "--z-max", "6.03", "--node", "40,80,120,160,200,240",
        "--max-hosts", "12",
        "--smbh-mass-source", "stellar-scaling", "--nsc-mass-source", "stellar-scaling",
        "--base-seed", "0",
    ]
    started = time.perf_counter()
    run_cli(argv)
    runtime = time.perf_counter() - started
    summary = json.loads((root / "summary.json").read_text())
    jobs = pd.read_csv(root / "jobs" / "job_summary.csv")
    hosts = pd.read_csv(root / "hosts" / "selected_hosts.csv")
    provenance = json.loads((root / "config" / "provenance.json").read_text())
    report.data.update({"argv": argv, "runtime_s": runtime, "summary": summary})
    report.check("CLI batch produced jobs", summary["jobs_total"] == 12, summary["jobs_total"])
    report.check("every job succeeded", summary["status_counts"]["success"] == summary["jobs_total"], summary["status_counts"])
    report.check("selected hosts respect the redshift window", ((hosts["redshift"] >= 6.005) & (hosts["redshift"] <= 6.03)).all())
    report.check("selected hosts respect the node list", hosts["node_id"].isin([40, 80, 120, 160, 200, 240]).all())
    report.check("hosts table and job table are aligned", hosts["host_id"].tolist() == jobs["host_id"].tolist())
    report.check("job identities are unique", jobs["job_id"].is_unique)
    report.check("per-job seeds are unique and reconstructible", jobs["seed"].is_unique and all(int(s) == derive_host_seed(2026, HaloHost(host_id=h, redshift=z, record_index=int(r), node_id=int(n), branch_id=int(b))) for h, z, r, n, b, s in zip(jobs["host_id"], jobs["redshift"], jobs["record_index"], jobs["node_id"], jobs["branch_id"], jobs["seed"])))
    report.check("every job has its own output directory with kernel results", all((root / "simulations" / j / IMRI_AVERAGED_RESULTS_FILENAME).is_file() and (root / "simulations" / j / "run.log").is_file() for j in jobs["job_id"]))
    report.check("provenance records tree, selection, seeds and versions", all(k in provenance for k in ("tree", "selection", "master_seed", "seed_policy", "code", "jobs")) and len(provenance["jobs"]) == 12)
    report.check("batch figures written", (root / "figures" / "selected_hosts_domain.png").is_file() and (root / "figures" / "batch_status.png").is_file())
    report.check("outputs stay host-level (no cosmic-rate fields)", not any("cosmic" in c or "comoving" in c for c in jobs.columns))
    report.data["job_table_head"] = jobs[["job_id", "seed", "status", "redshift", "smbh_mass_msun", "nsc_mass_msun", "n_imri", "rate_imri_per_myr", "runtime_s"]].head(12).to_dict(orient="records")
    report.save()
    return report


CHECKPOINTS: dict[str, Callable[[], Report]] = {
    "A": checkpoint_a,
    "B": checkpoint_b,
    "C": checkpoint_c,
    "D": checkpoint_d,
    "E": checkpoint_e,
    "F": checkpoint_f,
    "G": checkpoint_g,
    "H": checkpoint_h,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=sorted(CHECKPOINTS), default=None)
    arguments = parser.parse_args(argv)
    selected = arguments.only or list(CHECKPOINTS)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    overall: dict[str, str] = {}
    for letter in selected:
        print(f"\n=== Checkpoint {letter} ===")
        started = time.perf_counter()
        report = CHECKPOINTS[letter]()
        overall[letter] = report.status
        print(f"  ({time.perf_counter() - started:.1f} s)")
    with (CHECKPOINT_ROOT / "overall_status.json").open("w") as handle:
        json.dump(overall, handle, indent=2)
    print("\nOverall:", overall)
    return 0 if all(status != "FAIL" for status in overall.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
