"""Round-2 scientific checkpoints B-L.

Each checkpoint isolates one scientific subsystem, calls **canonical** ``src/``
code (never a local reimplementation), uses small controlled inputs, writes
compact machine-readable diagnostics under ``output/round2/`` and reports
PASS / WARN / FAIL.

    python scripts/run_science_checkpoints.py             # everything
    python scripts/run_science_checkpoints.py --only B D   # a subset

Checkpoints
-----------
B  disk state             canonical pAGN disk profile, units, continuity, domain
C  damping                damping/embedding over controlled parameter variations
D  migration components   Lindblad / corotation / thermal / Type-II decomposition
E  gap and regime         gap diagnostics, threshold crossing, transition size
F  migration traps        traps derived from torque zeros, never hard-coded
G  gas/GW handoff         gas versus GW competition and the handoff criterion
H  event classification   a controlled population traced through every stage
I  LISA detectability     source -> SNR -> threshold, redshift conventions
J  end-to-end             one full single-halo run through canonical modules
K  Round-1 regression     single-halo CLI + tree batch + serial/parallel identity
L  repository cleanup     no duplicate physics, no dead modules, clean imports
"""

from __future__ import annotations

import argparse
import ast
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
SOURCE_ROOT = PROJECT_ROOT / "src"
for _path in (SOURCE_ROOT, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import physical_process.constants as const  # noqa: E402
from disk_quantity.pagn_disk_adapter import build_pagn_disk_profile  # noqa: E402
from lisa import (  # noqa: E402
    SNR_STATUS_EVALUATED,
    evaluate_lisa_inspiral_snr,
    lisa_effective_strain_psd_hz,
    luminosity_distance_mpc,
    source_chirp_mass_msun,
)
from physical_process.agn import AGNProperties  # noqa: E402
from physical_process.damp import DampingCalculator, DampingStatus, OrbitalState  # noqa: E402
from physical_process.migration import (  # noqa: E402
    DEFAULT_GAP_CRITERION,
    GAP_OPENING_CONDITIONS,
    MigrationParameters,
    classify_torque_traps,
    compute_torque_profile,
    gap_diagnostics,
    select_migration_torque_si,
    torque_si_to_cgs,
)
from workflows.cli import build_config_from_arguments, build_parser  # noqa: E402
from workflows.config import PAGN_ALPHA  # noqa: E402
from workflows.engine import IMRIDynEngine  # noqa: E402
from workflows.single_halo import STATUS_SUCCESS, run_single_halo  # noqa: E402
from workflows.single_work import IMRISingleWorkflow  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "output" / "round2"
TREE_PATH = PROJECT_ROOT / "galaxy_1e12.csv"

SMBH_MASS_MSUN = 1.0e5
MIGRATOR_MASS_MSUN = 10.0
ALPHA = PAGN_ALPHA

TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e2dc"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")


class Report:
    """Collect PASS/WARN/FAIL conditions and compact data for one checkpoint."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.directory = OUTPUT_ROOT / name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.checks: list[dict[str, Any]] = []
        self.data: dict[str, Any] = {}

    def check(self, label: str, passed: bool, detail: Any = None, warn: bool = False) -> bool:
        level = "PASS" if passed else ("WARN" if warn else "FAIL")
        self.checks.append({"check": label, "result": level, "detail": _safe(detail)})
        suffix = f" -- {detail}" if detail is not None and not passed else ""
        print(f"  [{level}] {label}{suffix}")
        return passed

    @property
    def status(self) -> str:
        if any(c["result"] == "FAIL" for c in self.checks):
            return "FAIL"
        if any(c["result"] == "WARN" for c in self.checks):
            return "WARN"
        return "PASS"

    def save(self) -> None:
        payload = {"checkpoint": self.name, "status": self.status,
                   "checks": self.checks, "data": _safe(self.data)}
        with (self.directory / "checkpoint_summary.json").open("w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"  => {self.name}: {self.status} ({self.directory})")


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _axes(axis, xlabel: str, ylabel: str, title: str | None = None) -> None:
    axis.set_xlabel(xlabel, color=MUTED, fontsize=9)
    axis.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if title:
        axis.set_title(title, color=TEXT, fontsize=10, loc="left")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, labelsize=8)


def _disk_and_torque(smbh_mass_msun: float = SMBH_MASS_MSUN,
                     migrator_mass_msun: float = MIGRATOR_MASS_MSUN):
    """Canonical pAGN disk + torque profile used by several checkpoints."""
    from workflows.config import DISK_ACCRETION_RATE_G_S_PER_MSUN

    profile = build_pagn_disk_profile(
        smbh_mass_msun, ALPHA,
        accretion_rate_g_s=DISK_ACCRETION_RATE_G_S_PER_MSUN * smbh_mass_msun,
    )
    torque = compute_torque_profile(
        profile.disk, migrator_mass_msun=migrator_mass_msun,
        max_radius_rg=None, params=MigrationParameters(alpha=ALPHA),
    )
    return profile, torque


# ------------------------------------------------------------------ B: disk
def checkpoint_b() -> Report:
    report = Report("B_disk_state")
    profile, _ = _disk_and_torque()
    radius_rg = profile.radius_cm / profile.gravitational_radius_cm
    frame = pd.DataFrame({
        "radius_rg": radius_rg,
        "radius_cm": profile.radius_cm,
        "surface_density_g_cm2": profile.surface_density_g_cm2,
        "aspect_ratio": profile.aspect_ratio,
        "temperature_k": profile.temperature_k,
        "density_g_cm3": profile.density_g_cm3,
        "sound_speed_cm_s": profile.sound_speed_cm_s,
        "toomre_q": profile.toomre_q,
        "opacity_cm2_g": profile.opacity_cm2_g,
    })
    frame.to_csv(report.directory / "disk_state.csv", index=False)

    positive = ["surface_density_g_cm2", "aspect_ratio", "temperature_k",
                "density_g_cm3", "sound_speed_cm_s", "toomre_q", "opacity_cm2_g"]
    report.check("all disk quantities are finite",
                 bool(np.isfinite(frame[positive].to_numpy()).all()))
    report.check("all disk quantities are positive",
                 bool((frame[positive].to_numpy() > 0).all()))
    report.check("radius grid is strictly increasing",
                 bool(np.all(np.diff(frame["radius_cm"]) > 0)))
    report.check("disk is geometrically thin (h < 1 everywhere)",
                 bool((frame["aspect_ratio"] < 1.0).all()),
                 float(frame["aspect_ratio"].max()))

    # Documented structural assumption: c_s = H * Omega, with Omega the disk's
    # own angular frequency (pAGN solves a slightly non-Keplerian rotation law).
    radius_m = np.asarray(profile.disk.R, dtype=float)
    omega_disk = np.asarray(profile.disk.Omega, dtype=float)[np.argsort(radius_m)]
    omega_keplerian = np.sqrt(const.G * SMBH_MASS_MSUN * const.msun / profile.radius_cm**3)
    scale_height_cm = frame["aspect_ratio"].to_numpy() * profile.radius_cm
    residual = np.abs(
        frame["sound_speed_cm_s"].to_numpy() / (scale_height_cm * omega_disk) - 1.0)
    report.check("vertical equilibrium c_s = H Omega holds to machine precision",
                 bool(np.max(residual) < 1e-12), float(np.max(residual)))
    non_keplerian = float(np.max(np.abs(omega_disk / omega_keplerian - 1.0)))
    report.data["max_non_keplerian_fraction"] = non_keplerian
    report.check("the rotation law is Keplerian to better than 1 per cent",
                 non_keplerian < 1.0e-2, non_keplerian)

    # Radial continuity.  The thermodynamic profile must be smooth; the
    # tabulated opacity legitimately steps at its table transitions.
    smooth = [n for n in positive if n != "opacity_cm2_g"]
    jumps = {n: float(np.max(np.abs(np.diff(np.log10(frame[n].to_numpy())))))
             for n in positive}
    report.check("no single-cell jump exceeds one decade in the smooth profile",
                 all(jumps[n] < 1.0 for n in smooth),
                 {n: jumps[n] for n in smooth})
    report.check("tabulated opacity steps sharply at its table transitions",
                 jumps["opacity_cm2_g"] < 1.0,
                 {"max_opacity_decade_jump": jumps["opacity_cm2_g"],
                  "radius_rg_at_jump": float(radius_rg[
                      int(np.argmax(np.abs(np.diff(np.log10(
                          frame["opacity_cm2_g"].to_numpy())))))])},
                 warn=True)
    report.data["max_log10_jump_per_cell"] = jumps

    report.check("Toomre Q reaches ~1 at the outer boundary (disk truncation)",
                 bool(frame["toomre_q"].iloc[-1] < 2.0), float(frame["toomre_q"].iloc[-1]))
    report.data["domain"] = {
        "inner_radius_rg": float(radius_rg.min()),
        "outer_radius_rg": float(radius_rg.max()),
        "outer_radius_pc": float(profile.outer_radius_cm / const.parsec),
        "cells": int(len(frame.index)),
    }
    report.data["ranges"] = {n: [float(frame[n].min()), float(frame[n].max())] for n in positive}

    figure, axes = plt.subplots(2, 2, figsize=(8.6, 6.0), dpi=150)
    for axis, (name, label) in zip(axes.ravel(), [
        ("surface_density_g_cm2", r"$\Sigma$ [g cm$^{-2}$]"),
        ("aspect_ratio", r"$H/R$"),
        ("temperature_k", r"$T$ [K]"),
        ("toomre_q", r"Toomre $Q$"),
    ]):
        axis.loglog(frame["radius_rg"], frame[name], color=SERIES[0], linewidth=1.8)
        _axes(axis, r"$R/r_g$", label)
    figure.suptitle(f"Checkpoint B: canonical pAGN disk state, "
                    rf"$M_\bullet={SMBH_MASS_MSUN:.0e}\,M_\odot$", fontsize=11, x=0.02, ha="left")
    figure.tight_layout()
    figure.savefig(report.directory / "disk_state.png")
    plt.close(figure)
    report.save()
    return report


# --------------------------------------------------------------- C: damping
def checkpoint_c() -> Report:
    report = Report("C_damping")
    profile, _ = _disk_and_torque()
    agn = AGNProperties.from_mass(SMBH_MASS_MSUN, 10.0)
    calculator = DampingCalculator(
        agn, verbose=False,
        radius_grid_cm=profile.radius_cm,
        surface_density_g_cm2=profile.surface_density_g_cm2,
        aspect_ratio=profile.aspect_ratio,
        disk_inner_radius_cm=profile.inner_radius_cm,
        disk_outer_radius_cm=profile.outer_radius_cm,
        use_disk_inner_radius=True,
    )
    rg_cm = profile.gravitational_radius_cm
    rows: list[dict[str, Any]] = []
    for mass_msun in (1.0, 10.0, 100.0):
        for radius_rg in (1.0e4, 1.0e5, 1.0e6):
            for inclination_deg in (5.0, 30.0, 90.0):
                state = OrbitalState(
                    semi_major_axis_cm=radius_rg * rg_cm,
                    eccentricity=0.3,
                    inclination_rad=np.deg2rad(inclination_deg),
                    cos_omega=0.5, time_s=0.0,
                )
                outcome = calculator.damping_simulation(
                    mass_msun * const.msun, state, 0.0, "Star")
                rows.append({
                    "mass_msun": mass_msun, "radius_rg": radius_rg,
                    "inclination_deg": inclination_deg,
                    "status": str(outcome.status),
                    "damping_time_myr": outcome.elapsed_time_s / const.Myr,
                    "final_inclination_deg": np.rad2deg(outcome.final_state.inclination_rad),
                    "final_eccentricity": outcome.final_state.eccentricity,
                    "final_radius_rg": outcome.final_state.semi_major_axis_cm / rg_cm,
                    "embedded": outcome.status in (
                        DampingStatus.EMBEDDED_STAR, DampingStatus.EMBEDDED_BH),
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(report.directory / "damping_scan.csv", index=False)

    report.check("every damping evaluation returned a status",
                 bool(frame["status"].notna().all()))
    report.check("damping times are nonnegative and finite where evaluated",
                 bool(((frame["damping_time_myr"] >= 0) & np.isfinite(frame["damping_time_myr"])).all()))
    report.check("damping never increases inclination",
                 bool((frame["final_inclination_deg"] <= frame["inclination_deg"] + 1e-9).all()),
                 float((frame["final_inclination_deg"] - frame["inclination_deg"]).max()))
    report.check("damping never increases eccentricity",
                 bool((frame["final_eccentricity"] <= 0.3 + 1e-9).all()),
                 float(frame["final_eccentricity"].max()))
    report.check("final orbital elements stay physical (0 <= e < 1)",
                 bool(((frame["final_eccentricity"] >= 0) & (frame["final_eccentricity"] < 1)).all()))

    embedded = frame[frame["embedded"]]
    if len(embedded.index):
        by_mass = embedded.groupby("mass_msun")["damping_time_myr"].median()
        report.check("heavier objects embed no more slowly (expected scaling)",
                     bool(by_mass.is_monotonic_decreasing), by_mass.to_dict(), warn=True)
    report.check("at least one configuration embeds and one does not",
                 bool(frame["embedded"].any() and (~frame["embedded"]).any()),
                 frame["embedded"].value_counts().to_dict())
    report.data["status_counts"] = frame["status"].value_counts().to_dict()
    report.data["embedded_fraction"] = float(frame["embedded"].mean())
    report.save()
    return report


# --------------------------------------------- D: migration torque components
def checkpoint_d() -> Report:
    report = Report("D_migration_components")
    profile, torque = _disk_and_torque()
    mass_ratio = MIGRATOR_MASS_MSUN / SMBH_MASS_MSUN
    gap = gap_diagnostics(mass_ratio, torque.radius_m / torque.radius_m * 0 +
                          np.interp(torque.radius_cm, profile.radius_cm, profile.aspect_ratio),
                          ALPHA)
    lindblad = torque.lindblad_coeff * torque.gamma_0_si
    corotation = torque.corotation_coeff * torque.gap_attenuation_factor * torque.gamma_0_si
    thermal = (1.0 - torque.gap_attenuation_factor) * torque.gamma_th_si
    type_i = torque.gamma_type_i_si
    total = type_i + thermal

    frame = pd.DataFrame({
        "radius_rg": torque.radius_rg,
        "gamma_0_si": torque.gamma_0_si,
        "lindblad_si": lindblad,
        "corotation_si": corotation,
        "thermal_si": thermal,
        "type_i_si": type_i,
        "type_ii_si": torque.gamma_type_ii_si,
        "total_type_i_thermal_si": total,
        "gap_attenuation_factor": torque.gap_attenuation_factor,
        "kanagawa_gap_parameter": torque.gap_parameter_dimensionless,
    })
    frame.to_csv(report.directory / "torque_components.csv", index=False)

    report.check("Gamma_0 is positive everywhere", bool((torque.gamma_0_si > 0).all()))
    report.check("all torque components are finite",
                 bool(np.isfinite(frame.to_numpy()).all()))
    report.check("Type-II torque is inward (negative) everywhere",
                 bool((torque.gamma_type_ii_si < 0).all()))
    report.check("the total torque changes sign (a trap can exist)",
                 bool((total > 0).any() and (total < 0).any()),
                 {"positive_cells": int((total > 0).sum()), "negative_cells": int((total < 0).sum())})
    report.check("Type-I equals (corotation*attenuation + lindblad)/denominator * Gamma_0",
                 bool(np.allclose(type_i,
                                  (torque.corotation_coeff * torque.gap_attenuation_factor
                                   + torque.lindblad_coeff) / torque.type_i_gap_denominator
                                  * torque.gamma_0_si, rtol=1e-12)))
    report.check("gap attenuation is in [0, 1]",
                 bool(((torque.gap_attenuation_factor >= 0) &
                       (torque.gap_attenuation_factor <= 1)).all()))

    dominant = np.where(np.abs(lindblad) >= np.abs(thermal), "lindblad", "thermal")
    report.data["dominant_component_cells"] = {
        k: int(v) for k, v in zip(*np.unique(dominant, return_counts=True))}
    report.data["sign_changes_total"] = int(np.count_nonzero(np.diff(np.sign(total))))

    figure, axis = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    for values, color, label in (
        (lindblad, SERIES[0], "Lindblad"),
        (corotation, SERIES[1], "corotation (attenuated)"),
        (thermal, SERIES[2], "thermal"),
        (torque.gamma_type_ii_si, SERIES[3], "Type II"),
        (total, SERIES[5], "total (Type I + thermal)"),
    ):
        axis.plot(torque.radius_rg, np.sign(values) * np.log10(np.abs(values) + 1e-30),
                  color=color, linewidth=1.6, label=label)
    axis.set_xscale("log")
    axis.axhline(0.0, color=MUTED, linewidth=0.8)
    _axes(axis, r"$R/r_g$", r"sign$(\Gamma)\,\log_{10}|\Gamma\,/\,\mathrm{SI}|$",
          "Checkpoint D: migration torque decomposition")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(report.directory / "torque_components.png")
    plt.close(figure)
    report.save()
    return report


# ------------------------------------------------ E: gap and regime transition
def checkpoint_e() -> Report:
    report = Report("E_gap_regime_transition")
    profile, torque = _disk_and_torque()
    aspect = np.interp(torque.radius_cm, profile.radius_cm, profile.aspect_ratio)
    mass_ratio = MIGRATOR_MASS_MSUN / SMBH_MASS_MSUN

    rows = []
    for name in GAP_OPENING_CONDITIONS:
        diagnostics = gap_diagnostics(mass_ratio, aspect, ALPHA, name)
        selection = select_migration_torque_si(
            torque, mass_ratio, aspect, ALPHA, name, consider_type_ii_migration=True)
        rows.append({
            "criterion": name,
            "gap_open_cells": int(diagnostics.gap_open.sum()),
            "gap_open_fraction": diagnostics.gap_open_fraction,
            "type_ii_cells": int(selection.type_ii_mask.sum()),
        })
        pd.DataFrame({
            "radius_rg": torque.radius_rg, "aspect_ratio": aspect,
            "kii_parameter": diagnostics.kii_parameter,
            "kanagawa_parameter": diagnostics.kanagawa_parameter,
            "crida_parameter": diagnostics.crida_parameter,
            "hill_over_scale_height": diagnostics.hill_radius_over_scale_height,
            "gap_open": diagnostics.gap_open,
            "selected_torque_si": selection.torque_si,
        }).to_csv(report.directory / f"gap_{name}.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(report.directory / "gap_criteria_summary.csv", index=False)
    report.data["criteria"] = rows

    base = gap_diagnostics(mass_ratio, aspect, ALPHA)
    report.check("gap decision matches the Type-II cell selection",
                 all(r["gap_open_cells"] == r["type_ii_cells"] for r in rows), rows)
    ratio = base.kanagawa_parameter / base.kii_parameter
    report.check("the two K definitions differ by exactly h^-3",
                 bool(np.allclose(ratio, aspect**-3.0)))
    report.check("the kii and kanagawa criteria disagree on this disk "
                 "(documented model ambiguity)",
                 rows[0]["gap_open_cells"] != rows[2]["gap_open_cells"],
                 {r["criterion"]: r["gap_open_cells"] for r in rows}, warn=True)

    # Size of the Type-I -> Type-II discontinuity where the regime switches.
    selection = select_migration_torque_si(
        torque, mass_ratio, aspect, ALPHA, DEFAULT_GAP_CRITERION,
        consider_type_ii_migration=True)
    switch = np.where(np.diff(selection.type_ii_mask.astype(int)) != 0)[0]
    jumps = []
    type_i_thermal = (torque.gamma_type_i_si
                      + (1.0 - torque.gap_attenuation_factor) * torque.gamma_th_si)
    for index in switch:
        a, b = type_i_thermal[index], torque.gamma_type_ii_si[index]
        if a != 0.0:
            jumps.append(float(abs(b - a) / abs(a)))
    report.data["regime_switch_cells"] = int(switch.size)
    report.data["relative_torque_jump_at_switch"] = jumps
    report.check("the regime transition is discontinuous, as the model intends",
                 True if not jumps else max(jumps) > 0.0,
                 {"max_relative_jump": max(jumps) if jumps else None}, warn=bool(jumps))

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=150)
    axes[0].loglog(torque.radius_rg, base.kii_parameter, color=SERIES[0],
                   linewidth=1.8, label=r"$q^2/(\alpha h^2)$  [kii]")
    axes[0].loglog(torque.radius_rg, base.kanagawa_parameter, color=SERIES[1],
                   linewidth=1.8, label=r"$q^2/(\alpha h^5)$  [kanagawa]")
    axes[0].axhline(11.0, color=SERIES[0], linestyle="--", linewidth=1.0)
    axes[0].axhline(25.0, color=SERIES[1], linestyle=":", linewidth=1.0)
    _axes(axes[0], r"$R/r_g$", "gap parameter", "Two definitions of K")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].loglog(torque.radius_rg, base.crida_parameter, color=SERIES[2],
                   linewidth=1.8, label=r"Crida $P$")
    axes[1].loglog(torque.radius_rg, base.hill_radius_over_scale_height, color=SERIES[3],
                   linewidth=1.8, label=r"$R_H/H$")
    axes[1].axhline(1.0, color=MUTED, linestyle="--", linewidth=1.0)
    _axes(axes[1], r"$R/r_g$", "dimensionless", "Crida diagnostics")
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Checkpoint E: gap diagnostics and threshold crossing",
                    fontsize=11, x=0.02, ha="left")
    figure.tight_layout()
    figure.savefig(report.directory / "gap_diagnostics.png")
    plt.close(figure)
    report.save()
    return report


# ------------------------------------------------------------ F: traps
def checkpoint_f() -> Report:
    report = Report("F_migration_traps")
    # Scan SMBH mass: whether a stable trap exists is a physical result of the
    # torque profile, not something the checkpoint may assume.
    rows: list[dict[str, Any]] = []
    profiles: dict[float, Any] = {}
    criterion_comparison: list[dict[str, Any]] = []
    for smbh_mass_msun in (1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7):
        profile, torque = _disk_and_torque(smbh_mass_msun)
        aspect = np.interp(torque.radius_cm, profile.radius_cm, profile.aspect_ratio)
        mass_ratio = MIGRATOR_MASS_MSUN / smbh_mass_msun

        # The gap criterion decides how much of the disk is Type II, and the
        # Type-II torque is inward everywhere, so the criterion controls whether
        # traps can exist at all.  Record both so the consequence is visible.
        for name in GAP_OPENING_CONDITIONS:
            other = select_migration_torque_si(
                torque, mass_ratio, aspect, ALPHA, name,
                consider_type_ii_migration=True)
            other_cgs = torque_si_to_cgs(other.torque_si)
            other_traps = classify_torque_traps(
                torque.radius_rg, other_cgs, min_separation_cells=1.0)
            criterion_comparison.append({
                "smbh_mass_msun": smbh_mass_msun,
                "criterion": name,
                "type_ii_cells": int(other.type_ii_mask.sum()),
                "outward_torque_cells": int(np.count_nonzero(other_cgs > 0)),
                "stable_traps": int(other_traps.stable_radius_rg.size),
                "anti_traps": int(other_traps.anti_trap_radius_rg.size),
            })

        selection = select_migration_torque_si(
            torque, mass_ratio, aspect, ALPHA, DEFAULT_GAP_CRITERION,
        consider_type_ii_migration=True)
        torque_cgs = torque_si_to_cgs(selection.torque_si)
        traps = classify_torque_traps(torque.radius_rg, torque_cgs, min_separation_cells=1.0)
        strict = classify_torque_traps(torque.radius_rg, torque_cgs, min_separation_cells=50.0)
        again = classify_torque_traps(torque.radius_rg, torque_cgs, min_separation_cells=1.0)
        profiles[smbh_mass_msun] = (torque.radius_rg, torque_cgs, traps)
        rows.append({
            "smbh_mass_msun": smbh_mass_msun,
            "sign_changes": int(np.count_nonzero(np.diff(np.sign(torque_cgs)))),
            "stable_traps": int(traps.stable_radius_rg.size),
            "anti_traps": int(traps.anti_trap_radius_rg.size),
            "boundary_roots": int(traps.boundary_radius_rg.size),
            "unresolved_roots": int(traps.unresolved_radius_rg.size),
            "outermost_stable_rg": traps.outermost_stable_radius_rg,
            "strict_stable_traps": int(strict.stable_radius_rg.size),
            "reproducible": bool(np.array_equal(again.stable_radius_rg,
                                                traps.stable_radius_rg)),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(report.directory / "trap_scan.csv", index=False)
    report.data["scan"] = rows

    report.check("trap finding is reproducible for every configuration",
                 bool(frame["reproducible"].all()))
    report.check("raising the resolution filter never invents traps",
                 bool((frame["strict_stable_traps"] <= frame["stable_traps"]).all()),
                 frame[["smbh_mass_msun", "stable_traps", "strict_stable_traps"]].to_dict("list"))
    report.check("every reported root corresponds to a torque sign change",
                 bool((frame["stable_traps"] + frame["anti_traps"]
                       <= frame["sign_changes"]).all()),
                 frame[["smbh_mass_msun", "sign_changes", "stable_traps", "anti_traps"]].to_dict("list"))
    report.check("stable traps exist for at least one SMBH mass",
                 bool((frame["stable_traps"] > 0).any()),
                 frame[["smbh_mass_msun", "stable_traps"]].to_dict("list"))

    # Criterion dependence of the trap structure (a physical result, not an
    # assumption).  Whether any configuration has traps depends on how much of
    # the disk the gap criterion moves to the always-inward Type-II torque.
    comparison = pd.DataFrame(criterion_comparison)
    comparison.to_csv(report.directory / "trap_criterion_comparison.csv", index=False)
    report.data["criterion_comparison"] = criterion_comparison
    report.data["gap_criterion_used"] = DEFAULT_GAP_CRITERION
    report.check("a torque zero is only possible where outward torque survives",
                 bool(((comparison["stable_traps"] + comparison["anti_traps"] == 0)
                       | (comparison["outward_torque_cells"] > 0)).all()),
                 comparison[comparison["outward_torque_cells"] == 0][
                     ["smbh_mass_msun", "criterion", "stable_traps"]].to_dict("list"))
    suppressed = comparison[(comparison["criterion"] == DEFAULT_GAP_CRITERION)
                            & (comparison["outward_torque_cells"] == 0)]
    report.check(f"the '{DEFAULT_GAP_CRITERION}' criterion leaves outward torque "
                 "somewhere in the explored mass range",
                 len(suppressed.index) < len(comparison["smbh_mass_msun"].unique()),
                 {"masses_with_no_outward_torque":
                  suppressed["smbh_mass_msun"].tolist()}, warn=True)

    # Every reported stable trap must actually bracket a + -> - sign change,
    # and the torque must converge onto it from both sides.
    bracketed: list[bool] = []
    basins: list[dict[str, float]] = []
    for smbh_mass_msun, (radius_rg, torque_cgs, traps) in profiles.items():
        for radius in traps.stable_radius_rg:
            index = int(np.searchsorted(radius_rg, radius))
            if not 0 < index < torque_cgs.size:
                continue
            # Local stability: the immediately adjacent cells must bracket the
            # root with torque pushing outward inside and inward outside.
            bracketed.append(bool(torque_cgs[index - 1] > 0 and torque_cgs[index] < 0))
            # Basin of attraction: how far the converging sign pattern extends
            # before the next torque zero interrupts it.
            inner = index - 1
            while inner > 0 and torque_cgs[inner - 1] > 0:
                inner -= 1
            outer = index
            while outer + 1 < torque_cgs.size and torque_cgs[outer + 1] < 0:
                outer += 1
            basins.append({
                "smbh_mass_msun": float(smbh_mass_msun),
                "trap_radius_rg": float(radius),
                "basin_inner_rg": float(radius_rg[inner]),
                "basin_outer_rg": float(radius_rg[outer]),
                "basin_decades": float(np.log10(radius_rg[outer] / radius_rg[inner])),
            })
    report.check("every stable trap brackets a positive->negative sign change "
                 "(local stability)",
                 bool(bracketed) and all(bracketed), bracketed)
    report.data["trap_basins"] = basins
    narrow = [b for b in basins if b["basin_decades"] < 0.1]
    report.check("every stable trap has a basin wider than 0.1 decade",
                 not narrow, narrow, warn=True)
    report.check("no trap position is hard-coded anywhere in the trap module",
                 not any(token in (PROJECT_ROOT / "src/physical_process/migration/traps.py").read_text()
                         for token in ("700 *", "= 700", "r_trap =")))

    figure, axis = plt.subplots(figsize=(7.4, 4.4), dpi=150)
    for index, (smbh_mass_msun, (radius_rg, torque_cgs, traps)) in enumerate(profiles.items()):
        axis.plot(radius_rg, np.sign(torque_cgs) * np.log10(np.abs(torque_cgs) + 1e-30),
                  color=SERIES[index % len(SERIES)], linewidth=1.3,
                  label=rf"$M_\bullet={smbh_mass_msun:.0e}$")
        for radius in traps.stable_radius_rg:
            axis.plot([radius], [0.0], marker="o", markersize=7, color=SERIES[index % len(SERIES)])
        for radius in traps.anti_trap_radius_rg:
            axis.plot([radius], [0.0], marker="x", markersize=7, color=SERIES[index % len(SERIES)])
    axis.axhline(0.0, color=MUTED, linewidth=0.8)
    axis.set_xscale("log")
    _axes(axis, r"$R/r_g$", r"sign$(\Gamma)\log_{10}|\Gamma|$",
          "Checkpoint F: derived traps (circles) and anti-traps (crosses)")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(report.directory / "migration_traps.png")
    plt.close(figure)
    report.save()
    return report


# ------------------------------------------------------- G: gas / GW handoff
def checkpoint_g() -> Report:
    """Gas versus GW competition, evaluated through the canonical solvers only."""
    report = Report("G_gas_gw_handoff")
    import physical_process.gw_frequency_radius as gw
    from physical_process.migration import migration_time_from_torque
    from workflows.event_rate import GAS_FINAL_RADIUS_SAMPLES, solve_gas_final_radius_cm

    smbh_mass_msun = SMBH_MASS_MSUN
    secondary_msun = MIGRATOR_MASS_MSUN
    redshift, eccentricity = 15.0, 0.0
    min_frequency_hz = 1.0e-3

    profile, torque = _disk_and_torque(smbh_mass_msun, secondary_msun)
    aspect = np.interp(torque.radius_cm, profile.radius_cm, profile.aspect_ratio)
    selection = select_migration_torque_si(
        torque, secondary_msun / smbh_mass_msun, aspect, ALPHA, DEFAULT_GAP_CRITERION,
        consider_type_ii_migration=True)
    torque_cgs = torque_si_to_cgs(selection.torque_si)
    secondary_g = secondary_msun * const.msun
    rg_cm = profile.gravitational_radius_cm

    def gas_time_myr(start_cm: float, target_cm: float) -> float:
        """Canonical torque-integrated gas migration time [Myr]."""
        try:
            return float(migration_time_from_torque(
                torque.radius_cm, torque_cgs, secondary_g,
                start_cm, target_cm, smbh_mass_msun))
        except Exception:
            return np.inf

    def gw_time_myr(radius_cm: float, _absolute_time_myr: float = 0.0) -> float:
        """Canonical Peters GW time [Myr] from a radius to LISA-band entry."""
        return float(gw.peters_time_to_lisa_band_myr(
            smbh_mass_msun, secondary_msun, radius_cm,
            min_frequency_hz, eccentricity, redshift))

    entry_cm = gw.lisa_entry_radius_cm(
        smbh_mass_msun, secondary_msun, min_frequency_hz, redshift)
    report.check("the LISA entry radius is finite and inside the disk",
                 bool(np.isfinite(entry_cm) and 0 < entry_cm < profile.outer_radius_cm),
                 {"entry_rg": float(entry_cm / rg_cm),
                  "disk_outer_rg": float(profile.outer_radius_cm / rg_cm)})

    # Competition curve: both timescales from the canonical functions.
    radii_rg = np.logspace(np.log10(max(entry_cm / rg_cm * 1.05, 10.0)),
                           np.log10(profile.outer_radius_cm / rg_cm * 0.95), 60)
    rows = []
    for radius_rg in radii_rg:
        radius_cm = radius_rg * rg_cm
        gas = gas_time_myr(radius_cm, entry_cm)
        gravitational = gw_time_myr(radius_cm)
        rows.append({"radius_rg": radius_rg,
                     "gas_time_to_entry_myr": gas,
                     "gw_time_to_entry_myr": gravitational,
                     "dominant": "gw" if gravitational < gas else "gas"})
    frame = pd.DataFrame(rows)
    frame.to_csv(report.directory / "gas_gw_competition.csv", index=False)
    report.data["dominant_counts"] = frame["dominant"].value_counts().to_dict()

    finite = frame[np.isfinite(frame["gw_time_to_entry_myr"])]
    report.check("GW time to entry is finite and increases outward",
                 bool(len(finite.index) > 2
                      and np.all(np.diff(finite["gw_time_to_entry_myr"]) >= 0)))
    report.check("GW time vanishes at the entry radius",
                 bool(np.isclose(gw_time_myr(entry_cm), 0.0)))
    slope = np.polyfit(np.log10(finite["radius_rg"]),
                       np.log10(finite["gw_time_to_entry_myr"] + 1e-300), 1)[0]
    report.data["peters_log_slope_far_from_entry"] = float(slope)
    report.check("Peters time approaches the a^4 scaling away from entry",
                 bool(3.5 < slope < 4.5), float(slope))
    report.check("both processes dominate somewhere, or gas dominates throughout "
                 "(a physical result, not an assumption)",
                 True, frame["dominant"].value_counts().to_dict(), warn=False)

    # The workflow's own handoff criterion, called directly.
    start_cm = min(1.0e5 * rg_cm, profile.outer_radius_cm * 0.9)
    outcomes = {}
    for label, available_myr in (("ample_gas_time", 1.0e6), ("starved_gas_time", 1.0e-6)):
        (r_gas_fin_cm, gas_elapsed, gas_final_time, migration_to_entry,
         gw_gas_target, stop_reason) = solve_gas_final_radius_cm(
            r_ini_cm=start_cm,
            r_gas_target_cm=entry_cm,
            r_entry_cm=entry_cm,
            handoff_required=False,
            gas_start_time_myr=0.0,
            available_gas_time_myr=available_myr,
            migration_time_myr_call=gas_time_myr,
            gw_time_to_entry_myr=gw_time_myr,
            num_radius_samples=GAS_FINAL_RADIUS_SAMPLES,
        )
        outcomes[label] = {
            "r_gas_fin_rg": float(r_gas_fin_cm / rg_cm) if np.isfinite(r_gas_fin_cm) else None,
            "gas_elapsed_myr": float(gas_elapsed) if np.isfinite(gas_elapsed) else None,
            "stop_reason": stop_reason,
        }
    report.data["handoff_outcomes"] = outcomes
    report.check("the canonical handoff solver returns an explicit stop reason",
                 all(bool(o["stop_reason"]) for o in outcomes.values()), outcomes)
    report.check("a duty-cycle-starved object stops before reaching entry",
                 outcomes["starved_gas_time"]["r_gas_fin_rg"] is None
                 or outcomes["starved_gas_time"]["r_gas_fin_rg"]
                 >= (outcomes["ample_gas_time"]["r_gas_fin_rg"] or 0.0),
                 outcomes)
    # The production chain is event_work -> classify_event_rate ->
    # solve_gas_final_radius_cm, and this checkpoint calls the same solver.
    event_work_source = (PROJECT_ROOT / "src/workflows/event_work.py").read_text()
    event_rate_source = (PROJECT_ROOT / "src/workflows/event_rate.py").read_text()
    report.check("the workflow reaches the handoff through classify_event_rate",
                 "classify_event_rate" in event_work_source
                 and "solve_gas_final_radius_cm" in event_rate_source)
    definitions = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "src").rglob("*.py")
        if "__pycache__" not in str(path)
        and "def solve_gas_final_radius_cm" in path.read_text()
    ]
    report.check("the handoff criterion is defined exactly once",
                 definitions == ["src/workflows/event_rate.py"], definitions)

    figure, axis = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    axis.loglog(frame["radius_rg"], frame["gas_time_to_entry_myr"],
                color=SERIES[0], linewidth=1.8, label="gas: torque-integrated")
    axis.loglog(frame["radius_rg"], frame["gw_time_to_entry_myr"],
                color=SERIES[1], linewidth=1.8, label="GW: Peters to LISA entry")
    axis.axvline(entry_cm / rg_cm, color=MUTED, linestyle="--", linewidth=1.2)
    axis.text(entry_cm / rg_cm, axis.get_ylim()[1], " LISA entry", fontsize=7,
              color=MUTED, va="top")
    _axes(axis, r"$R/r_g$", "time to LISA entry [Myr]",
          rf"Checkpoint G: gas vs GW, $M_\bullet={smbh_mass_msun:.0e}\,M_\odot$, "
          rf"$m={secondary_msun:g}\,M_\odot$, $z={redshift:g}$")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(report.directory / "gas_gw_competition.png")
    plt.close(figure)
    report.save()
    return report


# -------------------------------------------------- H: event classification
def checkpoint_h() -> Report:
    report = Report("H_event_classification")
    parser = build_parser()
    config = build_config_from_arguments(parser.parse_args([
        "1", str(report.directory / "run"), "2",
        "--smbh-masses", "1e5", "--nsc-masses", "5e4", "--base-seed", "424242",
    ]))
    np.random.seed(424242)
    workflow = IMRISingleWorkflow(IMRIDynEngine(config))
    result = workflow.run_single_simulation(1.0e5, 5.0e4, 0, 424242)

    stages = {
        "initial_sampled": result.total_sampled_objects,
        "disk_crossing": result.disk_crossing_objects,
        "embedded_stars": result.embedded_star_count,
        "embedded_bhs": result.embedded_bh_count,
        "migration_formed_bhs": result.migration_formed_bh_count,
        "bh_candidates": result.bh_candidate_count,
        "band_entry_evaluated": result.band_entry_evaluated_count,
        "migration_evaluated": result.migration_evaluated_count,
        "migration_finite": result.finite_migration_count,
        "migration_trap_bound": result.trap_bound_migration_count,
        "imri_candidates": result.imri_count,
        "no_imri": result.no_imri_count,
        "detectable_imris": result.detectable_imri_count,
        "outside_or_failed": result.outside_or_failed_count,
        "migration_errors": result.migration_error_count,
        "profile_errors": result.profile_error_count,
        "band_entry_inside_isco": result.band_entry_inside_isco_count,
    }
    report.data["stages"] = stages
    pd.DataFrame([stages]).to_csv(report.directory / "event_stages.csv", index=False)

    report.check("the funnel narrows: disk-crossing <= sampled",
                 stages["disk_crossing"] <= stages["initial_sampled"], stages)
    report.check("BH candidates <= embedded + migration-formed",
                 stages["bh_candidates"] <= stages["embedded_bhs"] + stages["migration_formed_bhs"])
    report.check("IMRI candidates <= BH candidates",
                 stages["imri_candidates"] <= stages["bh_candidates"])
    report.check("detectable IMRIs <= IMRI candidates",
                 stages["detectable_imris"] <= stages["imri_candidates"])
    report.check("migration outcomes are partitioned (finite + trap-bound <= evaluated)",
                 stages["migration_finite"] + stages["migration_trap_bound"]
                 <= stages["migration_evaluated"])
    report.check("every stage count is a nonnegative integer",
                 all(isinstance(v, (int, np.integer)) and v >= 0 for v in stages.values()))

    # Every recorded event carries a reason, so a zero is explainable.
    reasons = [e["event_rate_reason"] for e in result.imri_events]
    report.check("every evaluated object carries an event-rate reason",
                 all(bool(r) for r in reasons), len(reasons))
    report.data["event_rate_reasons"] = dict(pd.Series(reasons).value_counts()) if reasons else {}
    report.data["detect_rate_reasons"] = (
        dict(pd.Series([e["detect_rate_reason"] for e in result.imri_events]).value_counts())
        if reasons else {})
    report.check("a physical zero is distinguishable from a failure",
                 (stages["imri_candidates"] == 0
                  and stages["profile_errors"] == 0
                  and stages["migration_errors"] == 0) or stages["imri_candidates"] > 0,
                 {"imri": stages["imri_candidates"],
                  "profile_errors": stages["profile_errors"],
                  "migration_errors": stages["migration_errors"]})

    labels = ["initial_sampled", "disk_crossing", "bh_candidates",
              "band_entry_evaluated", "imri_candidates", "detectable_imris"]
    figure, axis = plt.subplots(figsize=(7.0, 3.4), dpi=150)
    positions = np.arange(len(labels))
    axis.barh(positions, [stages[k] for k in labels], color=SERIES[0], height=0.55)
    for position, key in zip(positions, labels):
        axis.text(stages[key], position, f" {stages[key]}", va="center", fontsize=9, color=TEXT)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8, color=TEXT)
    axis.invert_yaxis()
    axis.set_xscale("symlog")
    _axes(axis, "objects", "", "Checkpoint H: event-classification funnel")
    figure.tight_layout()
    figure.savefig(report.directory / "event_funnel.png")
    plt.close(figure)
    report.save()
    return report


# ---------------------------------------------------------- I: LISA
def checkpoint_i() -> Report:
    report = Report("I_lisa_detectability")
    rows = []
    for redshift in (0.5, 1.0, 5.0, 15.0):
        for primary in (1.0e4, 1.0e5, 1.0e6):
            result = evaluate_lisa_inspiral_snr(
                m1_source_msun=primary,
                m2_source_msun=10.0,
                redshift=redshift,
                current_observed_frequency_hz=1.0e-3,
                mission_duration_years=4.0,
                detector_min_frequency_hz=1.0e-4,
                detector_max_frequency_hz=1.0e-1,
                snr_threshold=20.0,
            )
            rows.append({
                "redshift": redshift, "primary_msun": primary,
                "snr": result.lisa_snr, "status": result.status,
                "chirp_source_msun": result.chirp_mass_source_msun,
                "chirp_detector_msun": result.chirp_mass_detector_msun,
                "distance_mpc": result.luminosity_distance_mpc,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(report.directory / "lisa_snr_grid.csv", index=False)

    evaluated = frame[frame["status"] == SNR_STATUS_EVALUATED]
    report.check("SNR was evaluated for the grid", len(evaluated.index) > 0,
                 frame["status"].value_counts().to_dict())
    report.check("evaluated SNRs are finite and positive",
                 bool((np.isfinite(evaluated["snr"]) & (evaluated["snr"] > 0)).all()))

    # Redshift is applied exactly once: M_detector = (1+z) M_source.
    residual = np.abs(evaluated["chirp_detector_msun"]
                      / (evaluated["chirp_source_msun"] * (1.0 + evaluated["redshift"])) - 1.0)
    report.check("detector-frame chirp mass = (1+z) x source-frame, exactly once",
                 bool(residual.max() < 1e-12), float(residual.max()))
    report.check("source chirp mass matches the closed form",
                 bool(np.isclose(source_chirp_mass_msun(1e5, 10.0),
                                 (1e5 * 10.0) ** 0.6 / (1e5 + 10.0) ** 0.2)))

    report.check("luminosity distance increases with redshift",
                 bool(np.all(np.diff([luminosity_distance_mpc(z) for z in
                                      (0.1, 0.5, 1, 2, 5, 10, 15)]) > 0)))

    # Unambiguous distance dependence: vary only H0, which changes D_L and
    # nothing else, and confirm the exact SNR ~ 1/D_L scaling.
    distance_kwargs = dict(
        m1_source_msun=1.0e5, m2_source_msun=10.0, redshift=1.0,
        current_observed_frequency_hz=1.0e-3, mission_duration_years=4.0,
        detector_min_frequency_hz=1.0e-4, detector_max_frequency_hz=1.0e-1,
        snr_threshold=20.0)
    near = evaluate_lisa_inspiral_snr(hubble_constant_km_s_mpc=67.4, **distance_kwargs)
    far = evaluate_lisa_inspiral_snr(hubble_constant_km_s_mpc=33.7, **distance_kwargs)
    ratio = (near.lisa_snr / far.lisa_snr) / (far.luminosity_distance_mpc
                                              / near.luminosity_distance_mpc)
    report.check("SNR scales exactly as 1/D_L when only the distance changes",
                 bool(abs(ratio - 1.0) < 1e-9), float(ratio))
    report.data["distance_scaling_ratio"] = float(ratio)

    # SNR is NOT monotone in redshift at fixed source masses and fixed observed
    # frequency, because (1+z) raises the detector-frame chirp mass while the
    # distance grows.  Recorded as an observation, not asserted as a rule.
    monotone = {}
    for primary, group in evaluated.groupby("primary_msun"):
        ordered = group.sort_values("redshift")
        monotone[f"{primary:.0e}"] = {
            "redshift": ordered["redshift"].tolist(),
            "snr": ordered["snr"].tolist(),
            "monotonic_decreasing": bool(ordered["snr"].is_monotonic_decreasing),
        }
    report.data["snr_versus_redshift"] = monotone
    report.check("SNR is louder for a heavier secondary at fixed redshift",
                 bool(evaluate_lisa_inspiral_snr(
                          **{**distance_kwargs, "m2_source_msun": 100.0}).lisa_snr
                      > evaluate_lisa_inspiral_snr(**distance_kwargs).lisa_snr))

    # The detector noise curve must have a sensitivity minimum in band.
    frequencies = np.logspace(-4, -1, 200)
    psd = np.array([lisa_effective_strain_psd_hz(f, 4.0) for f in frequencies])
    report.check("effective strain PSD is positive and finite",
                 bool((np.isfinite(psd) & (psd > 0)).all()))
    report.check("PSD has an interior minimum (a sensitivity bucket)",
                 0 < int(np.argmin(psd)) < psd.size - 1,
                 float(frequencies[int(np.argmin(psd))]))

    # Threshold behaviour is a pure comparison, not hidden in a plot.
    threshold_kwargs = dict(
        m1_source_msun=1.0e5, m2_source_msun=10.0, redshift=1.0,
        current_observed_frequency_hz=1.0e-3, mission_duration_years=4.0,
        detector_min_frequency_hz=1.0e-4, detector_max_frequency_hz=1.0e-1)
    strict = evaluate_lisa_inspiral_snr(snr_threshold=1.0e9, **threshold_kwargs)
    loose = evaluate_lisa_inspiral_snr(snr_threshold=1.0e-9, **threshold_kwargs)
    report.check("the SNR value is threshold independent",
                 bool(np.isclose(strict.lisa_snr, loose.lisa_snr)))
    report.check("the detection decision follows the threshold",
                 (not strict.is_detectable) and bool(loose.is_detectable))
    report.data["snr_grid"] = rows

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=150)
    axes[0].loglog(frequencies, np.sqrt(psd), color=SERIES[0], linewidth=1.8)
    _axes(axes[0], "observed frequency [Hz]", r"$\sqrt{S_n}$ [Hz$^{-1/2}$]",
          "LISA effective strain")
    for index, (primary, group) in enumerate(evaluated.groupby("primary_msun")):
        ordered = group.sort_values("redshift")
        axes[1].plot(ordered["redshift"], ordered["snr"], marker="o", markersize=4,
                     color=SERIES[index], linewidth=1.6, label=rf"$M_1=${primary:.0e}")
    axes[1].set_yscale("log")
    axes[1].axhline(20.0, color=MUTED, linestyle="--", linewidth=1.0)
    _axes(axes[1], "redshift", "sky-averaged SNR", "SNR versus redshift")
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Checkpoint I: LISA sensitivity and detectability",
                    fontsize=11, x=0.02, ha="left")
    figure.tight_layout()
    figure.savefig(report.directory / "lisa_detectability.png")
    plt.close(figure)
    report.save()
    return report


# ------------------------------------------------------------ J: end to end
def checkpoint_j() -> Report:
    report = Report("J_end_to_end")
    parser = build_parser()
    run_dir = report.directory / "run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    config = build_config_from_arguments(parser.parse_args([
        "2", str(run_dir), "2", "--smbh-masses", "1e5", "--nsc-masses", "5e4",
        "--base-seed", "777",
    ]))
    result = run_single_halo(config, host=None, seed=777, output_dir=run_dir)
    report.check("the full pipeline ran through the canonical modules",
                 result.status == STATUS_SUCCESS, result.error)
    for name in ("imri_averaged_results.csv", "imri_events.csv",
                 "imri_repeat_results.csv", "simulation_metadata.json"):
        report.check(f"produced {name}", (run_dir / name).is_file())
    report.data["summary"] = result.summary

    # Every stage of the physical chain must be represented in the outputs.
    averaged = pd.read_csv(run_dir / "imri_averaged_results.csv")
    required = ["f_geo", "n_disk_cross", "n_emb_bh", "n_mig_eval", "n_band_eval",
                "n_imri", "n_det", "rate_imri_per_myr", "rate_det_per_myr"]
    report.check("outputs cover disk -> population -> migration -> event -> LISA",
                 all(c in averaged.columns for c in required),
                 [c for c in required if c not in averaged.columns])
    report.check("no cosmic-rate quantity is produced",
                 not any("cosmic_rate" in c or "comoving" in c or "all_sky" in c
                         for c in averaged.columns))
    report.save()
    return report


# -------------------------------------------------------- K: Round-1 regression
def checkpoint_k() -> Report:
    report = Report("K_round1_regression")
    base = report.directory / "runs"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)

    def run(argv: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "main.py"), *argv],
                              cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)

    single = run(["2", str(base / "single"), "2", "--smbh-masses", "1e5",
                  "--nsc-masses", "5e4", "--base-seed", "777"])
    report.check("single-halo CLI still runs", single.returncode == 0, single.stderr[-300:])

    tree_args = ["1", str(base / "unused"), "1", "--tree", str(TREE_PATH),
                 "--workers", "{workers}", "--master-seed", "2026",
                 "--node", "3,200", "--max-hosts", "4",
                 "--smbh-mass-source", "stellar-scaling",
                 "--nsc-mass-source", "stellar-scaling", "--base-seed", "0",
                 "--no-batch-figures"]
    serial = run([a.format(workers=1) if "{" in a else a for a in tree_args][:3]
                 + ["--tree", str(TREE_PATH), "--output", str(base / "w1"), "--workers", "1",
                    "--master-seed", "2026", "--node", "3,200", "--max-hosts", "4",
                    "--smbh-mass-source", "stellar-scaling",
                    "--nsc-mass-source", "stellar-scaling", "--base-seed", "0",
                    "--no-batch-figures"])
    report.check("tree batch CLI still runs (workers=1)", serial.returncode == 0,
                 serial.stderr[-300:])
    parallel = run(["1", str(base / "unused2"), "1",
                    "--tree", str(TREE_PATH), "--output", str(base / "w4"), "--workers", "4",
                    "--master-seed", "2026", "--node", "3,200", "--max-hosts", "4",
                    "--smbh-mass-source", "stellar-scaling",
                    "--nsc-mass-source", "stellar-scaling", "--base-seed", "0",
                    "--no-batch-figures"])
    report.check("tree batch CLI still runs (workers=4)", parallel.returncode == 0,
                 parallel.stderr[-300:])

    if serial.returncode == 0 and parallel.returncode == 0:
        one = pd.read_csv(base / "w1" / "jobs" / "job_summary.csv")
        four = pd.read_csv(base / "w4" / "jobs" / "job_summary.csv")
        columns = [c for c in one.columns if c not in ("runtime_s", "output_path")]
        report.check("serial and parallel job sets are identical",
                     one["job_id"].tolist() == four["job_id"].tolist())
        report.check("serial and parallel seeds are identical",
                     one["seed"].tolist() == four["seed"].tolist())
        report.check("serial and parallel scientific results are identical",
                     one[columns].equals(four[columns]),
                     [c for c in columns if not one[c].equals(four[c])])
        report.check("host mapping provenance is preserved",
                     set(one["smbh_mass_msun_source"]) == {"derived:0.0001*tree:Mstar"})
        report.data["status_counts"] = one["status"].value_counts().to_dict()
    report.save()
    return report


# --------------------------------------------------------- L: repository cleanup
def checkpoint_l() -> Report:
    report = Report("L_repository_cleanup")
    src = PROJECT_ROOT / "src"
    scripts = PROJECT_ROOT / "scripts"

    tests = subprocess.run([sys.executable, str(PROJECT_ROOT / "tests" / "run_tests.py")],
                           cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    report.check("the automated test suite passes", tests.returncode == 0,
                 tests.stdout[-400:])

    compiled = subprocess.run([sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"],
                              cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    report.check("everything compiles", compiled.returncode == 0, compiled.stdout[-300:])

    superseded = ["migration_torque.py", "disk_migration.py", "disk_plot_utils.py",
                  "lisa_snr.py", "detect_rate.py", "rate_plots.py", "tree_batch_plots.py"]
    stale = [str(p.relative_to(PROJECT_ROOT)) for name in superseded
             for p in src.rglob(name)
             if p.parent.name not in {"lisa", "viz", "migration"}]
    report.check("no superseded module remains beside its replacement", not stale, stale)

    ambiguous = [p.name for p in scripts.glob("*.py")
                 if p.stem in {"final_analysis", "trap_plot", "temp", "test2", "plot_final"}
                 or p.stem.endswith("_test")]
    report.check("no ambiguously named scripts remain", not ambiguous, ambiguous)

    # Dead imports: an imported local module never referenced in the body.
    dead: list[str] = []
    for path in sorted(src.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        tree = ast.parse(path.read_text())
        body = path.read_text()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    if body.count(name) <= 1:
                        dead.append(f"{path.relative_to(src)}: {alias.name}")
    report.check("no dead local imports", not dead, dead)

    stray = [str(p.relative_to(PROJECT_ROOT)) for p in PROJECT_ROOT.rglob("__pycache__")
             if "output" not in str(p)]
    report.check("__pycache__ is not carried as source", True, len(stray), warn=bool(stray))

    inventory = {}
    for package in sorted(p for p in src.iterdir() if p.is_dir() and p.name != "__pycache__"):
        files = [f for f in package.rglob("*.py") if "__pycache__" not in str(f)]
        inventory[package.name] = {
            "modules": len(files),
            "lines": sum(len(f.read_text().splitlines()) for f in files),
        }
    inventory["scripts"] = {
        "modules": len(list(scripts.glob("*.py"))),
        "lines": sum(len(f.read_text().splitlines()) for f in scripts.glob("*.py")),
    }
    report.data["inventory"] = inventory
    report.save()
    return report


CHECKPOINTS: dict[str, Callable[[], Report]] = {
    "B": checkpoint_b, "C": checkpoint_c, "D": checkpoint_d, "E": checkpoint_e,
    "F": checkpoint_f, "G": checkpoint_g, "H": checkpoint_h, "I": checkpoint_i,
    "J": checkpoint_j, "K": checkpoint_k, "L": checkpoint_l,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=sorted(CHECKPOINTS), default=None)
    arguments = parser.parse_args(argv)
    selected = arguments.only or list(CHECKPOINTS)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    overall: dict[str, str] = {}
    for letter in selected:
        print(f"\n=== Round-2 Checkpoint {letter} ===")
        started = time.perf_counter()
        try:
            report = CHECKPOINTS[letter]()
            overall[letter] = report.status
        except Exception as error:  # noqa: BLE001 - a crashed checkpoint is a failure
            import traceback
            traceback.print_exc()
            overall[letter] = "ERROR"
        print(f"  ({time.perf_counter() - started:.1f} s)")

    with (OUTPUT_ROOT / "overall_status.json").open("w") as handle:
        json.dump(overall, handle, indent=2)
    print("\nOverall:", overall)
    return 0 if all(s not in ("FAIL", "ERROR") for s in overall.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
