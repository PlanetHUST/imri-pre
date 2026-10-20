"""Compare IMRI with both cos_tde_revised damping sources on shared inputs.

The reference is read in isolated subprocesses, so its legacy module names
cannot replace IMRI's imports. Native physical results are comparisons, not
parity assertions: the density, crossing geometry and capture rules differ.
A separate controlled experiment holds the derivatives and stopping threshold
fixed to check numerical-method alignment with the newer root-level solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
SMBH_MASS_MSUN = 3000.0
AGN_LIFETIME_MYR = 10.0
NATIVE_MAX_ITER = 5000


def _evaluate(inputs_path: Path, source: Path | None) -> dict:
    data = np.load(inputs_path)
    if source is None:
        sys.path.insert(0, str(SOURCE_ROOT))
        from physical_process import constants as const
        from physical_process import damp
        from physical_process.agn import AGNProperties

        agn = AGNProperties.from_mass(SMBH_MASS_MSUN, AGN_LIFETIME_MYR)
        calculator = damp.DampingCalculator(
            agn, False, data["radius_cm"], data["surface_density_g_cm2"],
            data["aspect_ratio"], float(data["inner_radius_cm"]),
            float(data["outer_radius_cm"]), density_g_cm3=data["density_g_cm3"],
        )
        source_file = Path(damp.__file__)
    else:
        sys.path.insert(0, str(source))
        import constants as const
        import damp
        from general_class import AGN_properties

        agn = AGN_properties()
        agn.init_AGN(SMBH_MASS_MSUN, AGN_LIFETIME_MYR)
        calculator = damp.DampingCalculator(
            agn, False, data["radius_cm"], data["surface_density_g_cm2"],
            data["aspect_ratio"],
        )
        source_file = source / "damp.py"

    rg_cm = const.G * SMBH_MASS_MSUN * const.msun / const.c**2
    rows = []
    cases = (
        ("symmetric_nodes", 1.0e5 * rg_cm, 0.3, 30.0, 0.0),
        ("eccentric_retrograde", 1.0e5 * rg_cm, 0.6, 120.0, 0.6),
        ("below_legacy_8deg", 1.0e5 * rg_cm, 0.3, 5.0, 0.6),
        ("one_node_outside", float(data["outer_radius_cm"]), 0.2, 60.0, 0.6),
    )
    for name, axis_cm, eccentricity, inclination_deg, cos_omega in cases:
        for species, mass_msun, formation_myr in (("Star", 1.0, 100.0), ("BH", 10.0, 0.0)):
            state = damp.OrbitalState(
                axis_cm, eccentricity, np.deg2rad(inclination_deg), cos_omega,
            )
            terms = calculator.timescale_calc.compute_all_timescales(
                mass_msun * const.msun, state, species,
            )
            drag = terms["drag"]
            disk = terms["disk"]
            tau = drag.drag_time_s if source is None else drag.tau
            rho = disk.density_g_cm3 if source is None else disk.rho
            crossing = terms["crossing_time_s"]
            coefficient_args = (state, *tau, *crossing)
            if source is None:
                coefficient_args += (terms["active_node_mask"],)
            coefficients = calculator.compute_ode_coefficients(*coefficient_args)
            semi_s = calculator.damping_semi_analysis(mass_msun * const.msun, state, species)
            arguments = (mass_msun * const.msun, state, formation_myr, species)
            if source is None:
                outcome = calculator.damping_simulation(*arguments, max_iter=NATIVE_MAX_ITER)
                time_s = outcome.elapsed_time_s
                final_axis_cm = outcome.final_state.semi_major_axis_cm
                final_e = outcome.final_state.eccentricity
                final_i = outcome.final_state.inclination_rad
                status = outcome.status.value
            else:
                time_s, final_axis_cm, flag = calculator.damping_simulation(
                    *arguments, 0, max_iter=NATIVE_MAX_ITER,
                )
                final = getattr(calculator, "last_damping_final_state", None)
                final_e = final.e if final is not None else None
                final_i = final.i if final is not None else None
                status = f"legacy_flag_{flag}"
            row = dict(
                case=name, species=species, mass_msun=mass_msun,
                initial_axis_cm=axis_cm, initial_eccentricity=eccentricity,
                initial_inclination_deg=inclination_deg, cos_omega=cos_omega,
                density_1_g_cm3=rho[0], density_2_g_cm3=rho[1],
                crossing_1_s=crossing[0], crossing_2_s=crossing[1],
                orbital_period_s=terms["orbital_period_s"],
                semi_analytical_time_myr=semi_s / const.Myr,
                native_status=status, native_returned_time_myr=time_s / const.Myr,
                native_returned_axis_cm=final_axis_cm,
                native_final_eccentricity=final_e, native_final_inclination_rad=final_i,
            )
            row.update(zip(("da1_cm", "da2_cm", "di1_rad", "di2_rad", "de1", "de2"), coefficients))
            rows.append(row)

    # Identical ODE, constant timescales, common 0.02 rad target. This isolates
    # the step controller from all physical differences above.
    calculator.timescale_calc.compute_all_timescales = lambda *_: {
        "drag": SimpleNamespace(**{"drag_time_s" if source is None else "tau": (1.0, 1.0)}),
        "crossing_time_s": (1.0, 1.0), "orbital_period_s": 1.0,
        "active_node_mask": (True, True),
    }
    calculator.compute_ode_coefficients = lambda *_: (-1.0e16, 0.0, -1.0e5, 0.0, 0.0, 0.0)
    if source is None:
        calculator.inclination_is_damped = lambda state: state.inclination_rad <= 0.02
        outcome = calculator.damping_simulation(
            1.0, damp.OrbitalState(1.0e12, 0.005, 0.2, 0.5), 0.0, "BH", max_iter=1000,
        )
        controlled = [outcome.elapsed_time_s, outcome.final_state.semi_major_axis_cm]
    else:
        calculator.minimum_inclination_rad = 0.02
        controlled = list(calculator.damping_simulation(
            1.0, damp.OrbitalState(1.0e12, 0.005, 0.2, 0.5), 0.0, "BH", 0, max_iter=1000,
        )[:2])
    return {
        "source_file": str(source_file),
        "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "rows": rows, "controlled_time_s_axis_cm": controlled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=Path("/home/planet/Phy/cos_tde_revised"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--inputs-npz", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inputs_npz is not None:
        print(json.dumps(_evaluate(args.inputs_npz, args.worker_source)))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required")
    sys.path.insert(0, str(SOURCE_ROOT))
    import pandas as pd
    from disk_quantity.pagn_disk_adapter import build_pagn_disk_profile
    from workflows.config import DISK_ACCRETION_RATE_G_S_PER_MSUN, PAGN_ALPHA

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile = build_pagn_disk_profile(
        SMBH_MASS_MSUN, PAGN_ALPHA, n_resolution=400,
        accretion_rate_g_s=DISK_ACCRETION_RATE_G_S_PER_MSUN * SMBH_MASS_MSUN,
    )
    inputs = output / "shared_disk.npz"
    np.savez(inputs, **{
        key: getattr(profile, key) for key in (
            "radius_cm", "surface_density_g_cm2", "aspect_ratio", "density_g_cm3",
            "inner_radius_cm", "outer_radius_cm",
        )
    })
    results = {}
    for label, source in (
        ("imri", None), ("tde_root", args.reference_root.resolve()),
        ("tde_pinned", args.reference_root.resolve() / "vendor/july_physics"),
    ):
        command = [sys.executable, str(Path(__file__).resolve()), "--inputs-npz", str(inputs)]
        if source is not None:
            command += ["--worker-source", str(source)]
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        results[label] = json.loads(process.stdout)

    # The root controller and IMRI must evolve the controlled ODE identically;
    # the pinned floor is retained solely as historical comparison evidence.
    np.testing.assert_allclose(
        results["imri"]["controlled_time_s_axis_cm"],
        results["tde_root"]["controlled_time_s_axis_cm"], rtol=1.0e-13,
    )
    assert results["imri"]["controlled_time_s_axis_cm"][0] > 0.0
    assert results["tde_pinned"]["controlled_time_s_axis_cm"][0] == 0.0
    table = pd.DataFrame([
        dict(implementation=label, **row)
        for label, result in results.items() for row in result["rows"]
    ])
    # Common orbital geometry is unchanged. At cos(omega)=0, both nodes have
    # the same radius. The legacy Sigma/H is twice the solved density, with
    # a small r_grid/r_node correction from nearest-cell sampling of H/R.
    for label in ("tde_root", "tde_pinned"):
        common = table.query("implementation == 'imri'").merge(
            table.query("implementation == @label"), on=["case", "species"],
            suffixes=("_imri", "_reference"),
        )
        np.testing.assert_allclose(
            common.orbital_period_s_imri, common.orbital_period_s_reference,
            rtol=1.0e-14,
        )
        symmetric = common.query("case == 'symmetric_nodes'")
        node_cm = (
            symmetric.initial_axis_cm_imri.to_numpy()
            * (1.0 - symmetric.initial_eccentricity_imri.to_numpy()**2)
        )
        grid_cm = profile.radius_cm[
            np.abs(profile.radius_cm[:, None] - node_cm).argmin(axis=0)
        ]
        np.testing.assert_allclose(
            symmetric.density_1_g_cm3_reference,
            2.0 * grid_cm / node_cm * symmetric.density_1_g_cm3_imri,
            rtol=1.0e-14,
        )
    table.to_csv(output / "damping_comparison.csv", index=False, float_format="%.17g")
    summary = {
        "smbh_mass_msun": SMBH_MASS_MSUN, "agn_lifetime_myr": AGN_LIFETIME_MYR,
        "native_max_accepted_steps_or_legacy_trials": NATIVE_MAX_ITER,
        "shared_disk_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "controlled_numerical_parity": True,
        "legacy_floor_stall_reproduced": True,
        "common_orbital_period_parity": True,
        "legacy_density_normalization_difference_verified": True,
        "native_physics_parity_expected": False,
        "sources": {label: {k: v for k, v in result.items() if k != "rows"}
                    for label, result in results.items()},
    }
    (output / "validation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(table[["implementation", "case", "species", "native_status", "native_returned_time_myr"]].to_string(index=False))
    print(f"Controlled numerical parity: passed; artifacts: {output}")


if __name__ == "__main__":
    main()
