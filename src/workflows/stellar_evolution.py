"""Unit-explicit SEVN endpoint and orbital-kick boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from physical_process.damp import OrbitalState
from physical_process.natal_kick import apply_orbital_kick
from physical_process.sampling import SOLAR_METALLICITY


@dataclass(frozen=True)
class StellarEndpoint:
    source_id: int
    zams_mass_msun: float
    metallicity_absolute_z: float
    collapse_time_myr: float
    presupernova_mass_msun: float
    presupernova_phase: int
    remnant_type: int
    remnant_mass_msun: float
    kick_speed_km_s: float
    seed: int

    @property
    def is_black_hole(self) -> bool:
        return self.remnant_type == 6 and self.remnant_mass_msun > 0.0


_SESSION_TABLES: tuple[Path, Path] | None = None
_SESSION_PID: int | None = None


def _ensure_session(hydrogen_tables: Path, helium_tables: Path) -> object:
    """Load SEVN tables once in each worker process."""
    global _SESSION_TABLES, _SESSION_PID
    try:
        from sevnpy.sevn import SEVNmanager
    except ImportError as error:
        raise RuntimeError("SEVN mode requires the sevnpy package") from error
    requested = (hydrogen_tables, helium_tables)
    if (
        _SESSION_PID != os.getpid()
        or _SESSION_TABLES != requested
        or not SEVNmanager.check_initiliased()
    ):
        SEVNmanager.init({"tables": str(hydrogen_tables), "tables_HE": str(helium_tables)})
        _SESSION_TABLES = requested
        _SESSION_PID = os.getpid()
    return SEVNmanager


def evolve_stellar_endpoint(
    *,
    source_id: int,
    zams_mass_msun: float,
    metallicity_zsun: float,
    task_seed: int | None,
    hydrogen_tables: Path,
    helium_tables: Path,
) -> StellarEndpoint:
    """Evolve one sampled source, keeping SEVN's remnant and kick together."""
    manager = _ensure_session(hydrogen_tables, helium_tables)
    from sevnpy.sevn import Star

    metallicity_absolute_z = metallicity_zsun * SOLAR_METALLICITY
    coverage = manager.tables_info()
    if not (
        coverage["min_z"] <= metallicity_absolute_z <= coverage["max_z"]
        and coverage["min_zams"] <= zams_mass_msun <= coverage["max_zams"]
    ):
        raise ValueError(
            "source outside SEVN H-track coverage: "
            f"M_ZAMS={zams_mass_msun:g} Msun, Z={metallicity_absolute_z:g}"
        )
    source_sequence = (
        np.random.SeedSequence([task_seed, source_id, 7212026])
        if task_seed is not None else np.random.SeedSequence()
    )
    seed = int(source_sequence.generate_state(1, dtype=np.uint32)[0])
    star = Star(Mzams=zams_mass_msun, Z=metallicity_absolute_z, rseed=seed)
    star.evolve("end")
    history = star.evolve_table
    if len(history) < 2:
        raise ValueError("SEVN evolution has no pre-remnant snapshot")
    endpoint = history.iloc[-1]
    presupernova = history.iloc[-2]
    collapse_time_myr = float(endpoint["Worldtime"])
    presupernova_mass_msun = float(presupernova["Mass"])
    presupernova_phase = int(presupernova["Phase"])
    remnant_type = int(endpoint["RemnantType"])
    remnant_mass_msun = float(endpoint["Mass"])
    if remnant_type not in (-1, 1, 2, 3, 4, 5, 6):
        raise ValueError(f"SEVN did not finish with a known remnant type: {remnant_type}")
    if not np.isfinite(collapse_time_myr) or collapse_time_myr < 0.0:
        raise ValueError("SEVN produced an invalid collapse time")
    if not np.isfinite(presupernova_mass_msun) or presupernova_mass_msun <= 0.0:
        raise ValueError("SEVN produced an invalid pre-supernova mass")
    # SEVN's pair-instability no-remnant endpoint is type -1 with NaN Mass;
    # its SN log reports a zero gravitational remnant mass.
    if remnant_type == -1:
        remnant_mass_msun = 0.0
    if not np.isfinite(remnant_mass_msun) or remnant_mass_msun < 0.0:
        raise ValueError("SEVN produced an invalid remnant mass")
    raw_kick = np.asarray(star.get_SN_kick()["Vkick"], dtype=float)
    if raw_kick.shape != (3,):
        raise ValueError("SEVN kick must have three Cartesian components")
    if np.all(np.isnan(raw_kick)):
        if remnant_type in (-1, 1, 2, 3):
            kick_speed_km_s = 0.0
        else:
            raise ValueError("SEVN compact remnant has no recorded kick")
    elif np.all(np.isfinite(raw_kick)):
        kick_speed_km_s = float(np.linalg.norm(raw_kick))
    else:
        raise ValueError("SEVN kick has mixed finite and nonfinite components")
    return StellarEndpoint(
        source_id=source_id,
        zams_mass_msun=zams_mass_msun,
        metallicity_absolute_z=metallicity_absolute_z,
        collapse_time_myr=collapse_time_myr,
        presupernova_mass_msun=presupernova_mass_msun,
        presupernova_phase=presupernova_phase,
        remnant_type=remnant_type,
        remnant_mass_msun=remnant_mass_msun,
        kick_speed_km_s=kick_speed_km_s,
        seed=seed,
    )


def apply_sevn_kick(
    state: OrbitalState,
    *,
    kick_speed_km_s: float,
    smbh_mass_msun: float,
    seed: int,
) -> OrbitalState | None:
    """Map SEVN's isotropic kick speed onto the disk-frame orbit."""
    if not np.isfinite(kick_speed_km_s) or kick_speed_km_s < 0.0:
        raise ValueError("kick speed must be finite and nonnegative")
    if kick_speed_km_s == 0.0:
        return state
    rng = np.random.default_rng(np.random.SeedSequence([seed, 412]))
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    kick_velocity_cm_s = tuple(
        float(value * kick_speed_km_s * 1.0e5) for value in direction
    )
    return apply_orbital_kick(state, kick_velocity_cm_s, smbh_mass_msun, rng)
