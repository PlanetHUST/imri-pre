"""Physics workflow for one dynamical TDE simulation realization."""

from typing import Sequence

import numpy as np

import physical_process.constants as const
import disk_quantity.disk_solve as disk
import physical_process.general_class as agn
import physical_process.helper_function as helpers
import physical_process.sampling as stellar_sampling
from disk_quantity.pagn_disk_adapter import build_pagn_disk_profile
from disk_quantity.migration_torque import (
    MigrationParameters,
    compute_torque_profile,
)

from .config import (
    DISK_ACCRETION_RATE_G_S_PER_MSUN,
    DISK_INNER_RADIUS_RG,
    MIN_RELAXATION_TIME_MYR,
    YEARS_PER_MYR,
    TDEDynConfig,
)
from .models import (
    BoolArray,
    DampingResult,
    DampingTask,
    DiskProfiles,
    FloatArray,
    GeometricPopulation,
    IntermediatePopulation,
    StellarPopulation,
    TDEOrbitalParams,
    TimescaleRecord,
    TDEDynSingleResult,
)
from .runtime import vprint
from .workers import get_hpc_cpu_count, run_damping_tasks_parallel


def _vprint_population_ranges(
    label: str,
    mass_msun: FloatArray,
    formation_time_myr: FloatArray,
    semi_major_axis_cm: FloatArray,
    eccentricity: FloatArray,
    inclination_rad: FloatArray,
) -> None:
    """Print compact ranges for sampled or selected initial conditions."""
    if len(mass_msun) == 0:
        vprint(f"{label}: no stars")
        return
    vprint(
        f"{label} mass range is " f"{mass_msun.min():.3e} to {mass_msun.max():.3e} Msun"
    )
    vprint(
        f"{label} formation time range is "
        f"{formation_time_myr.min():.3e} to {formation_time_myr.max():.3e} Myr"
    )
    vprint(
        f"{label} sma range is "
        f"{semi_major_axis_cm.min():.3e} to {semi_major_axis_cm.max():.3e} cm"
    )
    vprint(
        f"{label} eccentricity range is "
        f"{eccentricity.min():.3e} to {eccentricity.max():.3e}"
    )
    vprint(
        f"{label} inclination range is "
        f"{inclination_rad.min():.3e} to {inclination_rad.max():.3e} rad"
    )


def _intermediate_population_from_arrays(
    mass_msun: FloatArray,
    formation_time_myr: FloatArray,
    semi_major_axis_cm: FloatArray,
    eccentricity: FloatArray,
    inclination_rad: FloatArray,
) -> IntermediatePopulation:
    return {
        "mass_msun": np.asarray(mass_msun, dtype=float),
        "formation_time_myr": np.asarray(formation_time_myr, dtype=float),
        "semi_major_axis_cm": np.asarray(semi_major_axis_cm, dtype=float),
        "eccentricity": np.asarray(eccentricity, dtype=float),
        "inclination_rad": np.asarray(inclination_rad, dtype=float),
    }


class TDEDynEngine:
    """Dynamical TDE engine with geometric selection and resupply."""

    def __init__(self, config: TDEDynConfig) -> None:
        self.config = config
        self.duty_cycle_myr = config.duty_cycle_myr
        self.disk_cache: dict[float, DiskProfiles] = {}

    def estimate_nsc_radius_cm(self, m_nsc_msun: float) -> float:
        """Estimate NSC radius [cm] from NSC mass [Msun]."""
        nsc_radius_pc = (
            self.config.nsc_radius_pc_at_reference_mass
            * (m_nsc_msun / self.config.nsc_radius_reference_mass_msun)
            ** self.config.nsc_radius_mass_slope
        )
        return float(nsc_radius_pc * const.parsec)

    def get_disk_profiles(self, m_smbh_msun: float) -> DiskProfiles:
        """Return cached disk profiles for an SMBH mass [Msun]."""
        if m_smbh_msun not in self.disk_cache:
            accretion_rate_g_s = DISK_ACCRETION_RATE_G_S_PER_MSUN * m_smbh_msun
            (
                outer_radius_cm,
                radius_cm,
                _temperature_k,
                _toomre_q,
                surface_density_g_cm2,
                _optical_depth,
                aspect_ratio,
                _sound_speed_cm_s,
                _density_g_cm3,
            ) = disk.radial_structure(m_smbh_msun, accretion_rate_g_s)
            gravitational_radius_cm = const.G * m_smbh_msun * const.msun / const.c**2
            self.disk_cache[m_smbh_msun] = {
                "inner_radius_cm": float(
                    DISK_INNER_RADIUS_RG * gravitational_radius_cm
                ),
                "outer_radius_cm": float(outer_radius_cm),
                "radius_cm": np.asarray(radius_cm, dtype=float),
                "surface_density_g_cm2": np.asarray(surface_density_g_cm2, dtype=float),
                "aspect_ratio": np.asarray(aspect_ratio, dtype=float),
                "gravitational_radius_cm": float(gravitational_radius_cm),
            }
            try:
                pagn_alpha = 0.01
                pagn_disk = build_pagn_disk_profile(
                    m_smbh_msun,
                    mdot_cgs=accretion_rate_g_s,
                    alpha=pagn_alpha,
                    eddington_ratio=0.1,
                )
                self.disk_cache[m_smbh_msun]["torque_disk"] = pagn_disk.disk
                self.disk_cache[m_smbh_msun]["torque_alpha"] = pagn_alpha
                vprint(
                    f"  cached pAGN disk for torque integration at "
                    f"M_SMBH={m_smbh_msun:.3e} Msun"
                )
            except Exception:
                vprint(
                    f"  pAGN torque unavailable for M_SMBH={m_smbh_msun:.3e} Msun; "
                    f"falling back to disk_solve migration"
                )
            vprint(f"  cached disk for M_SMBH={m_smbh_msun:.3e} Msun")
        return self.disk_cache[m_smbh_msun]

    def migration_time_myr(
        self,
        result: TDEDynSingleResult,
        agn_properties: agn.AGN_properties,
        damping_result: DampingResult,
        tde_radius_cm: float,
        radius_cm: FloatArray,
        surface_density_g_cm2: FloatArray,
        aspect_ratio: FloatArray,
        disk_data: DiskProfiles | None,
    ) -> float:
        """Return the migration time from either torque integration or fallback."""
        if disk_data is not None and "torque_disk" in disk_data:
            torque = compute_torque_profile(
                disk_data["torque_disk"],
                migrator_mass_msun=damping_result["mass_msun"],
                params=MigrationParameters(
                    alpha=float(disk_data.get("torque_alpha", 0.01)),
                    include_type_ii=self.config.consider_type_ii_migration,
                ),
            )
            return helpers.t_migration_from_torque(
                torque.radius_cm,
                np.asarray(torque.gamma_tot, dtype=float) * 1.0e7,
                damping_result["mass_g"],
                damping_result["initial_semi_major_axis_cm"],
                tde_radius_cm,
                result.m_smbh_msun,
            )

        return helpers.t_migration(
            agn_properties,
            damping_result["mass_g"],
            damping_result["initial_semi_major_axis_cm"],
            h_list=aspect_ratio,
            sigma_list=surface_density_g_cm2,
            r_list=radius_cm,
            consider_type_ii=self.config.consider_type_ii_migration,
        )

    def select_disk_crossing_stars(
        self,
        semi_major_axis_cm: FloatArray,
        eccentricity: FloatArray,
        cos_omega: FloatArray,
        disk_inner_radius_cm: float,
        disk_outer_radius_cm: float,
    ) -> tuple[BoolArray, FloatArray, FloatArray]:
        """Select disk-crossing orbits and return node radii [cm]."""
        inner_node_radius_cm = (
            semi_major_axis_cm
            * (1.0 - eccentricity**2)
            / (1.0 + eccentricity * cos_omega)
        )
        outer_node_radius_cm = (
            semi_major_axis_cm
            * (1.0 - eccentricity**2)
            / (1.0 - eccentricity * cos_omega)
        )
        if self.config.use_disk_inner_radius:
            mask = (inner_node_radius_cm < disk_outer_radius_cm) & (
                outer_node_radius_cm > disk_inner_radius_cm
            )
        else:
            mask = inner_node_radius_cm < disk_outer_radius_cm
        return mask, inner_node_radius_cm, outer_node_radius_cm

    def sample_full_population(
        self, m_smbh_msun: float, m_nsc_msun: float
    ) -> StellarPopulation:
        """Sample stellar masses, lifetimes, and orbital elements."""
        stellar_mass_msun, stellar_lifetime_yr = stellar_sampling.sample_efficient(
            m_nsc_msun, Z=self.config.metallicity
        )

        vprint(
            "sampled stellar mass range is "
            f"{stellar_mass_msun.min():.3e} to {stellar_mass_msun.max():.3e} Msun"
        )

        nsc_radius_cm = self.estimate_nsc_radius_cm(m_nsc_msun)
        (
            semi_major_axis_cm,
            eccentricity,
            inclination_rad,
            cos_omega,
        ) = helpers.initial_star(
            len(stellar_mass_msun),
            m_smbh_msun,
            rmax=nsc_radius_cm,
            gamma=self.config.initial_stellar_density_gamma,
        )
        return {
            "mass_msun": np.asarray(stellar_mass_msun, dtype=float),
            "formation_time_myr": np.asarray(
                stellar_lifetime_yr / YEARS_PER_MYR, dtype=float
            ),
            "semi_major_axis_cm": np.asarray(semi_major_axis_cm, dtype=float),
            "eccentricity": np.asarray(eccentricity, dtype=float),
            "inclination_rad": np.asarray(inclination_rad, dtype=float),
            "cos_omega": np.asarray(cos_omega, dtype=float),
            "nsc_radius_cm": nsc_radius_cm,
        }

    def build_primary_tasks(
        self, geo_pool: GeometricPopulation, flags: agn.damp_flag
    ) -> list[DampingTask]:
        """Build damping tasks for the initial disk-crossing population."""
        tasks: list[DampingTask] = []
        for index in range(len(geo_pool["mass_msun"])):
            tasks.append(
                {
                    "task_kind": "primary",
                    "idx": index,
                    "origin_primary_idx": -1,
                    "copy_number": -1,
                    "mass_g": float(geo_pool["mass_msun"][index] * const.msun),
                    "semi_major_axis_cm": float(geo_pool["semi_major_axis_cm"][index]),
                    "eccentricity": float(geo_pool["eccentricity"][index]),
                    "inclination_rad": float(geo_pool["inclination_rad"][index]),
                    "cos_omega": float(geo_pool["cos_omega"][index]),
                    "formation_time_myr": float(geo_pool["formation_time_myr"][index]),
                    "state_time_s": 0.0,
                    "type_flag": int(flags.type_flag[index]),
                }
            )
        return tasks

    def build_copy_tasks(
        self,
        agn_properties: agn.AGN_properties,
        primary_results: Sequence[DampingResult],
        geo_pool: GeometricPopulation,
        relaxation_time_myr_geo: FloatArray,
    ) -> list[DampingTask]:
        """Build relaxation-driven resupply tasks."""
        if not self.config.dyn_resupply:
            return []
        population_size = len(geo_pool["mass_msun"])
        if population_size == 0:
            return []

        tasks: list[DampingTask] = []
        tde_radius_cm = self.config.tde_radius_rg * agn_properties.rg
        for result in primary_results:
            if result["obj_type"] not in (0, 1):
                continue
            if result["final_semi_major_axis_cm"] <= tde_radius_cm:
                continue

            relaxation_time_myr = max(
                float(relaxation_time_myr_geo[result["idx"]]),
                MIN_RELAXATION_TIME_MYR,
            )
            remaining_time_myr = max(
                0.0,
                agn_properties.AGNlifetime - result["final_time_myr"],
            )
            copy_count = min(
                int(np.ceil(remaining_time_myr / relaxation_time_myr)),
                self.config.max_copies_per_object,
            )

            for copy_number in range(copy_count):
                sampled_index = int(np.random.randint(0, population_size))
                if self.config.reuse_same_orbit_for_copies:
                    semi_major_axis_cm = result["initial_semi_major_axis_cm"]
                    eccentricity = result["eccentricity"]
                    inclination_rad = result["inclination_rad"]
                    cos_omega = result["cos_omega"]
                else:
                    semi_major_axis_cm = float(
                        geo_pool["semi_major_axis_cm"][sampled_index]
                    )
                    eccentricity = float(geo_pool["eccentricity"][sampled_index])
                    inclination_rad = float(geo_pool["inclination_rad"][sampled_index])
                    cos_omega = float(geo_pool["cos_omega"][sampled_index])
                time_offset_myr = copy_number * relaxation_time_myr
                tasks.append(
                    {
                        "task_kind": "copy",
                        "idx": sampled_index,
                        "origin_primary_idx": result["idx"],
                        "copy_number": copy_number,
                        "mass_g": float(
                            geo_pool["mass_msun"][sampled_index] * const.msun
                        ),
                        "semi_major_axis_cm": semi_major_axis_cm,
                        "eccentricity": eccentricity,
                        "inclination_rad": inclination_rad,
                        "cos_omega": cos_omega,
                        "formation_time_myr": float(
                            geo_pool["formation_time_myr"][sampled_index]
                            + time_offset_myr
                        ),
                        "state_time_s": time_offset_myr * const.Myr,
                        "type_flag": 0,
                    }
                )
        return tasks

    def _record_primary_tde(
        self,
        result: TDEDynSingleResult,
        damping_result: DampingResult,
        event_time_myr: float,
    ) -> None:
        result.tde_star_count_primary += 1
        result.tde_mass_primary_msun += damping_result["mass_msun"]
        result.tde_masses_primary_msun.append(damping_result["mass_msun"])
        result.tde_times_primary_myr.append(float(event_time_myr))
        self._record_tde_orbital_params(result, damping_result)

    def _record_copy_tde(
        self,
        result: TDEDynSingleResult,
        damping_result: DampingResult,
        event_time_myr: float,
    ) -> None:
        result.num_copy_tdes += 1
        result.copy_tde_mass_msun += damping_result["mass_msun"]
        result.tde_masses_copies_msun.append(damping_result["mass_msun"])
        result.tde_times_copies_myr.append(float(event_time_myr))
        self._record_tde_orbital_params(result, damping_result)

    @staticmethod
    def _record_tde_orbital_params(
        result: TDEDynSingleResult,
        damping_result: DampingResult,
    ) -> None:
        orbital_params: TDEOrbitalParams = {
            "task_kind": damping_result["task_kind"],
            "semi_major_axis_cm": damping_result["initial_semi_major_axis_cm"],
            "eccentricity": damping_result["eccentricity"],
            "inclination_rad": damping_result["inclination_rad"],
            "mass_msun": damping_result["mass_msun"],
        }
        result.tde_orbital_params.append(orbital_params)

    def _append_timescale_record(
        self,
        result: TDEDynSingleResult,
        damping_result: DampingResult,
        relaxation_time_myr: float,
        migration_time_myr: float,
        outcome: str,
        counted_as_tde: bool,
        tde_mode: str,
    ) -> None:
        """Store one per-object timescale diagnostic row."""
        delivery_time_myr = damping_result["damping_time_myr"]
        if np.isfinite(migration_time_myr):
            delivery_time_myr += migration_time_myr
        record: TimescaleRecord = {
            "m_smbh_msun": result.m_smbh_msun,
            "m_nsc_msun": result.m_nsc_msun,
            "repeat": result.repeat_idx,
            "task_kind": damping_result["task_kind"],
            "object_index": damping_result["idx"],
            "origin_primary_index": damping_result["origin_primary_idx"],
            "copy_number": damping_result["copy_number"],
            "object_type_flag": damping_result["obj_type"],
            "mass_msun": damping_result["mass_msun"],
            "initial_semi_major_axis_cm": damping_result["initial_semi_major_axis_cm"],
            "final_semi_major_axis_cm": damping_result["final_semi_major_axis_cm"],
            "eccentricity": damping_result["eccentricity"],
            "inclination_rad": damping_result["inclination_rad"],
            "state_time_myr": damping_result["state_time_myr"],
            "formation_time_myr": damping_result["formation_time_myr"],
            "damping_time_myr": damping_result["damping_time_myr"],
            "migration_time_myr": migration_time_myr,
            "relaxation_time_myr": relaxation_time_myr,
            "delivery_time_myr": delivery_time_myr,
            "outcome": outcome,
            "counted_as_tde": counted_as_tde,
            "tde_mode": tde_mode,
        }
        result.timescale_records.append(record)

    def classify_primary_and_copies(
        self,
        result: TDEDynSingleResult,
        agn_properties: agn.AGN_properties,
        primary_results: Sequence[DampingResult],
        copies: Sequence[DampingResult],
        relaxation_time_myr_geo: FloatArray,
        radius_cm: FloatArray,
        surface_density_g_cm2: FloatArray,
        aspect_ratio: FloatArray,
        disk_data: DiskProfiles | None = None,
    ) -> None:
        """Classify direct and migration TDEs and accumulate their masses."""
        tde_radius_cm = self.config.tde_radius_rg * agn_properties.rg
        embedded_stars_primary = 0
        embedded_stars_copies = 0
        embedded_bh = 0
        outside = 0
        type_ii_migration_count = 0
        type_ii_migration_mass_msun = 0.0
        min_type_ii_threshold_msun = np.inf

        for damping_result in primary_results:
            relaxation_time_myr = float(relaxation_time_myr_geo[damping_result["idx"]])
            if damping_result["obj_type"] == 0:
                if damping_result["final_semi_major_axis_cm"] < tde_radius_cm:
                    if self.config.direct_tde:
                        result.direct_tde_primary_count += 1
                        self._record_primary_tde(
                            result,
                            damping_result,
                            damping_result["final_time_myr"],
                        )
                        outcome = "direct_tde"
                        counted_as_tde = True
                        tde_mode = "direct"
                    else:
                        outcome = "inside_tde_radius_not_counted"
                        counted_as_tde = False
                        tde_mode = "none"
                    self._append_timescale_record(
                        result,
                        damping_result,
                        relaxation_time_myr,
                        np.nan,
                        outcome,
                        counted_as_tde,
                        tde_mode,
                    )
                else:
                    embedded_stars_primary += 1
                    if self.config.consider_type_ii_migration:
                        type_ii_threshold_msun = helpers.type_ii_threshold_mass_msun(
                            agn_properties,
                            damping_result["initial_semi_major_axis_cm"],
                            h_list=aspect_ratio,
                            r_list=radius_cm,
                        )
                        min_type_ii_threshold_msun = min(
                            min_type_ii_threshold_msun,
                            type_ii_threshold_msun,
                        )
                        gap_parameter = helpers.type_ii_gap_parameter(
                            agn_properties,
                            damping_result["mass_g"],
                            damping_result["initial_semi_major_axis_cm"],
                            h_list=aspect_ratio,
                            r_list=radius_cm,
                        )
                        if gap_parameter > helpers.TYPE_II_GAP_PARAMETER_THRESHOLD:
                            type_ii_migration_count += 1
                            type_ii_migration_mass_msun += damping_result["mass_msun"]
                    migration_time_myr = self.migration_time_myr(
                        result,
                        agn_properties,
                        damping_result,
                        tde_radius_cm,
                        radius_cm,
                        surface_density_g_cm2,
                        aspect_ratio,
                        disk_data,
                    )
                    event_time_myr = (
                        migration_time_myr + damping_result["final_time_myr"]
                    )
                    if event_time_myr < agn_properties.AGNlifetime:
                        result.migration_tde_primary_count += 1
                        self._record_primary_tde(result, damping_result, event_time_myr)
                        outcome = "migration_tde"
                        counted_as_tde = True
                        tde_mode = "migration"
                    else:
                        outcome = "embedded_star"
                        counted_as_tde = False
                        tde_mode = "none"
                    self._append_timescale_record(
                        result,
                        damping_result,
                        relaxation_time_myr,
                        float(migration_time_myr),
                        outcome,
                        counted_as_tde,
                        tde_mode,
                    )
            elif damping_result["obj_type"] == 1:
                if damping_result["final_semi_major_axis_cm"] >= tde_radius_cm:
                    embedded_bh += 1
                self._append_timescale_record(
                    result,
                    damping_result,
                    relaxation_time_myr,
                    np.nan,
                    "embedded_bh",
                    False,
                    "none",
                )
            else:
                outside += 1
                self._append_timescale_record(
                    result,
                    damping_result,
                    relaxation_time_myr,
                    np.nan,
                    (
                        "damping_failed"
                        if damping_result["obj_type"] == -1
                        else "outside_disk"
                    ),
                    False,
                    "none",
                )

        result.embedded_stars_primary = embedded_stars_primary
        result.embedded_bh_primary = embedded_bh
        result.outside_primary = outside
        result.num_copies_generated = len(copies)

        for damping_result in copies:
            relaxation_time_myr = float(
                relaxation_time_myr_geo[damping_result["origin_primary_idx"]]
            )
            if damping_result["obj_type"] != 0:
                self._append_timescale_record(
                    result,
                    damping_result,
                    relaxation_time_myr,
                    np.nan,
                    (
                        "damping_failed"
                        if damping_result["obj_type"] == -1
                        else "outside_disk"
                    ),
                    False,
                    "none",
                )
                continue
            if damping_result["final_semi_major_axis_cm"] < tde_radius_cm:
                if self.config.direct_tde:
                    result.direct_tde_copy_count += 1
                    self._record_copy_tde(
                        result,
                        damping_result,
                        damping_result["final_time_myr"],
                    )
                    outcome = "direct_tde"
                    counted_as_tde = True
                    tde_mode = "direct"
                else:
                    outcome = "inside_tde_radius_not_counted"
                    counted_as_tde = False
                    tde_mode = "none"
                self._append_timescale_record(
                    result,
                    damping_result,
                    relaxation_time_myr,
                    np.nan,
                    outcome,
                    counted_as_tde,
                    tde_mode,
                )
            else:
                embedded_stars_copies += 1
                if self.config.consider_type_ii_migration:
                    type_ii_threshold_msun = helpers.type_ii_threshold_mass_msun(
                        agn_properties,
                        damping_result["initial_semi_major_axis_cm"],
                        h_list=aspect_ratio,
                        r_list=radius_cm,
                    )
                    min_type_ii_threshold_msun = min(
                        min_type_ii_threshold_msun,
                        type_ii_threshold_msun,
                    )
                    gap_parameter = helpers.type_ii_gap_parameter(
                        agn_properties,
                        damping_result["mass_g"],
                        damping_result["initial_semi_major_axis_cm"],
                        h_list=aspect_ratio,
                        r_list=radius_cm,
                    )
                    if gap_parameter > helpers.TYPE_II_GAP_PARAMETER_THRESHOLD:
                        type_ii_migration_count += 1
                        type_ii_migration_mass_msun += damping_result["mass_msun"]
                migration_time_myr = self.migration_time_myr(
                    result,
                    agn_properties,
                    damping_result,
                    tde_radius_cm,
                    radius_cm,
                    surface_density_g_cm2,
                    aspect_ratio,
                    disk_data,
                )
                event_time_myr = migration_time_myr + damping_result["final_time_myr"]
                if event_time_myr < agn_properties.AGNlifetime:
                    result.migration_tde_copy_count += 1
                    self._record_copy_tde(result, damping_result, event_time_myr)
                    outcome = "migration_tde"
                    counted_as_tde = True
                    tde_mode = "migration"
                else:
                    outcome = "embedded_star"
                    counted_as_tde = False
                    tde_mode = "none"
                self._append_timescale_record(
                    result,
                    damping_result,
                    relaxation_time_myr,
                    float(migration_time_myr),
                    outcome,
                    counted_as_tde,
                    tde_mode,
                )

        result.tde_star_count_total = (
            result.tde_star_count_primary + result.num_copy_tdes
        )
        result.tde_mass_total_msun = (
            result.tde_mass_primary_msun + result.copy_tde_mass_msun
        )
        result.tde_mass_rate_total_msun_per_myr = (
            result.tde_mass_total_msun / self.duty_cycle_myr
        )
        vprint(f" embedded stars primary: {embedded_stars_primary}")
        vprint(f" embedded stars copies: {embedded_stars_copies}")
        vprint(f" tde masses primary: " f"{result.tde_mass_primary_msun:.3e} Msun")
        vprint(f" tde masses copies: " f"{result.copy_tde_mass_msun:.3e} Msun")
        if not self.config.consider_type_ii_migration:
            vprint(" type-II migration: disabled; all stars use type-I migration")
        elif np.isfinite(min_type_ii_threshold_msun):
            vprint(
                " type-II migration minimum threshold mass: "
                f"{min_type_ii_threshold_msun:.3e} Msun"
            )
        else:
            vprint(" type-II migration minimum threshold mass: no embedded stars")
        vprint(
            " type-II migration influenced mass: "
            f"{type_ii_migration_mass_msun:.3e} Msun "
            f"({type_ii_migration_count} stars)"
        )

    def run_single_simulation(
        self, m_smbh_msun: float, m_nsc_msun: float, repeat_idx: int
    ) -> TDEDynSingleResult:
        """Run one Monte Carlo realization."""
        result = TDEDynSingleResult(m_smbh_msun, m_nsc_msun, repeat_idx)
        cpu_count = (
            self.config.damping_cpu_count
            if self.config.damping_cpu_count is not None
            else get_hpc_cpu_count()
        )
        agn_properties = agn.AGN_properties()
        agn_properties.init_AGN(m_smbh_msun, self.duty_cycle_myr)
        vprint("AGN initialized done.")

        disk_data = self.get_disk_profiles(m_smbh_msun)
        result.disk_inner_radius_cm = disk_data["inner_radius_cm"]
        result.disk_outer_radius_cm = disk_data["outer_radius_cm"]
        vprint(
            "Disk profiles cached. "
            f"r_out = {disk_data['outer_radius_cm']:.3e} cm = "
            f"{disk_data['outer_radius_cm']/const.parsec:.3e} pc"
        )

        population = self.sample_full_population(m_smbh_msun, m_nsc_msun)
        result.nsc_radius_cm = population["nsc_radius_cm"]
        result.total_stars = len(population["mass_msun"])
        mask, _, _ = self.select_disk_crossing_stars(
            population["semi_major_axis_cm"],
            population["eccentricity"],
            population["cos_omega"],
            disk_data["inner_radius_cm"],
            disk_data["outer_radius_cm"],
        )
        _vprint_population_ranges(
            "initial population",
            population["mass_msun"],
            population["formation_time_myr"],
            population["semi_major_axis_cm"],
            population["eccentricity"],
            population["inclination_rad"],
        )
        if self.config.create_intermediate_plots:
            result.intermediate_initial_population = (
                _intermediate_population_from_arrays(
                    population["mass_msun"],
                    population["formation_time_myr"],
                    population["semi_major_axis_cm"],
                    population["eccentricity"],
                    population["inclination_rad"],
                )
            )
        selected_indices = np.where(mask)[0]
        result.geo_selected_stars = len(selected_indices)
        result.f_geo = (
            len(selected_indices) / result.total_stars
            if result.total_stars > 0
            else 0.0
        )
        _vprint_population_ranges(
            "disk-crossing population",
            population["mass_msun"][selected_indices],
            population["formation_time_myr"][selected_indices],
            population["semi_major_axis_cm"][selected_indices],
            population["eccentricity"][selected_indices],
            population["inclination_rad"][selected_indices],
        )
        if self.config.create_intermediate_plots:
            result.intermediate_disk_crossing_population = (
                _intermediate_population_from_arrays(
                    population["mass_msun"][selected_indices],
                    population["formation_time_myr"][selected_indices],
                    population["semi_major_axis_cm"][selected_indices],
                    population["eccentricity"][selected_indices],
                    population["inclination_rad"][selected_indices],
                )
            )
        if len(selected_indices) == 0:
            return result
        vprint(
            f"Geometry selection: {len(selected_indices)}/"
            f"{result.total_stars} stars selected "
            f"(f_geo={result.f_geo:.3e})"
        )

        geo_pool: GeometricPopulation = {
            "mass_msun": population["mass_msun"][selected_indices],
            "formation_time_myr": population["formation_time_myr"][selected_indices],
            "semi_major_axis_cm": population["semi_major_axis_cm"][selected_indices],
            "eccentricity": population["eccentricity"][selected_indices],
            "inclination_rad": population["inclination_rad"][selected_indices],
            "cos_omega": population["cos_omega"][selected_indices],
        }
        relaxation_time_myr_geo = (
            np.asarray(
                [
                    helpers.t_relax(agn_properties.M_SMBH, orbit_radius_cm)
                    for orbit_radius_cm in geo_pool["semi_major_axis_cm"]
                ],
                dtype=float,
            )
            / const.Myr
        )
        result.mean_relaxation_time_myr_geo = float(np.mean(relaxation_time_myr_geo))
        vprint(
            "Relaxation times initialized, mean_trel_myr_geo = "
            f"{result.mean_relaxation_time_myr_geo:.3e} Myr"
        )

        primary_results = sorted(
            run_damping_tasks_parallel(
                self.build_primary_tasks(
                    geo_pool, agn.damp_flag(len(selected_indices))
                ),
                agn_properties,
                disk_data["radius_cm"],
                disk_data["surface_density_g_cm2"],
                disk_data["aspect_ratio"],
                cpu_count,
            ),
            key=lambda item: item["idx"],
        )
        copy_results = sorted(
            run_damping_tasks_parallel(
                self.build_copy_tasks(
                    agn_properties,
                    primary_results,
                    geo_pool,
                    relaxation_time_myr_geo,
                ),
                agn_properties,
                disk_data["radius_cm"],
                disk_data["surface_density_g_cm2"],
                disk_data["aspect_ratio"],
                cpu_count,
            ),
            key=lambda item: (
                item["origin_primary_idx"],
                item["copy_number"],
            ),
        )
        self.classify_primary_and_copies(
            result,
            agn_properties,
            primary_results,
            copy_results,
            relaxation_time_myr_geo,
            disk_data["radius_cm"],
            disk_data["surface_density_g_cm2"],
            disk_data["aspect_ratio"],
            disk_data,
        )
        return result
