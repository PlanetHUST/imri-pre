"""Physics engine for one dynamical IMRI simulation realization."""

from typing import Sequence

import numpy as np

import disk_quantity.disk_solve as disk
import physical_process.constants as const
import physical_process.general_class as agn
import physical_process.helper_function as helpers
import physical_process.sampling as stellar_sampling
from disk_quantity.pagn_disk_adapter import build_pagn_disk_profile

from .config import (
    DISK_ACCRETION_RATE_G_S_PER_MSUN,
    DISK_INNER_RADIUS_RG,
    IMRIDynConfig,
    YEARS_PER_MYR,
)
from .models import (
    BoolArray,
    DampingResult,
    DampingTask,
    DiskProfiles,
    FloatArray,
    GeometricPopulation,
    IMRIEventRecord,
    IMRIDynSingleResult,
    IntermediatePopulation,
    StellarPopulation,
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
        vprint(f"{label}: no objects")
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


class IMRIDynEngine:
    """Dynamical IMRI engine with geometric disk-crossing selection."""

    def __init__(self, config: IMRIDynConfig) -> None:
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
                    f"  cached pAGN disk for future torque work at "
                    f"M_SMBH={m_smbh_msun:.3e} Msun"
                )
            except Exception:
                vprint(
                    f"  pAGN disk unavailable for M_SMBH={m_smbh_msun:.3e} Msun; "
                    f"using disk_solve migration only"
                )
            vprint(f"  cached disk for M_SMBH={m_smbh_msun:.3e} Msun")
        return self.disk_cache[m_smbh_msun]

    def select_disk_crossing_objects(
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
        """Sample progenitor masses, lifetimes, and orbital elements."""

        mass_msun, lifetime_yr = stellar_sampling.sample_efficient(
            m_nsc_msun, Z=self.config.metallicity
        )
        vprint(
            "sampled progenitor mass range is "
            f"{mass_msun.min():.3e} to {mass_msun.max():.3e} Msun"
        )

        nsc_radius_cm = self.estimate_nsc_radius_cm(m_nsc_msun)

        (
            semi_major_axis_cm,
            eccentricity,
            inclination_rad,
            cos_omega,
        ) = helpers.initial_star(
            len(mass_msun),
            m_smbh_msun,
            rmax=nsc_radius_cm,
            gamma=self.config.initial_stellar_density_gamma,
        )

        return {
            "mass_msun": np.asarray(mass_msun, dtype=float),
            "formation_time_myr": np.asarray(lifetime_yr / YEARS_PER_MYR, dtype=float),
            "semi_major_axis_cm": np.asarray(semi_major_axis_cm, dtype=float),
            "eccentricity": np.asarray(eccentricity, dtype=float),
            "inclination_rad": np.asarray(inclination_rad, dtype=float),
            "cos_omega": np.asarray(cos_omega, dtype=float),
            "nsc_radius_cm": nsc_radius_cm,
        }

    def build_damping_tasks(
        self, geo_pool: GeometricPopulation, flags: agn.damp_flag
    ) -> list[DampingTask]:
        """Build damping tasks for the initial disk-crossing population."""
        tasks: list[DampingTask] = []
        for index in range(len(geo_pool["mass_msun"])):
            tasks.append(
                {
                    "task_kind": "primary",
                    "idx": index,
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

    def run_single_simulation(
        self, m_smbh_msun: float, m_nsc_msun: float, repeat_idx: int
    ) -> IMRIDynSingleResult:
        """Run one Monte Carlo realization and classify IMRI candidates."""
        result = IMRIDynSingleResult(m_smbh_msun, m_nsc_msun, repeat_idx)
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
        result.total_sampled_objects = len(population["mass_msun"])
        mask, _, _ = self.select_disk_crossing_objects(
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
        result.intermediate_initial_population = _intermediate_population_from_arrays(
            population["mass_msun"],
            population["formation_time_myr"],
            population["semi_major_axis_cm"],
            population["eccentricity"],
            population["inclination_rad"],
        )

        selected_indices = np.where(mask)[0]
        result.disk_crossing_objects = len(selected_indices)
        result.f_geo = (
            len(selected_indices) / result.total_sampled_objects
            if result.total_sampled_objects > 0
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
            f"{result.total_sampled_objects} objects selected "
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

        damping_results = sorted(
            run_damping_tasks_parallel(
                self.build_damping_tasks(
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
        self.classify_imri_candidates(
            result, agn_properties, disk_data, damping_results
        )
        return result

    def classify_imri_candidates(
        self,
        result: IMRIDynSingleResult,
        agn_properties: agn.AGN_properties,
        disk_data: DiskProfiles,
        damping_results: Sequence[DampingResult],
    ) -> None:
        """Classify embedded compact objects as IMRI candidates."""
        rg_cm = disk_data["gravitational_radius_cm"]
        for damping_result in damping_results:
            if damping_result["obj_type"] == 0:
                result.embedded_star_count += 1
                continue
            if damping_result["obj_type"] != 1:
                result.outside_or_failed_count += 1
                continue

            result.embedded_bh_count += 1
            event = self._imri_event_record(
                result,
                agn_properties,
                disk_data,
                damping_result,
                rg_cm,
            )
            result.imri_events.append(event)
            if event["bh_candidate"]:
                result.bh_candidate_count += 1
            if event["counted_as_imri"]:
                result.imri_count += 1
                result.imri_mass_msun += event["mass_msun"]

        result.imri_rate_per_myr = result.imri_count / self.duty_cycle_myr
        result.imri_mass_rate_msun_per_myr = result.imri_mass_msun / self.duty_cycle_myr

    def _imri_event_record(
        self,
        result: IMRIDynSingleResult,
        agn_properties: agn.AGN_properties,
        disk_data: DiskProfiles,
        damping_result: DampingResult,
        rg_cm: float,
    ) -> IMRIEventRecord:
        compact_mass_msun = damping_result["mass_msun"]
        start_radius_cm = damping_result["final_semi_major_axis_cm"]
        total_mass_msun = result.m_smbh_msun + compact_mass_msun
        bh_candidate = bool(
            np.isfinite(damping_result["formation_time_myr"])
            and np.isfinite(damping_result["damping_time_myr"])
            and damping_result["formation_time_myr"]
            < damping_result["damping_time_myr"]
        )
        lisa_entry_radius_cm = helpers.lisa_entry_radius_cm(
            result.m_smbh_msun,
            compact_mass_msun,
            self.config.lisa_min_gw_frequency_hz,
        )
        inner_radius_cm = helpers.imri_lisa_hubble_threshold_radius_cm(
            result.m_smbh_msun,
            compact_mass_msun,
            self.config.max_gw_time_myr,
            self.config.lisa_min_gw_frequency_hz,
            self.config.gw_eccentricity,
        )
        gw_time_to_lisa_myr = (
            helpers.peters_time_to_lisa_band_myr(
                result.m_smbh_msun,
                compact_mass_msun,
                inner_radius_cm,
                self.config.lisa_min_gw_frequency_hz,
                self.config.gw_eccentricity,
            )
            if np.isfinite(inner_radius_cm)
            else np.inf
        )
        migration_time_myr = self.semi_analytic_migration_time_to_radius_myr(
            agn_properties,
            disk_data,
            damping_result["mass_g"],
            start_radius_cm,
            inner_radius_cm,
        )
        arrival_time_myr = damping_result["final_time_myr"] + migration_time_myr
        reaches_lisa_in_time = bool(
            np.isfinite(gw_time_to_lisa_myr)
            and gw_time_to_lisa_myr <= self.config.max_gw_time_myr * (1.0 + 1.0e-10)
        )
        reaches_inner_before_duty = bool(
            np.isfinite(arrival_time_myr)
            and arrival_time_myr <= self.config.duty_cycle_myr
        )
        counted_as_imri = bool(
            bh_candidate and reaches_lisa_in_time and reaches_inner_before_duty
        )
        if not bh_candidate:
            outcome = "not_bh_before_damping"
        elif not reaches_lisa_in_time:
            outcome = "no_lisa_entry_within_gw_time"
        elif not np.isfinite(migration_time_myr):
            outcome = "migration_unavailable"
        elif not reaches_inner_before_duty:
            outcome = "misses_duty_cycle"
        else:
            outcome = "imri"

        return {
            "m_smbh_msun": result.m_smbh_msun,
            "m_nsc_msun": result.m_nsc_msun,
            "repeat": result.repeat_idx,
            "object_index": damping_result["idx"],
            "mass_msun": compact_mass_msun,
            "formation_time_myr": damping_result["formation_time_myr"],
            "damping_time_myr": damping_result["damping_time_myr"],
            "bh_embedding_time_myr": damping_result["final_time_myr"],
            "start_radius_cm": start_radius_cm,
            "start_radius_rg": start_radius_cm / rg_cm,
            "lisa_entry_radius_cm": lisa_entry_radius_cm,
            "lisa_entry_radius_rg": lisa_entry_radius_cm / rg_cm,
            "inner_radius_cm": inner_radius_cm,
            "inner_radius_rg": inner_radius_cm / rg_cm,
            "start_gw_frequency_hz": helpers.gw_frequency_hz(
                total_mass_msun,
                start_radius_cm,
            ),
            "inner_gw_frequency_hz": (
                helpers.gw_frequency_hz(total_mass_msun, inner_radius_cm)
                if np.isfinite(inner_radius_cm)
                else np.nan
            ),
            "gw_time_to_lisa_myr": gw_time_to_lisa_myr,
            "migration_time_myr": migration_time_myr,
            "arrival_time_myr": arrival_time_myr,
            "bh_candidate": bh_candidate,
            "counted_as_imri": counted_as_imri,
            "outcome": outcome,
        }

    def semi_analytic_migration_time_to_radius_myr(
        self,
        agn_properties: agn.AGN_properties,
        disk_data: DiskProfiles,
        mass_g: float,
        start_radius_cm: float,
        target_radius_cm: float,
    ) -> float:
        """Approximate migration time [Myr] using the local semi-analytic time."""
        if not np.isfinite(target_radius_cm) or start_radius_cm <= 0.0:
            return np.inf
        if start_radius_cm <= target_radius_cm:
            return 0.0
        local_time_myr = helpers.t_migration(
            agn_properties,
            mass_g,
            start_radius_cm,
            h_list=disk_data["aspect_ratio"],
            sigma_list=disk_data["surface_density_g_cm2"],
            r_list=disk_data["radius_cm"],
            consider_type_ii=self.config.consider_type_ii_migration,
        )
        if not np.isfinite(local_time_myr) or local_time_myr <= 0.0:
            return np.inf
        return float(local_time_myr * abs(np.log(start_radius_cm / target_radius_cm)))
