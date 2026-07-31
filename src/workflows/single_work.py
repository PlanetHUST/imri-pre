"""Single-realization workflow for the dynamical IMRI simulation."""

from typing import Sequence

import numpy as np

import physical_process.constants as const
import physical_process.disk_migration as disk_migration
import physical_process.general_class as agn
import physical_process.gw_frequency_radius as gw_helpers
import physical_process.nsc_orbital as nsc_orbital
import physical_process.sampling as stellar_sampling

from .config import IMRIDynConfig, YEARS_PER_MYR
from .engine import IMRIDynEngine
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
        f"{label} mass range is "
        f"{mass_msun.min():.3e} to {mass_msun.max():.3e} Msun"
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


class IMRISingleWorkflow:
    """Run one Monte Carlo realization using an IMRI engine."""

    def __init__(self, engine: IMRIDynEngine) -> None:
        self.engine = engine
        self.config: IMRIDynConfig = engine.config
        self.duty_cycle_myr = engine.duty_cycle_myr

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

        nsc_radius_cm = self.engine.estimate_nsc_radius_cm(m_nsc_msun)

        (
            semi_major_axis_cm,
            eccentricity,
            inclination_rad,
            cos_omega,
        ) = nsc_orbital.initial_star(
            len(mass_msun),
            m_smbh_msun,
            rmax=nsc_radius_cm,
            gamma=self.config.initial_stellar_density_gamma,
        )

        return {
            "mass_msun": np.asarray(mass_msun, dtype=float),
            "formation_time_myr": np.asarray(
                lifetime_yr / YEARS_PER_MYR,
                dtype=float,
            ),
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
                    "semi_major_axis_cm": float(
                        geo_pool["semi_major_axis_cm"][index]
                    ),
                    "eccentricity": float(geo_pool["eccentricity"][index]),
                    "inclination_rad": float(geo_pool["inclination_rad"][index]),
                    "cos_omega": float(geo_pool["cos_omega"][index]),
                    "formation_time_myr": float(
                        geo_pool["formation_time_myr"][index]
                    ),
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

        disk_data = self.engine.get_disk_profiles(m_smbh_msun)
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
                    nsc_orbital.t_relax(agn_properties.M_SMBH, orbit_radius_cm)
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
        redshift = self.config.redshift
        redshifted_total_mass_msun = gw_helpers.redshifted_mass_msun(
            total_mass_msun,
            redshift,
        )
        source_lisa_min_gw_frequency_hz = gw_helpers.source_frame_gw_frequency_hz(
            self.config.lisa_min_gw_frequency_hz,
            redshift,
        )
        source_lisa_max_gw_frequency_hz = gw_helpers.source_frame_gw_frequency_hz(
            self.config.lisa_max_gw_frequency_hz,
            redshift,
        )
        redshift_transport_time_myr = gw_helpers.redshift_transport_time_myr(
            redshift,
            self.config.hubble_constant_per_myr,
        )
        gw_time_budget_source_myr = gw_helpers.redshift_source_time_budget_myr(
            redshift,
            self.config.hubble_constant_per_myr,
            self.config.max_gw_time_myr,
        )
        bh_candidate = bool(
            np.isfinite(damping_result["formation_time_myr"])
            and np.isfinite(damping_result["damping_time_myr"])
            and damping_result["formation_time_myr"]
            < damping_result["damping_time_myr"]
        )
        lisa_entry_radius_cm = gw_helpers.lisa_entry_radius_cm(
            result.m_smbh_msun,
            compact_mass_msun,
            self.config.lisa_min_gw_frequency_hz,
            redshift,
        )
        inner_radius_cm = gw_helpers.imri_lisa_hubble_threshold_radius_cm(
            result.m_smbh_msun,
            compact_mass_msun,
            gw_time_budget_source_myr,
            self.config.lisa_min_gw_frequency_hz,
            self.config.gw_eccentricity,
            redshift,
        )
        gw_time_to_lisa_myr = (
            gw_helpers.peters_time_to_lisa_band_myr(
                result.m_smbh_msun,
                compact_mass_msun,
                inner_radius_cm,
                self.config.lisa_min_gw_frequency_hz,
                self.config.gw_eccentricity,
                redshift,
            )
            if np.isfinite(inner_radius_cm)
            else np.inf
        )
        gw_time_to_lisa_observed_myr = gw_helpers.observed_frame_time_myr(
            gw_time_to_lisa_myr,
            redshift,
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
            and gw_time_to_lisa_myr
            <= gw_time_budget_source_myr * (1.0 + 1.0e-10)
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
        start_gw_frequency_hz = gw_helpers.gw_frequency_hz(
            total_mass_msun,
            start_radius_cm,
        )
        inner_gw_frequency_hz = (
            gw_helpers.gw_frequency_hz(total_mass_msun, inner_radius_cm)
            if np.isfinite(inner_radius_cm)
            else np.nan
        )

        return {
            "m_smbh_msun": result.m_smbh_msun,
            "m_nsc_msun": result.m_nsc_msun,
            "repeat": result.repeat_idx,
            "redshift": redshift,
            "object_index": damping_result["idx"],
            "mass_msun": compact_mass_msun,
            "redshifted_total_mass_msun": redshifted_total_mass_msun,
            "formation_time_myr": damping_result["formation_time_myr"],
            "damping_time_myr": damping_result["damping_time_myr"],
            "bh_embedding_time_myr": damping_result["final_time_myr"],
            "start_radius_cm": start_radius_cm,
            "start_radius_rg": start_radius_cm / rg_cm,
            "observed_lisa_min_gw_frequency_hz": (
                self.config.lisa_min_gw_frequency_hz
            ),
            "source_lisa_min_gw_frequency_hz": source_lisa_min_gw_frequency_hz,
            "observed_lisa_max_gw_frequency_hz": (
                self.config.lisa_max_gw_frequency_hz
            ),
            "source_lisa_max_gw_frequency_hz": source_lisa_max_gw_frequency_hz,
            "lisa_entry_radius_cm": lisa_entry_radius_cm,
            "lisa_entry_radius_rg": lisa_entry_radius_cm / rg_cm,
            "inner_radius_cm": inner_radius_cm,
            "inner_radius_rg": inner_radius_cm / rg_cm,
            "start_gw_frequency_hz": start_gw_frequency_hz,
            "start_observed_gw_frequency_hz": gw_helpers.gw_frequency_hz(
                total_mass_msun,
                start_radius_cm,
                redshift,
            ),
            "inner_gw_frequency_hz": inner_gw_frequency_hz,
            "inner_observed_gw_frequency_hz": (
                gw_helpers.gw_frequency_hz(
                    total_mass_msun,
                    inner_radius_cm,
                    redshift,
                )
                if np.isfinite(inner_gw_frequency_hz)
                else np.nan
            ),
            "redshift_transport_time_myr": redshift_transport_time_myr,
            "gw_time_budget_source_myr": gw_time_budget_source_myr,
            "gw_time_to_lisa_myr": gw_time_to_lisa_myr,
            "gw_time_to_lisa_observed_myr": gw_time_to_lisa_observed_myr,
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
        local_time_myr = disk_migration.t_migration(
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
