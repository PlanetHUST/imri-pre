"""Single-realization workflow for the dynamical IMRI simulation."""

from typing import Sequence

import numpy as np

import physical_process.constants as const
import physical_process.disk_migration as disk_migration
import physical_process.general_class as agn
import physical_process.gw_frequency_radius as gw_helpers
import physical_process.nsc_orbital as nsc_orbital
import physical_process.sampling as stellar_sampling
from disk_quantity.migration_torque import TorqueProfile, torque_si_to_cgs

from .config import IMRIDynConfig, YEARS_PER_MYR
from .detect_rate import classify_detectability
from .engine import IMRIDynEngine
from .event_rate import BH_ORIGIN_MIGRATION_FORMED, classify_event_rate
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
from .redshift_evolve import RedshiftEvolution
from .runtime import vprint
from .workers import get_hpc_cpu_count, run_damping_tasks_parallel

MICROHERTZ_IN_HZ = 1.0e-6
MILLIHERTZ_IN_HZ = 1.0e-3


def _safe_gw_frequency_hz(total_mass_msun: float, radius_cm: float) -> float:
    if not np.isfinite(radius_cm) or radius_cm <= 0.0:
        return np.nan
    return gw_helpers.gw_frequency_hz(total_mass_msun, radius_cm)


def _safe_observed_gw_frequency_hz(
    source_frequency_hz: float,
    redshift: float,
) -> float:
    if not np.isfinite(source_frequency_hz) or source_frequency_hz <= 0.0:
        return np.nan
    return gw_helpers.observed_frame_gw_frequency_hz(source_frequency_hz, redshift)


def _scaled_frequency(frequency_hz: float, scale_hz: float) -> float:
    if not np.isfinite(frequency_hz):
        return np.nan
    return float(frequency_hz / scale_hz)


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
            star_count=len(mass_msun),
            smbh_mass_msun=m_smbh_msun,
            max_radius_cm=nsc_radius_cm,
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
                    nsc_orbital.t_relax(
                        agn_properties.smbh_mass_msun,
                        orbit_radius_cm,
                    )
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
        """Classify embedded BHs and delayed-BH embedded stars as IMRI candidates."""
        rg_cm = disk_data["gravitational_radius_cm"]
        for damping_result in damping_results:

            if damping_result["obj_type"] == 0:
                result.embedded_star_count += 1

            elif damping_result["obj_type"] == 1:
                result.embedded_bh_count += 1

            else:
                result.outside_or_failed_count += 1
                continue

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

            if event["bh_origin"] == BH_ORIGIN_MIGRATION_FORMED:
                result.migration_formed_bh_count += 1

            if event["is_imri_candidate"]:
                result.imri_count += 1
                result.imri_mass_msun += event["mass_msun"]
                if event["is_lisa_detectable"]:
                    result.detectable_imri_count += 1
                    result.detectable_imri_mass_msun += event["mass_msun"]
                else:
                    result.no_detectable_imri_count += 1

            else:
                result.no_imri_count += 1

        result.imri_rate_per_myr = result.imri_count / self.duty_cycle_myr
        result.imri_mass_rate_msun_per_myr = result.imri_mass_msun / self.duty_cycle_myr
        result.detectable_imri_rate_per_myr = (
            result.detectable_imri_count / self.duty_cycle_myr
        )
        result.detectable_imri_mass_rate_msun_per_myr = (
            result.detectable_imri_mass_msun / self.duty_cycle_myr
        )

    def _imri_event_record(
        self,
        result: IMRIDynSingleResult,
        agn_properties: agn.AGN_properties,
        disk_data: DiskProfiles,
        damping_result: DampingResult,
        rg_cm: float,
    ) -> IMRIEventRecord:

        redshift = self.config.redshift

        migrator_mass_msun = damping_result["mass_msun"]
        r_ini_cm = damping_result["final_semi_major_axis_cm"]
        total_mass_msun = result.m_smbh_msun + migrator_mass_msun

        formation_time_myr = damping_result["formation_time_myr"]
        damping_time_myr = damping_result["damping_time_myr"]

        r_entry_cm = gw_helpers.lisa_entry_radius_cm(
            result.m_smbh_msun,
            migrator_mass_msun,
            self.config.lisa_min_gw_frequency_hz,
            redshift,
        )
        print(f"r_ini is {r_ini_cm:.3e} cm, r_entry is {r_entry_cm:.3e} cm")

        def migration_time_myr_call(
            start_radius_cm: float, target_radius_cm: float
        ) -> float:
            return self.torque_migration_time_to_radius_myr(
                agn_properties,
                disk_data,
                damping_result["mass_g"],
                start_radius_cm,
                target_radius_cm,
            )
            # return self.semi_analytic_migration_time_to_radius_myr(
            #     agn_properties,
            #     disk_data,
            #     migrator_mass_msun * const.msun,
            #     start_radius_cm,
            #     target_radius_cm,
            # )

        def gw_time_to_entry_myr_call(radius_cm: float) -> float:
            return gw_helpers.peters_time_to_lisa_band_myr(
                result.m_smbh_msun,
                migrator_mass_msun,
                radius_cm,
                self.config.lisa_min_gw_frequency_hz,
                self.config.gw_eccentricity,
                redshift,
            )

        event_decision = classify_event_rate(
            object_type=damping_result["obj_type"],
            formation_time_myr=formation_time_myr,
            damping_time_myr=damping_time_myr,
            embed_time_myr=damping_result["final_time_myr"],
            r_ini_cm=r_ini_cm,
            r_entry_cm=r_entry_cm,
            migration_time_myr_call=migration_time_myr_call,
            gw_time_to_entry_myr_call=gw_time_to_entry_myr_call,
            duty_cycle_myr=self.config.duty_cycle_myr,
        )

        redshift_at_gas_fin = RedshiftEvolution(
            self.config.hubble_constant_per_myr
        ).redshift_after_time(redshift, event_decision.gas_final_time_myr)

        f_ini_source_hz = _safe_gw_frequency_hz(total_mass_msun, r_ini_cm)
        f_gas_fin_source_hz = _safe_gw_frequency_hz(
            total_mass_msun,
            event_decision.r_gas_fin_cm,
        )
        f_entry_source_hz = _safe_gw_frequency_hz(total_mass_msun, r_entry_cm)

        f_ini_observed_hz = _safe_observed_gw_frequency_hz(
            f_ini_source_hz,
            redshift,
        )
        f_gas_fin_observed_hz = _safe_observed_gw_frequency_hz(
            f_gas_fin_source_hz,
            redshift_at_gas_fin,
        )

        detect_decision = classify_detectability(
            is_imri_candidate=event_decision.is_imri_candidate,
            primary_mass_msun=result.m_smbh_msun,
            secondary_mass_msun=migrator_mass_msun,
            r_gas_fin_cm=event_decision.r_gas_fin_cm,
            gas_final_time_myr=event_decision.gas_final_time_myr,
            initial_redshift=redshift,
            hubble_constant_per_myr=self.config.hubble_constant_per_myr,
            min_observed_frequency_hz=self.config.lisa_min_gw_frequency_hz,
            max_observed_frequency_hz=self.config.lisa_max_gw_frequency_hz,
            eccentricity=self.config.gw_eccentricity,
            cosmic_merge_time_myr=self.config.cosmic_merge_time_myr,
            cosmic_merge_smbh_mass_factor=self.config.cosmic_merge_smbh_mass_factor,
            cosmic_merge_radius_factor=self.config.cosmic_merge_radius_factor,
            snr_threshold=self.config.snr_threshold,
        )

        f_entry_output_source_hz = (
            detect_decision.source_frequency_hz
            if np.isfinite(detect_decision.source_frequency_hz)
            else f_entry_source_hz
        )
        f_entry_output_observed_hz = (
            detect_decision.observed_frequency_hz
            if np.isfinite(detect_decision.observed_frequency_hz)
            else _safe_observed_gw_frequency_hz(f_entry_source_hz, redshift)
        )

        return {
            "m_smbh_msun": result.m_smbh_msun,
            "m_nsc_msun": result.m_nsc_msun,
            "repeat": result.repeat_idx,
            "redshift": redshift,
            "detection_redshift": detect_decision.detection_redshift,
            "object_index": damping_result["idx"],
            "mass_msun": migrator_mass_msun,
            "bh_origin": event_decision.bh_origin,
            "event_rate_class": event_decision.event_rate_class,
            "detect_rate_class": detect_decision.detect_rate_class,
            "formation_time_myr": formation_time_myr,
            "damping_time_myr": damping_time_myr,
            "bh_embedding_time_myr": event_decision.bh_embedding_time_myr,
            "r_ini_cm": r_ini_cm,
            "r_ini_rg": r_ini_cm / rg_cm,
            "r_gas_fin_cm": event_decision.r_gas_fin_cm,
            "r_gas_fin_rg": event_decision.r_gas_fin_cm / rg_cm,
            "r_entry_cm": r_entry_cm,
            "r_entry_rg": r_entry_cm / rg_cm,
            "r_det_entry_cm": detect_decision.final_entry_radius_cm,
            "r_det_entry_rg": detect_decision.final_entry_radius_cm / rg_cm,
            "f_ini_src_microhz": _scaled_frequency(
                f_ini_source_hz,
                MICROHERTZ_IN_HZ,
            ),
            "f_ini_obs_microhz": _scaled_frequency(
                f_ini_observed_hz,
                MICROHERTZ_IN_HZ,
            ),
            "f_gas_fin_src_microhz": _scaled_frequency(
                f_gas_fin_source_hz,
                MICROHERTZ_IN_HZ,
            ),
            "f_gas_fin_obs_microhz": _scaled_frequency(
                f_gas_fin_observed_hz,
                MICROHERTZ_IN_HZ,
            ),
            "f_entry_src_mhz": _scaled_frequency(
                f_entry_output_source_hz,
                MILLIHERTZ_IN_HZ,
            ),
            "f_entry_obs_mhz": _scaled_frequency(
                f_entry_output_observed_hz,
                MILLIHERTZ_IN_HZ,
            ),
            "gas_elapsed_time_myr": event_decision.gas_elapsed_time_myr,
            "gas_final_time_myr": event_decision.gas_final_time_myr,
            "migration_time_to_entry_myr": (event_decision.migration_time_to_entry_myr),
            "gw_time_to_entry_myr": event_decision.gw_time_to_entry_myr,
            "gw_time_after_gas_myr": detect_decision.gw_time_after_gas_myr,
            "detection_time_myr": detect_decision.detection_time_myr,
            "final_smbh_mass_msun": detect_decision.final_smbh_mass_msun,
            "cosmic_merge_applied": detect_decision.cosmic_merge_applied,
            "bh_candidate": event_decision.bh_candidate,
            "is_imri_candidate": event_decision.is_imri_candidate,
            "is_lisa_detectable": detect_decision.is_lisa_detectable,
            "snr_proxy": detect_decision.snr_proxy,
            "event_rate_reason": event_decision.event_rate_reason,
            "detect_rate_reason": detect_decision.detect_rate_reason,
        }

    def torque_migration_time_to_radius_myr(
        self,
        agn_properties: agn.AGN_properties,
        disk_data: DiskProfiles,
        mass_g: float,
        start_radius_cm: float,
        target_radius_cm: float,
    ) -> float:
        """Integrate the cached pAGN torque profile from start to target radius."""
        if not np.isfinite(target_radius_cm) or start_radius_cm <= 0.0:
            return np.inf
        if start_radius_cm <= target_radius_cm:
            return 0.0

        migrator_mass_msun = mass_g / const.msun
        torque_profile = self.engine.get_torque_profile(
            disk_data,
            agn_properties.smbh_mass_msun,
            migrator_mass_msun,
        )

        if torque_profile is None:
            return self.semi_analytic_migration_time_to_radius_myr(
                agn_properties,
                disk_data,
                mass_g,
                start_radius_cm,
                target_radius_cm,
            )

        torque_cgs = self._migration_torque_cgs(
            torque_profile,
            agn_properties,
            disk_data,
            mass_g,
        )

        return disk_migration.migration_time_from_torque(
            np.asarray(torque_profile.radius_cm, dtype=float),
            torque_cgs,
            mass_g,
            start_radius_cm,
            target_radius_cm,
            agn_properties.smbh_mass_msun,
        )

    def _migration_torque_cgs(
        self,
        torque_profile: TorqueProfile,
        agn_properties: agn.AGN_properties,
        disk_data: DiskProfiles,
        mass_g: float,
    ) -> FloatArray:
        """Compose the caller-selected migration torque and convert SI to CGS."""
        torque_si = np.asarray(
            torque_profile.gamma_type_i_si
            + torque_profile.gap_attenuation_factor * torque_profile.gamma_th_si,
            dtype=float,
        )
        if self.config.consider_type_ii_migration:
            torque_radius_cm = np.asarray(torque_profile.radius_cm, dtype=float)
            disk_radius_cm = np.asarray(disk_data["radius_cm"], dtype=float)
            aspect_ratio = np.asarray(disk_data["aspect_ratio"], dtype=float)
            order = np.argsort(disk_radius_cm)
            disk_radius_cm = disk_radius_cm[order]
            aspect_ratio = aspect_ratio[order]
            local_aspect_ratio = np.interp(
                torque_radius_cm,
                disk_radius_cm,
                aspect_ratio,
            )
            smbh_mass_g = agn_properties.smbh_mass_msun * const.msun
            type_ii_gap_parameter_dimensionless = mass_g**2 / (
                disk_migration.DISK_VISCOSITY_ALPHA
                * local_aspect_ratio**2
                * smbh_mass_g**2
            )
            type_ii_mask = np.isfinite(type_ii_gap_parameter_dimensionless) & (
                type_ii_gap_parameter_dimensionless
                > disk_migration.TYPE_II_GAP_PARAMETER_THRESHOLD
            )
            torque_si = torque_si + np.where(
                type_ii_mask,
                np.asarray(torque_profile.gamma_type_ii_si, dtype=float),
                0.0,
            )
        return torque_si_to_cgs(torque_si)

    def semi_analytic_migration_time_to_radius_myr(
        self,
        agn_properties: agn.AGN_properties,
        disk_data: DiskProfiles,
        mass_g: float,
        start_radius_cm: float,
        target_radius_cm: float,
    ) -> float:
        """Approximate migration time [Myr] using the semi-analytic radial integral."""
        if not np.isfinite(target_radius_cm) or start_radius_cm <= 0.0:
            return np.inf
        if start_radius_cm <= target_radius_cm:
            return 0.0
        migration_time_myr = (
            disk_migration.integrated_semi_analytical_migration_timescale(
                agn_properties,
                mass_g,
                start_radius_cm,
                target_radius_cm,
                aspect_ratio=disk_data["aspect_ratio"],
                surface_density_g_cm2=disk_data["surface_density_g_cm2"],
                radius_grid_cm=disk_data["radius_cm"],
                consider_type_ii=self.config.consider_type_ii_migration,
            )
        )
        if not np.isfinite(migration_time_myr) or migration_time_myr <= 0.0:
            return np.inf
        return float(migration_time_myr)
