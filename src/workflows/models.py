"""Typed payloads and result containers for the IMRI workflow."""

from typing import Any, NotRequired, Sequence, TypedDict

import numpy as np
import numpy.typing as npt

from physical_process.damp import DampingStatus

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class DampingTask(TypedDict):
    task_kind: str
    idx: int
    mass_g: float
    semi_major_axis_cm: float
    eccentricity: float
    inclination_rad: float
    cos_omega: float
    formation_time_myr: float
    state_time_s: float


class DampingResult(TypedDict):
    task_kind: str
    idx: int
    final_time_s: float
    final_time_myr: float
    final_semi_major_axis_cm: float
    final_eccentricity: float
    final_inclination_rad: float
    final_cos_omega: float
    status: DampingStatus
    formation_time_myr: float
    state_time_myr: float
    damping_time_myr: float
    initial_semi_major_axis_cm: float
    initial_eccentricity: float
    initial_inclination_rad: float
    initial_cos_omega: float
    mass_g: float
    mass_msun: float
    error: NotRequired[str]


class DiskProfiles(TypedDict):
    inner_radius_cm: float
    outer_radius_cm: float
    radius_cm: FloatArray
    surface_density_g_cm2: FloatArray
    aspect_ratio: FloatArray
    gravitational_radius_cm: float
    torque_disk: NotRequired[object]
    torque_alpha: NotRequired[float]
    torque_profile_error: NotRequired[str]


class StellarPopulation(TypedDict):
    mass_msun: FloatArray
    formation_time_myr: FloatArray
    semi_major_axis_cm: FloatArray
    eccentricity: FloatArray
    inclination_rad: FloatArray
    cos_omega: FloatArray
    nsc_radius_cm: float


class GeometricPopulation(TypedDict):
    mass_msun: FloatArray
    formation_time_myr: FloatArray
    semi_major_axis_cm: FloatArray
    eccentricity: FloatArray
    inclination_rad: FloatArray
    cos_omega: FloatArray


class IMRIEventRecord(TypedDict):
    m_smbh_msun: float
    m_nsc_msun: float
    repeat: int
    disk_model: str
    redshift: float
    detection_redshift: float
    object_index: int
    mass_msun: float
    bh_origin: str
    band_entry_status: str
    event_rate_class: str
    detect_rate_class: str
    formation_time_myr: float
    damping_time_myr: float
    bh_embedding_time_myr: float
    r_ini_cm: float
    r_ini_rg: float
    r_gas_fin_cm: float
    r_gas_fin_rg: float
    r_entry_cm: float
    r_entry_rg: float
    r_entry_formal_cm: float
    r_entry_formal_rg: float
    r_isco_cm: float
    r_isco_rg: float
    r_disk_inner_cm: float
    r_disk_inner_rg: float
    r_migration_min_cm: float
    r_migration_min_rg: float
    r_handoff_cm: float
    r_handoff_rg: float
    r_det_entry_cm: float
    r_det_entry_rg: float
    source_entry_frequency_hz: float
    f_ini_src_hz: float
    f_ini_obs_hz: float
    f_gas_fin_src_hz: float
    f_gas_fin_obs_hz: float
    f_entry_src_hz: float
    f_entry_obs_hz: float
    gas_elapsed_time_myr: float
    gas_final_time_myr: float
    migration_time_to_entry_myr: float
    migration_time_to_handoff_myr: float
    gw_time_handoff_to_entry_myr: float
    gas_stop_reason: str
    migration_evaluated: bool
    migration_status: str
    blocking_radius_rg: float | None
    gw_time_to_entry_myr: float
    gw_time_after_gas_myr: float
    detection_time_myr: float
    final_smbh_mass_msun: float
    cosmic_merge_applied: bool
    bh_candidate: bool
    is_imri_candidate: bool | None
    is_lisa_detectable: bool | None
    snr_proxy: float
    event_rate_reason: str
    detect_rate_reason: str


class MigrationErrorRecord(TypedDict):
    object_index: int
    mass_msun: float
    r_ini_rg: float
    r_entry_rg: float
    r_isco_rg: float
    band_entry_status: str
    error: str


TorqueErrorRecord = MigrationErrorRecord


class IMRIDynSingleResult:
    """Results for one repeat at one grid point."""

    def __init__(
        self,
        m_smbh_msun: float,
        m_nsc_msun: float,
        repeat_idx: int,
        disk_model: str,
        task_seed: int | None = None,
    ) -> None:
        self.m_smbh_msun = m_smbh_msun
        self.m_nsc_msun = m_nsc_msun
        self.repeat_idx = repeat_idx
        self.disk_model = disk_model
        self.task_seed = task_seed
        self.total_sampled_objects = 0
        self.disk_crossing_objects = 0
        self.f_geo = 0.0
        self.embedded_star_count = 0
        self.embedded_bh_count = 0
        self.migration_formed_bh_count = 0
        self.outside_or_failed_count = 0
        self.bh_candidate_count = 0
        self.no_imri_count = 0
        self.imri_count = 0
        self.imri_mass_msun = 0.0
        self.imri_rate_per_myr = 0.0
        self.imri_mass_rate_msun_per_myr = 0.0
        self.detectable_imri_count = 0
        self.no_detectable_imri_count = 0
        self.detectable_imri_mass_msun = 0.0
        self.detectable_imri_rate_per_myr = 0.0
        self.detectable_imri_mass_rate_msun_per_myr = 0.0
        self.band_entry_evaluated_count = 0
        self.entry_in_migration_domain_count = 0
        self.entry_handoff_count = 0
        self.band_entry_inside_isco_count = 0
        self.profile_error_count = 0
        self.migration_evaluated_count = 0
        self.finite_migration_count = 0
        self.trap_bound_migration_count = 0
        self.migration_error_count = 0
        self.trap_bound_mass_msun = 0.0
        self.disk_inner_radius_cm: float | None = None
        self.disk_outer_radius_cm: float | None = None
        self.gravitational_radius_cm: float | None = None
        self.nsc_radius_cm: float | None = None
        self.mean_relaxation_time_myr_geo = np.nan
        self.imri_events: list[IMRIEventRecord] = []
        self.migration_errors: list[MigrationErrorRecord] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "m_smbh_msun": self.m_smbh_msun,
            "m_nsc_msun": self.m_nsc_msun,
            "repeat": self.repeat_idx,
            "disk_model": self.disk_model,
            "task_seed": self.task_seed,
            "total_sampled_objects": self.total_sampled_objects,
            "disk_crossing_objects": self.disk_crossing_objects,
            "f_geo": self.f_geo,
            "embedded_star_count": self.embedded_star_count,
            "embedded_bh_count": self.embedded_bh_count,
            "migration_formed_bh_count": self.migration_formed_bh_count,
            "outside_or_failed_count": self.outside_or_failed_count,
            "bh_candidate_count": self.bh_candidate_count,
            "no_imri_count": self.no_imri_count,
            "imri_count": self.imri_count,
            "imri_mass_msun": self.imri_mass_msun,
            "imri_rate_per_myr": self.imri_rate_per_myr,
            "imri_mass_rate_msun_per_myr": self.imri_mass_rate_msun_per_myr,
            "detectable_imri_count": self.detectable_imri_count,
            "no_detectable_imri_count": self.no_detectable_imri_count,
            "detectable_imri_mass_msun": self.detectable_imri_mass_msun,
            "detectable_imri_rate_per_myr": self.detectable_imri_rate_per_myr,
            "detectable_imri_mass_rate_msun_per_myr": (
                self.detectable_imri_mass_rate_msun_per_myr
            ),
            "band_entry_evaluated_count": self.band_entry_evaluated_count,
            "entry_in_migration_domain_count": (self.entry_in_migration_domain_count),
            "entry_handoff_count": self.entry_handoff_count,
            "band_entry_inside_isco_count": self.band_entry_inside_isco_count,
            "profile_error_count": self.profile_error_count,
            "migration_evaluated_count": self.migration_evaluated_count,
            "finite_migration_count": self.finite_migration_count,
            "trap_bound_migration_count": self.trap_bound_migration_count,
            "migration_error_count": self.migration_error_count,
            "trap_bound_mass_msun": self.trap_bound_mass_msun,
            "disk_inner_radius_cm": self.disk_inner_radius_cm,
            "disk_outer_radius_cm": self.disk_outer_radius_cm,
            "gravitational_radius_cm": self.gravitational_radius_cm,
            "nsc_radius_cm": self.nsc_radius_cm,
            "mean_relaxation_time_myr_geo": self.mean_relaxation_time_myr_geo,
        }


class IMRIDynAveragedResult:
    """Average IMRI results across repeats at one grid point."""

    def __init__(
        self,
        m_smbh_msun: float,
        m_nsc_msun: float,
        disk_model: str,
        results: Sequence[IMRIDynSingleResult],
    ) -> None:
        self.m_smbh_msun = m_smbh_msun
        self.m_nsc_msun = m_nsc_msun
        self.disk_model = disk_model
        self.results = results

    def to_dict(self) -> dict[str, float | int | str]:
        def values(name: str) -> FloatArray:
            return np.asarray(
                [getattr(result, name) for result in self.results],
                dtype=float,
            )

        inside_isco_count = float(np.sum(values("band_entry_inside_isco_count")))
        profile_error_count = float(np.sum(values("profile_error_count")))
        rate_evaluation_complete = (
            inside_isco_count == 0.0 and profile_error_count == 0.0
        )
        if inside_isco_count > 0.0 and profile_error_count > 0.0:
            rate_evaluation_status = "not_evaluated_band_entry_and_profile_error"
        elif inside_isco_count > 0.0:
            rate_evaluation_status = "not_evaluated_band_entry_inside_isco"
        elif profile_error_count > 0.0:
            rate_evaluation_status = "incomplete_profile_error"
        else:
            rate_evaluation_status = "complete"

        def classified_stat(name: str, reducer: str = "mean") -> float:
            if not rate_evaluation_complete:
                return np.nan
            data = values(name)
            if reducer == "min":
                return float(np.min(data))
            if reducer == "max":
                return float(np.max(data))
            return float(np.mean(data))

        return {
            "m_smbh_msun": self.m_smbh_msun,
            "m_nsc_msun": self.m_nsc_msun,
            "disk_model": self.disk_model,
            "num_repeats": len(self.results),
            "rate_evaluation_status": rate_evaluation_status,
            "f_geo_mean": float(np.mean(values("f_geo"))),
            "disk_crossing_objects_mean": float(
                np.mean(values("disk_crossing_objects"))
            ),
            "embedded_star_count_mean": float(np.mean(values("embedded_star_count"))),
            "embedded_bh_count_mean": float(np.mean(values("embedded_bh_count"))),
            "migration_formed_bh_count_mean": float(
                np.mean(values("migration_formed_bh_count"))
            ),
            "bh_candidate_count_mean": float(np.mean(values("bh_candidate_count"))),
            "no_imri_count_mean": float(np.mean(values("no_imri_count"))),
            "imri_candidate_count_mean": classified_stat("imri_count"),
            "imri_candidate_count_min": classified_stat("imri_count", "min"),
            "imri_candidate_count_max": classified_stat("imri_count", "max"),
            "imri_candidate_mass_mean_msun": classified_stat("imri_mass_msun"),
            "imri_candidate_rate_mean_per_myr": classified_stat("imri_rate_per_myr"),
            "imri_candidate_mass_rate_mean_msun_per_myr": classified_stat(
                "imri_mass_rate_msun_per_myr"
            ),
            "detectable_imri_count_mean": classified_stat("detectable_imri_count"),
            "detectable_imri_count_min": classified_stat(
                "detectable_imri_count", "min"
            ),
            "detectable_imri_count_max": classified_stat(
                "detectable_imri_count", "max"
            ),
            "detectable_imri_mass_mean_msun": classified_stat(
                "detectable_imri_mass_msun"
            ),
            "detectable_imri_rate_mean_per_myr": classified_stat(
                "detectable_imri_rate_per_myr"
            ),
            "detectable_imri_mass_rate_mean_msun_per_myr": classified_stat(
                "detectable_imri_mass_rate_msun_per_myr"
            ),
            "no_detectable_imri_count_mean": classified_stat(
                "no_detectable_imri_count"
            ),
            "band_entry_evaluated_count_mean": float(
                np.mean(values("band_entry_evaluated_count"))
            ),
            "entry_in_migration_domain_count_mean": float(
                np.mean(values("entry_in_migration_domain_count"))
            ),
            "entry_handoff_count_mean": float(np.mean(values("entry_handoff_count"))),
            "band_entry_inside_isco_count_mean": float(
                np.mean(values("band_entry_inside_isco_count"))
            ),
            "profile_error_count_mean": float(np.mean(values("profile_error_count"))),
            "migration_evaluated_count_mean": float(
                np.mean(values("migration_evaluated_count"))
            ),
            "finite_migration_count_mean": float(
                np.mean(values("finite_migration_count"))
            ),
            "trap_bound_migration_count_mean": float(
                np.mean(values("trap_bound_migration_count"))
            ),
            "migration_error_count_mean": float(
                np.mean(values("migration_error_count"))
            ),
            "trap_bound_mass_mean_msun": float(np.mean(values("trap_bound_mass_msun"))),
            "mean_relaxation_time_geo_mean_myr": float(
                np.mean(values("mean_relaxation_time_myr_geo"))
            ),
        }
