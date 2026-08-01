"""Typed payloads and result containers for the IMRI workflow."""

from typing import Any, NotRequired, Sequence, TypedDict

import numpy as np
import numpy.typing as npt

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
    type_flag: int


class DampingResult(TypedDict):
    task_kind: str
    idx: int
    final_time_s: float
    final_time_myr: float
    final_semi_major_axis_cm: float
    obj_type: int
    formation_time_myr: float
    state_time_myr: float
    damping_time_myr: float
    initial_semi_major_axis_cm: float
    eccentricity: float
    inclination_rad: float
    cos_omega: float
    mass_g: float
    mass_msun: float


class DiskProfiles(TypedDict):
    inner_radius_cm: float
    outer_radius_cm: float
    radius_cm: FloatArray
    surface_density_g_cm2: FloatArray
    aspect_ratio: FloatArray
    gravitational_radius_cm: float
    torque_disk: NotRequired[object]
    torque_alpha: NotRequired[float]


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


class IntermediatePopulation(TypedDict):
    mass_msun: FloatArray
    formation_time_myr: FloatArray
    semi_major_axis_cm: FloatArray
    eccentricity: FloatArray
    inclination_rad: FloatArray


class IMRIEventRecord(TypedDict):
    m_smbh_msun: float
    m_nsc_msun: float
    repeat: int
    redshift: float
    object_index: int
    mass_msun: float
    bh_origin: str
    event_rate_class: str
    detect_rate_class: str
    formation_time_myr: float
    damping_time_myr: float
    bh_embedding_time_myr: float
    start_radius_cm: float
    start_radius_rg: float
    r_imri_cm: float
    r_imri_rg: float
    r_entry_cm: float
    r_entry_rg: float
    f_start_src_microhz: float
    f_start_obs_microhz: float
    f_imri_src_microhz: float
    f_imri_obs_microhz: float
    f_entry_src_mhz: float
    f_entry_obs_mhz: float
    redshift_transport_time_myr: float
    gw_time_budget_source_myr: float
    gw_time_from_r_imri_to_entry_myr: float
    gw_time_from_r_imri_to_entry_observed_myr: float
    migration_time_myr: float
    arrival_time_myr: float
    bh_candidate: bool
    is_imri_candidate: bool
    is_lisa_detectable: bool
    snr_proxy: float
    event_rate_reason: str
    detect_rate_reason: str


class IMRIDynSingleResult:
    """Results for one repeat at one grid point."""

    def __init__(
        self, m_smbh_msun: float, m_nsc_msun: float, repeat_idx: int
    ) -> None:
        self.m_smbh_msun = m_smbh_msun
        self.m_nsc_msun = m_nsc_msun
        self.repeat_idx = repeat_idx
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
        self.disk_inner_radius_cm: float | None = None
        self.disk_outer_radius_cm: float | None = None
        self.nsc_radius_cm: float | None = None
        self.mean_relaxation_time_myr_geo = np.nan
        self.imri_events: list[IMRIEventRecord] = []
        self.intermediate_initial_population: IntermediatePopulation | None = None
        self.intermediate_disk_crossing_population: IntermediatePopulation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "m_smbh_msun": self.m_smbh_msun,
            "m_nsc_msun": self.m_nsc_msun,
            "repeat": self.repeat_idx,
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
            "disk_inner_radius_cm": self.disk_inner_radius_cm,
            "disk_outer_radius_cm": self.disk_outer_radius_cm,
            "nsc_radius_cm": self.nsc_radius_cm,
            "mean_relaxation_time_myr_geo": self.mean_relaxation_time_myr_geo,
        }


class IMRIDynAveragedResult:
    """Average IMRI results across repeats at one grid point."""

    def __init__(
        self,
        m_smbh_msun: float,
        m_nsc_msun: float,
        results: Sequence[IMRIDynSingleResult],
    ) -> None:
        self.m_smbh_msun = m_smbh_msun
        self.m_nsc_msun = m_nsc_msun
        self.results = results

    def to_dict(self) -> dict[str, float | int]:
        def values(name: str) -> FloatArray:
            return np.asarray(
                [getattr(result, name) for result in self.results],
                dtype=float,
            )

        return {
            "m_smbh_msun": self.m_smbh_msun,
            "m_nsc_msun": self.m_nsc_msun,
            "num_repeats": len(self.results),
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
            "imri_candidate_count_mean": float(np.mean(values("imri_count"))),
            "imri_candidate_count_min": float(np.min(values("imri_count"))),
            "imri_candidate_count_max": float(np.max(values("imri_count"))),
            "imri_candidate_mass_mean_msun": float(
                np.mean(values("imri_mass_msun"))
            ),
            "imri_candidate_rate_mean_per_myr": float(
                np.mean(values("imri_rate_per_myr"))
            ),
            "imri_candidate_mass_rate_mean_msun_per_myr": float(
                np.mean(values("imri_mass_rate_msun_per_myr"))
            ),
            "detectable_imri_count_mean": float(
                np.mean(values("detectable_imri_count"))
            ),
            "detectable_imri_count_min": float(
                np.min(values("detectable_imri_count"))
            ),
            "detectable_imri_count_max": float(
                np.max(values("detectable_imri_count"))
            ),
            "detectable_imri_mass_mean_msun": float(
                np.mean(values("detectable_imri_mass_msun"))
            ),
            "detectable_imri_rate_mean_per_myr": float(
                np.mean(values("detectable_imri_rate_per_myr"))
            ),
            "detectable_imri_mass_rate_mean_msun_per_myr": float(
                np.mean(values("detectable_imri_mass_rate_msun_per_myr"))
            ),
            "no_detectable_imri_count_mean": float(
                np.mean(values("no_detectable_imri_count"))
            ),
            "mean_relaxation_time_geo_mean_myr": float(
                np.mean(values("mean_relaxation_time_myr_geo"))
            ),
        }
