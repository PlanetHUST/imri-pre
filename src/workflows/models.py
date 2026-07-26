"""Typed workflow payloads and result containers."""

from typing import Any, Sequence, TypedDict

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class DampingTask(TypedDict):
    task_kind: str
    idx: int
    origin_primary_idx: int
    copy_number: int
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
    origin_primary_idx: int
    copy_number: int
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


class TimescaleRecord(TypedDict):
    """Per-object timescale diagnostics written to ``timescales.csv``."""

    m_smbh_msun: float
    m_nsc_msun: float
    repeat: int
    task_kind: str
    object_index: int
    origin_primary_index: int
    copy_number: int
    object_type_flag: int
    mass_msun: float
    initial_semi_major_axis_cm: float
    final_semi_major_axis_cm: float
    eccentricity: float
    inclination_rad: float
    state_time_myr: float
    formation_time_myr: float
    damping_time_myr: float
    migration_time_myr: float
    relaxation_time_myr: float
    delivery_time_myr: float
    outcome: str
    counted_as_tde: bool
    tde_mode: str


TIMESCALE_COLUMNS: tuple[str, ...] = tuple(
    TimescaleRecord.__annotations__
)


class TDEOrbitalParams(TypedDict):
    """Initial properties of one star that is ultimately disrupted."""

    task_kind: str
    semi_major_axis_cm: float
    eccentricity: float
    inclination_rad: float
    mass_msun: float


class IntermediatePopulation(TypedDict):
    """Initial population properties used for intermediate diagnostics."""

    mass_msun: FloatArray
    formation_time_myr: FloatArray
    semi_major_axis_cm: FloatArray
    eccentricity: FloatArray
    inclination_rad: FloatArray


class TDEDynSingleResult:
    """Results for one repeat at one grid point."""

    def __init__(
        self, m_smbh_msun: float, m_nsc_msun: float, repeat_idx: int
    ) -> None:
        self.m_smbh_msun = m_smbh_msun
        self.m_nsc_msun = m_nsc_msun
        self.repeat_idx = repeat_idx
        self.total_stars = 0
        self.geo_selected_stars = 0
        self.f_geo = 0.0
        self.embedded_stars_primary = 0
        self.embedded_bh_primary = 0
        self.outside_primary = 0
        self.num_copies_generated = 0
        self.num_copy_tdes = 0
        self.copy_tde_mass_msun = 0.0
        self.tde_star_count_primary = 0
        self.tde_mass_primary_msun = 0.0
        self.tde_star_count_total = 0
        self.tde_mass_total_msun = 0.0
        self.tde_mass_rate_total_msun_per_myr = 0.0
        self.tde_masses_primary_msun: list[float] = []
        self.tde_masses_copies_msun: list[float] = []
        self.tde_times_primary_myr: list[float] = []
        self.tde_times_copies_myr: list[float] = []
        self.tde_orbital_params: list[TDEOrbitalParams] = []
        self.intermediate_initial_population: IntermediatePopulation | None = None
        self.intermediate_disk_crossing_population: IntermediatePopulation | None = None
        self.disk_inner_radius_cm: float | None = None
        self.disk_outer_radius_cm: float | None = None
        self.nsc_radius_cm: float | None = None
        self.mean_relaxation_time_myr_geo = np.nan
        self.direct_tde_primary_count = 0
        self.migration_tde_primary_count = 0
        self.direct_tde_copy_count = 0
        self.migration_tde_copy_count = 0
        self.timescale_records: list[TimescaleRecord] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "m_smbh_msun": self.m_smbh_msun,
            "m_nsc_msun": self.m_nsc_msun,
            "repeat": self.repeat_idx,
            "total_stars": self.total_stars,
            "geo_selected_stars": self.geo_selected_stars,
            "f_geo": self.f_geo,
            "embedded_stars_primary": self.embedded_stars_primary,
            "embedded_bh_primary": self.embedded_bh_primary,
            "outside_primary": self.outside_primary,
            "num_copies_generated": self.num_copies_generated,
            "num_copy_tdes": self.num_copy_tdes,
            "tde_star_count_primary": self.tde_star_count_primary,
            "tde_mass_primary_msun": self.tde_mass_primary_msun,
            "copy_tde_mass_msun": self.copy_tde_mass_msun,
            "tde_star_count_total": self.tde_star_count_total,
            "tde_mass_total_msun": self.tde_mass_total_msun,
            "tde_mass_rate_total_msun_per_myr": (
                self.tde_mass_rate_total_msun_per_myr
            ),
            "disk_inner_radius_cm": self.disk_inner_radius_cm,
            "disk_outer_radius_cm": self.disk_outer_radius_cm,
            "nsc_radius_cm": self.nsc_radius_cm,
            "mean_relaxation_time_myr_geo": self.mean_relaxation_time_myr_geo,
            "tde_masses_primary_msun": self.tde_masses_primary_msun,
            "tde_masses_copies_msun": self.tde_masses_copies_msun,
            "tde_times_primary_myr": self.tde_times_primary_myr,
            "tde_times_copies_myr": self.tde_times_copies_myr,
        }


class TDEDynAveragedResult:
    """Average across repeats at one grid point."""

    def __init__(
        self,
        m_smbh_msun: float,
        m_nsc_msun: float,
        results: Sequence[TDEDynSingleResult],
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
            "f_geo_mean": np.mean(values("f_geo")),
            "f_geo_min": np.min(values("f_geo")),
            "f_geo_max": np.max(values("f_geo")),
            "geo_selected_stars_mean": np.mean(
                values("geo_selected_stars")
            ),
            "tde_mass_primary_mean_msun": np.mean(
                values("tde_mass_primary_msun")
            ),
            "tde_mass_total_mean_msun": np.mean(
                values("tde_mass_total_msun")
            ),
            "tde_mass_rate_total_mean_msun_per_myr": np.mean(
                values("tde_mass_rate_total_msun_per_myr")
            ),
            "tde_mass_rate_total_min_msun_per_myr": np.min(
                values("tde_mass_rate_total_msun_per_myr")
            ),
            "tde_mass_rate_total_max_msun_per_myr": np.max(
                values("tde_mass_rate_total_msun_per_myr")
            ),
            "tde_star_count_total_mean": np.mean(
                values("tde_star_count_total")
            ),
            "num_copies_generated_mean": np.mean(
                values("num_copies_generated")
            ),
            "mean_relaxation_time_geo_mean_myr": np.mean(
                values("mean_relaxation_time_myr_geo")
            ),
        }
