"""Map one merger-tree host onto the existing single-halo simulation inputs.

This layer answers exactly one question::

    HaloHost + base IMRIDynConfig -> ResolvedHostParameters

It contains no CSV parsing and no IMRI physics.  It only decides *which value*
each host-dependent local input takes, records *where that value came from*,
and refuses hosts that fall outside the configured local-model domain.

Host-dependent local inputs of the existing single-halo model
-------------------------------------------------------------
The existing kernel (``workflows.workflow.run_simulation`` ->
``IMRISingleWorkflow.run_single_simulation``) consumes exactly four quantities
that can legitimately vary from host to host:

``redshift``
    ``IMRIDynConfig.redshift``; drives the source/observer frequency
    conversion, the LISA time budget and the redshift evolution model.
``smbh_mass_msun``
    One entry of ``IMRIDynConfig.smbh_masses_msun``; sets the AGN properties,
    the pAGN disk profile and the gravitational radius.
``nsc_mass_msun``
    One entry of ``IMRIDynConfig.nsc_masses_msun``; sets the sampled NSC
    population mass and the NSC radius.
``metallicity``
    ``IMRIDynConfig.metallicity``, expressed in solar units (``sampling.py``
    multiplies it by ``SOLAR_METALLICITY = 0.0145`` before use); selects the
    Pop II versus Pop III IMF branch.

Everything else in ``IMRIDynConfig`` is a global model parameter and is
inherited unchanged from the base configuration.

Provenance vocabulary
---------------------
Every resolved value is tagged so that a *direct tree value* is never confused
with a *derived local quantity*:

``tree:<column>``
    Read straight out of the merger-tree CSV.
``tree:<column>/Zsun``
    Read from the tree and converted with a verified unit convention.
``derived:<rule>``
    Computed from a tree field through an explicitly configured rule.
``config``
    Not host dependent; taken from the base configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from physical_process.sampling import SOLAR_METALLICITY

from .config import IMRIDynConfig
from .merger_tree import (
    HaloHost,
    TREE_COLUMN_METALLICITY,
    TREE_COLUMN_SMBH_MASS,
    TREE_COLUMN_REDSHIFT,
    TREE_COLUMN_STELLAR_MASS,
)

SMBH_MASS_SOURCE_TREE = "tree"
SMBH_MASS_SOURCE_CONFIG = "config"
SMBH_MASS_SOURCE_STELLAR_SCALING = "stellar-scaling"
SMBH_MASS_SOURCES = (
    SMBH_MASS_SOURCE_TREE,
    SMBH_MASS_SOURCE_CONFIG,
    SMBH_MASS_SOURCE_STELLAR_SCALING,
)

NSC_MASS_SOURCE_CONFIG = "config"
NSC_MASS_SOURCE_STELLAR_SCALING = "stellar-scaling"
NSC_MASS_SOURCES = (NSC_MASS_SOURCE_CONFIG, NSC_MASS_SOURCE_STELLAR_SCALING)

METALLICITY_SOURCE_CONFIG = "config"
METALLICITY_SOURCE_TREE = "tree"
METALLICITY_SOURCES = (METALLICITY_SOURCE_CONFIG, METALLICITY_SOURCE_TREE)

# Status vocabulary shared with the batch layer (see workflows.single_halo).
STATUS_MISSING_REQUIRED_INPUT = "missing_required_input"
STATUS_INVALID_HOST_INPUT = "invalid_host_input"
STATUS_OUTSIDE_LOCAL_MODEL_DOMAIN = "outside_local_model_domain"
STATUS_MAPPING_UNAVAILABLE = "mapping_unavailable"

#: Default local-model domain.  The SMBH bounds reproduce the mass range the
#: repository itself explores by default (``cli.DEFAULT_SMBH_MASSES_MSUN`` spans
#: 1e3 to 1e6 Msun); they are a configurable policy, not a hard solver limit.
DEFAULT_DOMAIN_SMBH_MASS_MIN_MSUN = 1.0e3
DEFAULT_DOMAIN_SMBH_MASS_MAX_MSUN = 1.0e6
DEFAULT_DOMAIN_NSC_MASS_MIN_MSUN = 1.0e2
DEFAULT_DOMAIN_NSC_MASS_MAX_MSUN = 1.0e8
DEFAULT_DOMAIN_REDSHIFT_MAX = 30.0

#: Placeholder linear scalings used only when the corresponding
#: ``stellar-scaling`` source is explicitly selected.  They are user-controlled
#: modelling choices: the supplied merger tree carries no NSC mass, and its
#: ``MBH`` column is empty, so neither quantity can be read from the file.
DEFAULT_SMBH_STELLAR_MASS_FRACTION = 1.0e-4
DEFAULT_NSC_STELLAR_MASS_FRACTION = 1.0e-4


class HostMappingError(ValueError):
    """Raised when a host cannot be turned into a valid local simulation input.

    ``status`` carries the batch-layer status category so the executor can keep
    "could not be evaluated" separate from "evaluated to a physical zero".
    """

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class HostMappingPolicy:
    """Explicit, serialisable rules for turning a host into local inputs."""

    smbh_mass_source: str = SMBH_MASS_SOURCE_TREE
    nsc_mass_source: str = NSC_MASS_SOURCE_CONFIG
    metallicity_source: str = METALLICITY_SOURCE_CONFIG
    smbh_stellar_mass_fraction: float = DEFAULT_SMBH_STELLAR_MASS_FRACTION
    nsc_stellar_mass_fraction: float = DEFAULT_NSC_STELLAR_MASS_FRACTION
    domain_smbh_mass_min_msun: float = DEFAULT_DOMAIN_SMBH_MASS_MIN_MSUN
    domain_smbh_mass_max_msun: float = DEFAULT_DOMAIN_SMBH_MASS_MAX_MSUN
    domain_nsc_mass_min_msun: float = DEFAULT_DOMAIN_NSC_MASS_MIN_MSUN
    domain_nsc_mass_max_msun: float = DEFAULT_DOMAIN_NSC_MASS_MAX_MSUN
    domain_redshift_max: float = DEFAULT_DOMAIN_REDSHIFT_MAX

    def __post_init__(self) -> None:
        if self.smbh_mass_source not in SMBH_MASS_SOURCES:
            raise ValueError(f"SMBH mass source must be one of {SMBH_MASS_SOURCES}")
        if self.nsc_mass_source not in NSC_MASS_SOURCES:
            raise ValueError(f"NSC mass source must be one of {NSC_MASS_SOURCES}")
        if self.metallicity_source not in METALLICITY_SOURCES:
            raise ValueError(f"metallicity source must be one of {METALLICITY_SOURCES}")
        for name in ("smbh_stellar_mass_fraction", "nsc_stellar_mass_fraction"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.domain_smbh_mass_min_msun > self.domain_smbh_mass_max_msun:
            raise ValueError("SMBH domain bounds must satisfy min <= max")
        if self.domain_nsc_mass_min_msun > self.domain_nsc_mass_max_msun:
            raise ValueError("NSC domain bounds must satisfy min <= max")
        if self.domain_redshift_max <= 0.0:
            raise ValueError("domain redshift maximum must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedHostParameters:
    """The four host-dependent inputs of one single-halo run, plus provenance."""

    host_id: str
    redshift: float
    smbh_mass_msun: float
    nsc_mass_msun: float
    metallicity_zsun: float
    sources: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        sources = record.pop("sources")
        for name, origin in sources.items():
            record[f"{name}_source"] = origin
        return record


def _require_positive(
    value: float | None,
    label: str,
    origin: str,
) -> float:
    if value is None or not np.isfinite(value):
        raise HostMappingError(
            STATUS_MISSING_REQUIRED_INPUT,
            f"{label} is unavailable from {origin}",
        )
    if value <= 0.0:
        raise HostMappingError(
            STATUS_INVALID_HOST_INPUT,
            f"{label} from {origin} must be positive (got {value:.6g})",
        )
    return float(value)


def _resolve_smbh_mass(
    host: HaloHost,
    base_config: IMRIDynConfig,
    policy: HostMappingPolicy,
) -> tuple[float, str]:
    if policy.smbh_mass_source == SMBH_MASS_SOURCE_TREE:
        mass = _require_positive(
            host.smbh_mass_msun,
            "central black-hole mass",
            f"tree column '{TREE_COLUMN_SMBH_MASS}'",
        )
        return mass, f"tree:{TREE_COLUMN_SMBH_MASS}"

    if policy.smbh_mass_source == SMBH_MASS_SOURCE_STELLAR_SCALING:
        stellar_mass = _require_positive(
            host.stellar_mass_msun,
            "host stellar mass",
            f"tree column '{TREE_COLUMN_STELLAR_MASS}'",
        )
        mass = policy.smbh_stellar_mass_fraction * stellar_mass
        return (
            mass,
            f"derived:{policy.smbh_stellar_mass_fraction:.6g}"
            f"*tree:{TREE_COLUMN_STELLAR_MASS}",
        )

    if base_config.smbh_masses_msun.size != 1:
        raise HostMappingError(
            STATUS_MAPPING_UNAVAILABLE,
            "config SMBH mass source requires exactly one --smbh-masses value, "
            f"got {base_config.smbh_masses_msun.size}",
        )
    return float(base_config.smbh_masses_msun[0]), "config"


def _resolve_nsc_mass(
    host: HaloHost,
    base_config: IMRIDynConfig,
    policy: HostMappingPolicy,
) -> tuple[float, str]:
    if policy.nsc_mass_source == NSC_MASS_SOURCE_STELLAR_SCALING:
        stellar_mass = _require_positive(
            host.stellar_mass_msun,
            "host stellar mass",
            f"tree column '{TREE_COLUMN_STELLAR_MASS}'",
        )
        mass = policy.nsc_stellar_mass_fraction * stellar_mass
        return (
            mass,
            f"derived:{policy.nsc_stellar_mass_fraction:.6g}"
            f"*tree:{TREE_COLUMN_STELLAR_MASS}",
        )

    if base_config.nsc_masses_msun.size != 1:
        raise HostMappingError(
            STATUS_MAPPING_UNAVAILABLE,
            "config NSC mass source requires exactly one --nsc-masses value, "
            f"got {base_config.nsc_masses_msun.size}",
        )
    return float(base_config.nsc_masses_msun[0]), "config"


def _resolve_metallicity(
    host: HaloHost,
    base_config: IMRIDynConfig,
    policy: HostMappingPolicy,
) -> tuple[float, str]:
    if policy.metallicity_source == METALLICITY_SOURCE_CONFIG:
        return float(base_config.metallicity), "config"

    mass_fraction = host.metallicity_mass_fraction
    if mass_fraction is None or not np.isfinite(mass_fraction):
        raise HostMappingError(
            STATUS_MISSING_REQUIRED_INPUT,
            f"metallicity is unavailable from tree column '{TREE_COLUMN_METALLICITY}'",
        )
    if mass_fraction < 0.0:
        raise HostMappingError(
            STATUS_INVALID_HOST_INPUT,
            f"tree metallicity must be nonnegative (got {mass_fraction:.6g})",
        )
    # The tree carries an absolute metal mass fraction; IMRIDynConfig.metallicity
    # is in solar units because sampling.sample_efficient multiplies it by
    # SOLAR_METALLICITY before applying the Pop III threshold.
    return (
        float(mass_fraction) / SOLAR_METALLICITY,
        f"tree:{TREE_COLUMN_METALLICITY}/Zsun",
    )


def _check_domain(
    host_id: str,
    redshift: float,
    smbh_mass_msun: float,
    nsc_mass_msun: float,
    policy: HostMappingPolicy,
) -> None:
    if redshift > policy.domain_redshift_max:
        raise HostMappingError(
            STATUS_OUTSIDE_LOCAL_MODEL_DOMAIN,
            f"host {host_id}: redshift {redshift:.6g} exceeds the configured "
            f"local-model maximum {policy.domain_redshift_max:.6g}",
        )
    if not (
        policy.domain_smbh_mass_min_msun
        <= smbh_mass_msun
        <= policy.domain_smbh_mass_max_msun
    ):
        raise HostMappingError(
            STATUS_OUTSIDE_LOCAL_MODEL_DOMAIN,
            f"host {host_id}: SMBH mass {smbh_mass_msun:.6g} Msun is outside the "
            f"configured local-model domain "
            f"[{policy.domain_smbh_mass_min_msun:.6g}, "
            f"{policy.domain_smbh_mass_max_msun:.6g}] Msun",
        )
    if not (
        policy.domain_nsc_mass_min_msun
        <= nsc_mass_msun
        <= policy.domain_nsc_mass_max_msun
    ):
        raise HostMappingError(
            STATUS_OUTSIDE_LOCAL_MODEL_DOMAIN,
            f"host {host_id}: NSC mass {nsc_mass_msun:.6g} Msun is outside the "
            f"configured local-model domain "
            f"[{policy.domain_nsc_mass_min_msun:.6g}, "
            f"{policy.domain_nsc_mass_max_msun:.6g}] Msun",
        )


def resolve_host_parameters(
    host: HaloHost,
    base_config: IMRIDynConfig,
    policy: HostMappingPolicy | None = None,
) -> ResolvedHostParameters:
    """Resolve one host into the local simulation's host-dependent inputs.

    Raises :class:`HostMappingError` (carrying a batch status category) when the
    host cannot be turned into a usable local input.  This function is pure and
    cheap: it never touches the expensive physical simulation.
    """
    policy = policy or HostMappingPolicy()

    redshift = host.redshift
    if redshift is None or not np.isfinite(redshift):
        raise HostMappingError(
            STATUS_MISSING_REQUIRED_INPUT,
            f"redshift is unavailable from tree column '{TREE_COLUMN_REDSHIFT}'",
        )
    if redshift < 0.0:
        raise HostMappingError(
            STATUS_INVALID_HOST_INPUT,
            f"redshift must be nonnegative (got {redshift:.6g})",
        )

    smbh_mass_msun, smbh_source = _resolve_smbh_mass(host, base_config, policy)
    nsc_mass_msun, nsc_source = _resolve_nsc_mass(host, base_config, policy)
    metallicity_zsun, metallicity_source = _resolve_metallicity(
        host, base_config, policy
    )
    if not np.isfinite(metallicity_zsun) or metallicity_zsun < 0.0:
        raise HostMappingError(
            STATUS_INVALID_HOST_INPUT,
            f"resolved metallicity must be finite and nonnegative "
            f"(got {metallicity_zsun:.6g})",
        )

    _check_domain(host.host_id, redshift, smbh_mass_msun, nsc_mass_msun, policy)

    return ResolvedHostParameters(
        host_id=host.host_id,
        redshift=float(redshift),
        smbh_mass_msun=float(smbh_mass_msun),
        nsc_mass_msun=float(nsc_mass_msun),
        metallicity_zsun=float(metallicity_zsun),
        sources={
            "redshift": f"tree:{TREE_COLUMN_REDSHIFT}",
            "smbh_mass_msun": smbh_source,
            "nsc_mass_msun": nsc_source,
            "metallicity_zsun": metallicity_source,
        },
    )


def config_kwargs(config: IMRIDynConfig) -> dict[str, Any]:
    """Return the full constructor keyword mapping of an ``IMRIDynConfig``."""
    return {
        "smbh_masses_msun": tuple(float(mass) for mass in config.smbh_masses_msun),
        "nsc_masses_msun": tuple(float(mass) for mass in config.nsc_masses_msun),
        "num_repeats": config.num_repeats,
        "duty_cycle_myr": config.duty_cycle_myr,
        "output_dir": config.output_dir,
        "metallicity": config.metallicity,
        "initial_stellar_density_gamma": config.initial_stellar_density_gamma,
        "use_disk_inner_radius": config.use_disk_inner_radius,
        "lisa_min_gw_frequency_hz": config.lisa_min_gw_frequency_hz,
        "lisa_max_gw_frequency_hz": config.lisa_max_gw_frequency_hz,
        "max_gw_time_myr": config.max_gw_time_myr,
        "gw_eccentricity": config.gw_eccentricity,
        "redshift": config.redshift,
        "snr_threshold": config.snr_threshold,
        "lisa_mission_duration_years": config.lisa_mission_duration_years,
        "snr_window_placement": config.snr_window_placement,
        "detector_hubble_constant_km_s_mpc": (
            config.detector_hubble_constant_km_s_mpc
        ),
        "detector_omega_m": config.detector_omega_m,
        "cosmic_merge_time_myr": config.cosmic_merge_time_myr,
        "cosmic_merge_smbh_mass_factor": config.cosmic_merge_smbh_mass_factor,
        "cosmic_merge_radius_factor": config.cosmic_merge_radius_factor,
        "damping_cpu_count": config.damping_cpu_count,
        "checkpoint_grid_interval": config.checkpoint_grid_interval,
        "print_debug_results": config.print_debug_results,
        "base_seed": config.base_seed,
        "consumption_replenishment": config.consumption_replenishment,
        "nsc_radius_pc_at_reference_mass": config.nsc_radius_pc_at_reference_mass,
        "nsc_radius_mass_slope": config.nsc_radius_mass_slope,
        "nsc_radius_reference_mass_msun": config.nsc_radius_reference_mass_msun,
    }


def config_with_overrides(config: IMRIDynConfig, **overrides: Any) -> IMRIDynConfig:
    """Return a new ``IMRIDynConfig`` with the given constructor overrides.

    The base configuration is never mutated, so the result is safe to ship to a
    worker process.
    """
    kwargs = config_kwargs(config)
    unknown = set(overrides) - set(kwargs)
    if unknown:
        raise ValueError(f"unknown configuration override(s): {sorted(unknown)}")
    kwargs.update(overrides)
    return IMRIDynConfig(**kwargs)


__all__ = [
    "HostMappingError",
    "HostMappingPolicy",
    "ResolvedHostParameters",
    "METALLICITY_SOURCES",
    "METALLICITY_SOURCE_CONFIG",
    "METALLICITY_SOURCE_TREE",
    "NSC_MASS_SOURCES",
    "NSC_MASS_SOURCE_CONFIG",
    "NSC_MASS_SOURCE_STELLAR_SCALING",
    "SMBH_MASS_SOURCES",
    "SMBH_MASS_SOURCE_CONFIG",
    "SMBH_MASS_SOURCE_STELLAR_SCALING",
    "SMBH_MASS_SOURCE_TREE",
    "STATUS_INVALID_HOST_INPUT",
    "STATUS_MAPPING_UNAVAILABLE",
    "STATUS_MISSING_REQUIRED_INPUT",
    "STATUS_OUTSIDE_LOCAL_MODEL_DOMAIN",
    "config_kwargs",
    "config_with_overrides",
    "resolve_host_parameters",
]
