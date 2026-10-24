"""Programmatic single-halo interface for the dynamical IMRI workflow.

This module exposes the existing local simulation kernel
(:func:`workflows.workflow.run_simulation`) as one reusable, explicitly
parameterised, process-safe operation::

    result = run_single_halo(
        config=base_config,       # global model parameters
        host=host,                # host-dependent inputs (HaloHost or resolved)
        seed=seed,                # deterministic base seed for this run
        output_dir=output_dir,    # explicit output namespace
    )

It does **not** reimplement any physics.  ``run_single_halo`` builds one
resolved ``IMRIDynConfig`` (a 1 x 1 SMBH/NSC grid with the host's redshift and
metallicity) and calls the unchanged ``run_simulation`` on it, so the ordinary
CLI path and the merger-tree batch path execute exactly the same code.

What one invocation means
-------------------------
One call is one *complete* existing single-halo workflow for one host: the
kernel's own ``num_repeats`` Monte Carlo realisations are executed inside the
call (with the kernel's own ``derive_task_seed`` policy applied to the given
base seed) and averaged exactly as before.  Repeats are therefore *not*
expanded into separate jobs.
"""

from __future__ import annotations

import contextlib
import io
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    IMRI_AVERAGED_RESULTS_FILENAME,
    IMRI_REPEAT_RESULTS_FILENAME,
    IMRIDynConfig,
)
from .host_mapping import (
    HostMappingError,
    HostMappingPolicy,
    ResolvedHostParameters,
    STATUS_INVALID_HOST_INPUT,
    STATUS_MAPPING_UNAVAILABLE,
    STATUS_MISSING_REQUIRED_INPUT,
    STATUS_OUTSIDE_LOCAL_MODEL_DOMAIN,
    config_with_overrides,
    resolve_host_parameters,
)
from .merger_tree import HaloHost
from .workflow import run_simulation

STATUS_SUCCESS = "success"
STATUS_HOST_NOT_SELECTED = "host_not_selected"
STATUS_SIMULATION_FAILURE = "simulation_failure"
STATUS_NUMERICAL_FAILURE = "numerical_failure"
STATUS_OUTPUT_FAILURE = "output_failure"

#: Execution states of one single-halo job.  Only ``success`` means the local
#: physics was evaluated; everything else means it *could not* be evaluated and
#: must never be interpreted as a physical zero.
JOB_STATUSES: tuple[str, ...] = (
    STATUS_SUCCESS,
    STATUS_HOST_NOT_SELECTED,
    STATUS_MISSING_REQUIRED_INPUT,
    STATUS_INVALID_HOST_INPUT,
    STATUS_OUTSIDE_LOCAL_MODEL_DOMAIN,
    STATUS_MAPPING_UNAVAILABLE,
    STATUS_SIMULATION_FAILURE,
    STATUS_NUMERICAL_FAILURE,
    STATUS_OUTPUT_FAILURE,
)

RUN_LOG_FILENAME = "run.log"
MANUAL_HOST_ID = "manual"

#: Columns copied from the kernel's averaged-results table into the compact
#: per-job summary.  These are the quantities the existing project already
#: reports for one grid point.
SUMMARY_RESULT_COLUMNS: tuple[str, ...] = (
    "n_repeat",
    "rate_status",
    "detector_rate_status",
    "f_geo",
    "n_disk_cross",
    "n_source_initial",
    "n_source_remaining",
    "m_source_initial_msun",
    "m_source_remaining_msun",
    "n_replenished",
    "n_isco_consumed",
    "n_source_exhausted",
    "n_replenishment_unavailable",
    "n_emb_star",
    "n_emb_bh",
    "n_mfbh",
    "n_bh_candi",
    "n_imri",
    "n_imri_min",
    "n_imri_max",
    "m_imri_msun",
    "rate_imri_per_myr",
    "n_det",
    "n_nondet",
    "rate_det_per_myr",
    "n_detector_unavailable",
    "n_band_eval",
    "n_entry_in_domain",
    "n_entry_handoff",
    "n_entry_in_ISCO",
    "n_profile_error",
    "n_mig_eval",
    "n_mig_finite",
    "n_mig_trap",
    "n_mig_error",
)

_NUMERICAL_ERRORS: tuple[type[BaseException], ...] = (
    FloatingPointError,
    ZeroDivisionError,
    OverflowError,
    np.linalg.LinAlgError,
)


@dataclass(frozen=True)
class HaloSimulationResult:
    """Compact, picklable outcome of one single-halo invocation."""

    job_id: str
    host_id: str
    status: str
    seed: int | None
    output_path: str | None
    summary: dict[str, Any] = field(default_factory=dict)
    resolved: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    runtime_s: float | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == STATUS_SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def build_host_config(
    config: IMRIDynConfig,
    resolved: ResolvedHostParameters | None,
    *,
    seed: int | None = None,
    output_dir: str | Path | None = None,
) -> IMRIDynConfig:
    """Return the resolved single-halo configuration for one run.

    Global model parameters are inherited from ``config``; only the four
    host-dependent inputs, the base seed and the output directory are replaced.
    """
    overrides: dict[str, Any] = {}
    if resolved is not None:
        overrides.update(
            smbh_masses_msun=(resolved.smbh_mass_msun,),
            nsc_masses_msun=(resolved.nsc_mass_msun,),
            redshift=resolved.redshift,
            metallicity=resolved.metallicity_zsun,
        )
    if seed is not None:
        overrides["base_seed"] = int(seed)
    if output_dir is not None:
        overrides["output_dir"] = Path(output_dir)
    return config_with_overrides(config, **overrides)


def resolve_run_parameters(
    config: IMRIDynConfig,
    host: HaloHost | ResolvedHostParameters | None,
    policy: HostMappingPolicy | None = None,
) -> ResolvedHostParameters | None:
    """Resolve the host-dependent inputs without running any physics."""
    if host is None or isinstance(host, ResolvedHostParameters):
        return host
    return resolve_host_parameters(host, config, policy)


def summarise_output_dir(output_dir: Path) -> dict[str, Any]:
    """Collect the compact per-run summary from the kernel's own output files."""
    averaged = pd.read_csv(output_dir / IMRI_AVERAGED_RESULTS_FILENAME)
    if len(averaged.index) != 1:
        raise RuntimeError(
            "single-halo run must produce exactly one averaged grid point, "
            f"found {len(averaged.index)}"
        )
    row = averaged.iloc[0]
    summary: dict[str, Any] = {
        name: _json_scalar(row[name]) for name in SUMMARY_RESULT_COLUMNS
    }
    repeats = pd.read_csv(output_dir / IMRI_REPEAT_RESULTS_FILENAME)
    summary["n_sampled_mean"] = float(repeats["n_sampled"].mean())
    summary["task_seeds"] = [
        int(seed) for seed in repeats["task_seed"].tolist()
    ]
    return summary


def run_single_halo(
    config: IMRIDynConfig,
    host: HaloHost | ResolvedHostParameters | None = None,
    *,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    policy: HostMappingPolicy | None = None,
    job_id: str | None = None,
    log_to_file: bool = True,
) -> HaloSimulationResult:
    """Run the existing single-halo simulation once and return a structured result.

    Parameters
    ----------
    config
        Base configuration carrying every global model parameter.
    host
        Host-dependent inputs.  ``None`` runs ``config`` unchanged (the ordinary
        single-halo mode), a :class:`HaloHost` is resolved through ``policy``,
        and a :class:`ResolvedHostParameters` is used as is.
    seed
        Base seed for this run.  ``None`` keeps ``config.base_seed``.
    output_dir
        Explicit output namespace.  ``None`` keeps ``config.output_dir``.
    policy
        Host-mapping policy used when ``host`` is a raw :class:`HaloHost`.
    job_id
        Identity recorded in the result; defaults to the host identity.
    log_to_file
        Redirect the kernel's console output into ``<output_dir>/run.log`` so
        parallel workers do not interleave on one terminal.

    The function never raises for a per-host failure: every failure is mapped to
    one of :data:`JOB_STATUSES` and returned with its error message.
    """
    host_id = (
        host.host_id if host is not None else MANUAL_HOST_ID
    )
    job_id = job_id or host_id
    started_s = time.perf_counter()

    try:
        resolved = resolve_run_parameters(config, host, policy)
        run_config = build_host_config(
            config,
            resolved,
            seed=seed,
            output_dir=output_dir,
        )
    except HostMappingError as error:
        return HaloSimulationResult(
            job_id=job_id,
            host_id=host_id,
            status=error.status,
            seed=seed,
            output_path=None,
            error=error.message,
            runtime_s=time.perf_counter() - started_s,
        )
    except (TypeError, ValueError) as error:
        return HaloSimulationResult(
            job_id=job_id,
            host_id=host_id,
            status=STATUS_INVALID_HOST_INPUT,
            seed=seed,
            output_path=None,
            error=f"{type(error).__name__}: {error}",
            runtime_s=time.perf_counter() - started_s,
        )

    resolved_record = resolved.to_dict() if resolved is not None else {}
    run_output_dir = run_config.output_dir
    try:
        run_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return HaloSimulationResult(
            job_id=job_id,
            host_id=host_id,
            status=STATUS_OUTPUT_FAILURE,
            seed=run_config.base_seed,
            output_path=str(run_output_dir),
            resolved=resolved_record,
            error=f"{type(error).__name__}: {error}",
            runtime_s=time.perf_counter() - started_s,
        )

    log_context: Any
    if log_to_file:
        log_handle = (run_output_dir / RUN_LOG_FILENAME).open("w")
        log_context = contextlib.ExitStack()
        log_context.enter_context(log_handle)
        log_context.enter_context(contextlib.redirect_stdout(log_handle))
        log_context.enter_context(contextlib.redirect_stderr(log_handle))
    else:
        log_context = contextlib.nullcontext()

    status = STATUS_SUCCESS
    error_text: str | None = None
    summary: dict[str, Any] = {}
    try:
        with log_context:
            run_simulation(run_config)
        summary = summarise_output_dir(run_output_dir)
    except _NUMERICAL_ERRORS as error:
        status = STATUS_NUMERICAL_FAILURE
        error_text = f"{type(error).__name__}: {error}"
    except OSError as error:
        status = STATUS_OUTPUT_FAILURE
        error_text = f"{type(error).__name__}: {error}"
    except Exception as error:  # noqa: BLE001 - failures must stay isolated
        status = STATUS_SIMULATION_FAILURE
        error_text = f"{type(error).__name__}: {error}"
        trace = io.StringIO()
        traceback.print_exc(file=trace)
        with contextlib.suppress(OSError):
            with (run_output_dir / "error.txt").open("w") as handle:
                handle.write(trace.getvalue())

    return HaloSimulationResult(
        job_id=job_id,
        host_id=host_id,
        status=status,
        seed=run_config.base_seed,
        output_path=str(run_output_dir),
        summary=summary,
        resolved=resolved_record,
        error=error_text,
        runtime_s=time.perf_counter() - started_s,
    )


__all__ = [
    "HaloSimulationResult",
    "JOB_STATUSES",
    "MANUAL_HOST_ID",
    "RUN_LOG_FILENAME",
    "STATUS_HOST_NOT_SELECTED",
    "STATUS_NUMERICAL_FAILURE",
    "STATUS_OUTPUT_FAILURE",
    "STATUS_SIMULATION_FAILURE",
    "STATUS_SUCCESS",
    "SUMMARY_RESULT_COLUMNS",
    "build_host_config",
    "resolve_run_parameters",
    "run_single_halo",
    "summarise_output_dir",
]
