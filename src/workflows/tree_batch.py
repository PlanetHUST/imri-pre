"""Deterministic parallel execution of single-halo jobs built from a merger tree.

Pipeline implemented here::

    MergerTreeReader -> HaloHost -> HaloSimulationJob -> run_single_halo
                                          |
                              ProcessPoolExecutor (workers > 1)
                                          |
                                HaloSimulationResult -> compact outputs

Job definition
--------------
One :class:`HaloSimulationJob` is one selected merger-tree host.  Executing it
means one complete invocation of the existing single-halo workflow for that
host, including the kernel's own ``num_repeats`` Monte Carlo repeats.

Job identity
------------
``job_id == host.host_id`` = ``n<node>_b<branch>_r<record_index>``; it is built
from the merger-tree node/branch indices and the zero-based row position, so it
is unique, deterministic, chunk-size independent and filesystem safe.

Seed derivation
---------------
``seed = SeedSequence([master_seed, node_key, branch_key, record_index])``
(first ``uint32`` word).  The kernel then applies its own ``derive_task_seed``
policy per repeat.  Python's process-dependent ``hash`` is never used, and no
RNG stream is shared between workers: each worker seeds the global NumPy RNG
from its job's seed inside the unchanged kernel.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import platform
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from .config import IMRIDynConfig
from .host_mapping import HostMappingPolicy, config_kwargs, config_with_overrides
from .merger_tree import (
    HaloHost,
    HostSelection,
    IGNORED_TREE_COLUMNS,
    MergerTreeReader,
    REQUIRED_TREE_COLUMNS,
    hosts_to_frame,
)
from .single_halo import (
    HaloSimulationResult,
    JOB_STATUSES,
    STATUS_SUCCESS,
    run_single_halo,
)

CONFIG_DIRNAME = "config"
HOSTS_DIRNAME = "hosts"
JOBS_DIRNAME = "jobs"
SIMULATIONS_DIRNAME = "simulations"
DIAGNOSTICS_DIRNAME = "diagnostics"
FIGURES_DIRNAME = "figures"
RESOLVED_CONFIG_FILENAME = "resolved_config.json"
PROVENANCE_FILENAME = "provenance.json"
SELECTED_HOSTS_FILENAME = "selected_hosts.csv"
JOB_SUMMARY_FILENAME = "job_summary.csv"
BATCH_SUMMARY_FILENAME = "batch_summary.json"
SUMMARY_FILENAME = "summary.json"

DEFAULT_MASTER_SEED = 0
DEFAULT_WORKERS = 1
DEFAULT_MP_CONTEXT = "spawn"
DEFAULT_BATCH_DAMPING_CPU_COUNT = 1
PROVENANCE_HASH_BYTES = 65536

JOB_SUMMARY_COLUMNS: tuple[str, ...] = (
    "job_id",
    "host_id",
    "record_index",
    "node_id",
    "branch_id",
    "status",
    "seed",
    "output_path",
    "runtime_s",
    "error",
    "redshift",
    "smbh_mass_msun",
    "nsc_mass_msun",
    "metallicity_zsun",
    "redshift_source",
    "smbh_mass_msun_source",
    "nsc_mass_msun_source",
    "metallicity_zsun_source",
    "n_repeat",
    "task_seeds",
    "rate_status",
    "detector_rate_status",
    "n_sampled_mean",
    "f_geo",
    "n_disk_cross",
    "n_emb_star",
    "n_emb_bh",
    "n_mfbh",
    "n_bh_candi",
    "n_imri",
    "rate_imri_per_myr",
    "n_det",
    "rate_det_per_myr",
    "n_entry_in_ISCO",
    "n_profile_error",
    "n_mig_error",
)


class BatchExecutionError(RuntimeError):
    """Raised when the batch itself (not one host) cannot run safely."""


@dataclass(frozen=True)
class HaloSimulationJob:
    """One explicit, picklable single-halo job."""

    job_id: str
    host: HaloHost
    seed: int
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        host = record.pop("host")
        record.update({f"host_{key}": value for key, value in host.items()})
        return record


def _identity_key(value: int | float | str | None) -> int:
    """Return a stable nonnegative integer key for a node/branch identity."""
    if value is None:
        return 0
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return abs(int(value))
    if isinstance(value, float) and float(value).is_integer():
        return abs(int(value))
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def derive_host_seed(master_seed: int, host: HaloHost) -> int:
    """Derive the deterministic base seed of one host from stable identities."""
    if master_seed < 0:
        raise ValueError("master seed must be nonnegative")
    if host.record_index < 0:
        raise ValueError("record index must be nonnegative")
    sequence = np.random.SeedSequence(
        [
            int(master_seed),
            _identity_key(host.node_id),
            _identity_key(host.branch_id),
            int(host.record_index),
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def make_job_id(host: HaloHost) -> str:
    """Return the deterministic job identity of one host (one job per host)."""
    return host.host_id


def build_job(
    host: HaloHost,
    master_seed: int,
    simulations_dir: Path | str,
) -> HaloSimulationJob:
    """Turn one host into one explicit simulation job."""
    job_id = make_job_id(host)
    return HaloSimulationJob(
        job_id=job_id,
        host=host,
        seed=derive_host_seed(master_seed, host),
        output_dir=str(Path(simulations_dir) / job_id),
    )


def iter_halo_jobs(
    hosts: Iterable[HaloHost],
    master_seed: int,
    simulations_dir: Path | str,
) -> Iterator[HaloSimulationJob]:
    """Lazily build jobs from a host stream, rejecting duplicate identities."""
    seen: set[str] = set()
    for host in hosts:
        job = build_job(host, master_seed, simulations_dir)
        if job.job_id in seen:
            raise BatchExecutionError(f"duplicate job identity: {job.job_id}")
        seen.add(job.job_id)
        yield job


def _execute_job(
    payload: tuple[HaloSimulationJob, IMRIDynConfig, HostMappingPolicy],
) -> HaloSimulationResult:
    """Worker entry point: one job, one explicit config, one output namespace."""
    job, base_config, policy = payload
    return run_single_halo(
        base_config,
        job.host,
        seed=job.seed,
        output_dir=job.output_dir,
        policy=policy,
        job_id=job.job_id,
        log_to_file=True,
    )


def run_jobs(
    jobs: Iterable[HaloSimulationJob],
    base_config: IMRIDynConfig,
    policy: HostMappingPolicy,
    workers: int = DEFAULT_WORKERS,
    *,
    mp_context: str = DEFAULT_MP_CONTEXT,
    on_result: Callable[[HaloSimulationResult, int], None] | None = None,
) -> list[HaloSimulationResult]:
    """Execute jobs serially (``workers == 1``) or in a process pool.

    Results are returned in *submission* order regardless of completion order,
    and every job's outcome is fully determined by its own seed and inputs, so
    the returned list is independent of the worker count.  Jobs are submitted
    from the iterator with a bounded in-flight window so a long host stream is
    never materialised in memory.
    """
    if workers < 1:
        raise ValueError("worker count must be positive")

    results: list[HaloSimulationResult] = []
    if workers == 1:
        for index, job in enumerate(jobs):
            result = _execute_job((job, base_config, policy))
            results.append(result)
            if on_result is not None:
                on_result(result, index)
        return results

    ordered: dict[int, HaloSimulationResult] = {}
    window = 2 * workers
    context = multiprocessing.get_context(mp_context)
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        pending: dict[Future[HaloSimulationResult], int] = {}
        job_iterator = iter(jobs)
        submitted = 0
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < window:
                try:
                    job = next(job_iterator)
                except StopIteration:
                    exhausted = True
                    break
                future = executor.submit(_execute_job, (job, base_config, policy))
                pending[future] = submitted
                submitted += 1
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                result = future.result()
                ordered[index] = result
                if on_result is not None:
                    on_result(result, index)
    return [ordered[index] for index in sorted(ordered)]


@dataclass(frozen=True)
class BatchSettings:
    """Engineering controls of one batch run (never physical parameters)."""

    workers: int = DEFAULT_WORKERS
    chunk_size: int | None = None
    master_seed: int = DEFAULT_MASTER_SEED
    mp_context: str = DEFAULT_MP_CONTEXT
    make_figures: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchOutcome:
    """Everything the CLI needs after a batch run, plus compact tables."""

    output_root: Path
    results: list[HaloSimulationResult]
    reader_counters: dict[str, int]
    status_counts: dict[str, int]
    runtime_s: float
    job_summary: pd.DataFrame = field(repr=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)


def _tree_fingerprint(tree_path: Path) -> dict[str, Any]:
    """Identify the input tree without scanning or copying the full file."""
    stat = tree_path.stat()
    digest = hashlib.sha256()
    with tree_path.open("rb") as handle:
        digest.update(handle.read(PROVENANCE_HASH_BYTES))
    return {
        "path": str(tree_path.resolve()),
        "size_bytes": stat.st_size,
        "modified_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "sha256_first_bytes": digest.hexdigest(),
        "sha256_first_bytes_count": min(PROVENANCE_HASH_BYTES, stat.st_size),
    }


def _code_version() -> dict[str, Any]:
    try:
        package_version = importlib_metadata.version("bsi-trap")
    except importlib_metadata.PackageNotFoundError:
        package_version = None
    return {
        "package": "bsi-trap",
        "package_version": package_version,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
    }


def result_to_summary_row(result: HaloSimulationResult, job: HaloSimulationJob) -> dict[str, Any]:
    row: dict[str, Any] = {
        "job_id": result.job_id,
        "host_id": result.host_id,
        "record_index": job.host.record_index,
        "node_id": job.host.node_id,
        "branch_id": job.host.branch_id,
        "status": result.status,
        "seed": result.seed,
        "output_path": result.output_path,
        "runtime_s": result.runtime_s,
        "error": result.error,
    }
    for key in (
        "redshift",
        "smbh_mass_msun",
        "nsc_mass_msun",
        "metallicity_zsun",
        "redshift_source",
        "smbh_mass_msun_source",
        "nsc_mass_msun_source",
        "metallicity_zsun_source",
    ):
        row[key] = result.resolved.get(key)
    for key in JOB_SUMMARY_COLUMNS:
        if key in row:
            continue
        value = result.summary.get(key)
        if key == "task_seeds" and value is not None:
            value = ";".join(str(item) for item in value)
        row[key] = value
    return row


def run_tree_batch(
    tree_path: Path | str,
    base_config: IMRIDynConfig,
    output_root: Path | str,
    *,
    selection: HostSelection | None = None,
    policy: HostMappingPolicy | None = None,
    settings: BatchSettings | None = None,
    config_source: str | None = None,
    progress: Callable[[str], None] | None = print,
) -> BatchOutcome:
    """Run the full merger-tree batch pipeline and write host-indexed outputs.

    Output layout (``output_root``)::

        config/resolved_config.json   base configuration + mapping policy
        config/provenance.json        tree fingerprint, selection, seeds, versions
        hosts/selected_hosts.csv      compact table of selected hosts
        jobs/job_summary.csv          one row per job: identity, seed, status, results
        simulations/<job_id>/         unchanged single-halo outputs + run.log
        diagnostics/batch_summary.json reader counters and status counts
        summary.json                  compact top-level summary
        figures/                      interface validation figures (optional)
    """
    selection = selection or HostSelection()
    policy = policy or HostMappingPolicy()
    settings = settings or BatchSettings()
    tree_path = Path(tree_path)
    output_root = Path(output_root)

    say = progress if progress is not None else (lambda _message: None)
    started_s = time.perf_counter()

    for name in (
        CONFIG_DIRNAME,
        HOSTS_DIRNAME,
        JOBS_DIRNAME,
        SIMULATIONS_DIRNAME,
        DIAGNOSTICS_DIRNAME,
    ):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    simulations_dir = output_root / SIMULATIONS_DIRNAME

    # In batch mode the inner damping pool of every host run would otherwise
    # default to the full machine, oversubscribing it workers-fold.
    if base_config.damping_cpu_count is None:
        base_config = config_with_overrides(
            base_config, damping_cpu_count=DEFAULT_BATCH_DAMPING_CPU_COUNT
        )

    reader_kwargs: dict[str, Any] = {"path": tree_path, "selection": selection}
    if settings.chunk_size is not None:
        reader_kwargs["chunk_size"] = settings.chunk_size
    reader = MergerTreeReader(**reader_kwargs)

    say(f"[tree-batch] tree: {tree_path}")
    say(f"[tree-batch] required columns: {', '.join(REQUIRED_TREE_COLUMNS)}")
    say(f"[tree-batch] workers: {settings.workers}, master seed: {settings.master_seed}")

    jobs_by_id: dict[str, HaloSimulationJob] = {}
    hosts_path = output_root / HOSTS_DIRNAME / SELECTED_HOSTS_FILENAME
    hosts_columns = list(hosts_to_frame([]).columns)
    with hosts_path.open("w") as hosts_handle:
        hosts_handle.write(",".join(hosts_columns) + "\n")

        def recorded_jobs() -> Iterator[HaloSimulationJob]:
            for job in iter_halo_jobs(
                reader.iter_hosts(), settings.master_seed, simulations_dir
            ):
                jobs_by_id[job.job_id] = job
                frame = hosts_to_frame([job.host])
                frame.to_csv(hosts_handle, header=False, index=False)
                yield job

        def announce(result: HaloSimulationResult, index: int) -> None:
            runtime = f"{result.runtime_s:.1f}s" if result.runtime_s is not None else "-"
            detail = f" ({result.error})" if result.error else ""
            say(
                f"[tree-batch] job {index + 1} {result.job_id}: "
                f"{result.status} seed={result.seed} {runtime}{detail}"
            )

        results = run_jobs(
            recorded_jobs(),
            base_config,
            policy,
            settings.workers,
            mp_context=settings.mp_context,
            on_result=announce,
        )

    runtime_s = time.perf_counter() - started_s
    status_counts = {status: 0 for status in JOB_STATUSES}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1

    rows = [result_to_summary_row(result, jobs_by_id[result.job_id]) for result in results]
    job_summary = pd.DataFrame.from_records(rows, columns=list(JOB_SUMMARY_COLUMNS))
    job_summary.to_csv(output_root / JOBS_DIRNAME / JOB_SUMMARY_FILENAME, index=False)

    reader_counters = reader.counters.to_dict()
    reader_counters["jobs_created"] = len(results)

    resolved_config_payload = {
        "base_config": config_kwargs(base_config),
        "host_mapping_policy": policy.to_dict(),
        "host_selection": selection.to_dict(),
        "batch_settings": settings.to_dict(),
        "host_dependent_parameters": [
            "redshift",
            "smbh_mass_msun",
            "nsc_mass_msun",
            "metallicity_zsun",
        ],
    }
    _write_json(output_root / CONFIG_DIRNAME / RESOLVED_CONFIG_FILENAME, resolved_config_payload)

    provenance_payload = {
        "created_iso": datetime.now().isoformat(),
        "tree": _tree_fingerprint(tree_path),
        "tree_columns_required": list(REQUIRED_TREE_COLUMNS),
        "tree_columns_ignored": list(IGNORED_TREE_COLUMNS),
        "tree_columns_present": list(reader.header),
        "config_source": config_source,
        "selection": selection.to_dict(),
        "host_mapping_policy": policy.to_dict(),
        "master_seed": settings.master_seed,
        "seed_policy": (
            "numpy.random.SeedSequence([master_seed, node_key, branch_key, "
            "record_index]) -> host base seed; kernel derive_task_seed(base_seed, "
            "0, 0, repeat_index) per repeat"
        ),
        "job_identity_policy": "job_id = host_id = n<node>_b<branch>_r<record_index>",
        "workers": settings.workers,
        "mp_context": settings.mp_context,
        "damping_cpu_count_per_job": base_config.damping_cpu_count,
        "code": _code_version(),
        "jobs": [
            {
                "job_id": result.job_id,
                "host_id": result.host_id,
                "seed": result.seed,
                "status": result.status,
                "output_path": result.output_path,
            }
            for result in results
        ],
        "scope_note": (
            "Host-level single-halo results only; no cosmic-rate, comoving "
            "density or all-sky aggregation is performed."
        ),
    }
    _write_json(output_root / CONFIG_DIRNAME / PROVENANCE_FILENAME, provenance_payload)

    batch_summary_payload = {
        "reader_counters": reader_counters,
        "status_counts": status_counts,
        "jobs_total": len(results),
        "jobs_succeeded": status_counts.get(STATUS_SUCCESS, 0),
        "jobs_not_evaluated": len(results) - status_counts.get(STATUS_SUCCESS, 0),
        "physical_zero_note": (
            "A job with status 'success' and n_imri == 0 is a physical zero; any "
            "other status means the local model could not be evaluated."
        ),
        "runtime_s": runtime_s,
        "workers": settings.workers,
    }
    _write_json(
        output_root / DIAGNOSTICS_DIRNAME / BATCH_SUMMARY_FILENAME,
        batch_summary_payload,
    )

    succeeded = job_summary[job_summary["status"] == STATUS_SUCCESS]
    summary_payload = {
        "tree_path": str(tree_path),
        "output_root": str(output_root),
        "jobs_total": len(results),
        "status_counts": status_counts,
        "reader_counters": reader_counters,
        "succeeded_hosts_with_imri_candidates": int(
            (pd.to_numeric(succeeded["n_imri"], errors="coerce") > 0).sum()
        )
        if len(succeeded.index)
        else 0,
        "succeeded_hosts_with_physical_zero_imri": int(
            (pd.to_numeric(succeeded["n_imri"], errors="coerce") == 0).sum()
        )
        if len(succeeded.index)
        else 0,
        "runtime_s": runtime_s,
        "workers": settings.workers,
        "master_seed": settings.master_seed,
    }
    _write_json(output_root / SUMMARY_FILENAME, summary_payload)

    if settings.make_figures:
        try:
            from viz.batch_plots import write_batch_figures

            write_batch_figures(output_root, job_summary, status_counts)
        except Exception as error:  # noqa: BLE001 - figures must never break a batch
            say(f"[tree-batch] figure generation skipped: {type(error).__name__}: {error}")

    say(
        f"[tree-batch] done: {len(results)} jobs, "
        f"{status_counts.get(STATUS_SUCCESS, 0)} succeeded, "
        f"{runtime_s:.1f} s, outputs in {output_root}"
    )
    return BatchOutcome(
        output_root=output_root,
        results=results,
        reader_counters=reader_counters,
        status_counts=status_counts,
        runtime_s=runtime_s,
        job_summary=job_summary,
    )


__all__ = [
    "BatchExecutionError",
    "BatchOutcome",
    "BatchSettings",
    "DEFAULT_MASTER_SEED",
    "DEFAULT_WORKERS",
    "HaloSimulationJob",
    "JOB_SUMMARY_COLUMNS",
    "build_job",
    "derive_host_seed",
    "iter_halo_jobs",
    "make_job_id",
    "run_jobs",
    "run_tree_batch",
]
