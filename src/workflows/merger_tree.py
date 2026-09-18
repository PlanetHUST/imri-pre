"""Read-only adapter between an external merger-tree CSV and the IMRI workflow.

This module is the *only* place that knows about merger-tree CSV column names.
Its single responsibility is::

    raw CSV record -> validated :class:`HaloHost`

It performs no physics.  It never rewrites, normalises, or duplicates the input
file, and it loads only the small subset of columns that the existing
single-halo simulation actually needs.

Verified field semantics (established from the supplied file itself; see
``docs/merger_tree_interface.md``):

``z``
    Host redshift.  Used directly as the local-model source redshift.
``m``
    Halo mass in solar masses (the supplied file ``galaxy_1e12.csv`` carries
    ``m ~ 1e12``, consistent with its name).  Used for host selection and
    provenance only; the local model has no halo-mass input.
``Mstar``
    Host stellar mass in solar masses.  Only used when an explicitly selected
    mapping derives the SMBH or NSC mass from it.
``Z``
    Absolute metal mass fraction of the host gas.  Verified numerically on the
    supplied file: ``Z == mZ / (Mhot + Mcold)`` to a relative accuracy of
    5e-13 over all 9999 rows, which fixes ``Z`` as a dimensionless mass
    fraction rather than a solar-normalised value.
``MBH``
    Central black-hole mass in solar masses.  NOTE: in the supplied file this
    column is identically zero for every row, so it supplies no usable SMBH
    mass.  The reader reports it faithfully; the mapping layer decides what to
    do about it.
``inode``
    Merger-tree node index (320 distinct values in the supplied file).
``br``
    Branch index (constant ``1.0`` in the supplied file).

Deliberately ignored columns: ``plc``, ``fc``, ``sf``, ``bh``, ``brf``,
``Mhot``, ``Mcold``, ``Mout``, ``mZ``, ``dM``, ``Q``, ``mlist``, ``dt``,
``gen``, ``MSF``, ``MBH2``, ``Mgas``, ``Mgas2``, ``Mtde``, ``Mtde2``,
``Macc2``.  None of them is an input of the existing single-halo model and
their semantics are not established by the supplied materials.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

TREE_COLUMN_REDSHIFT = "z"
TREE_COLUMN_HALO_MASS = "m"
TREE_COLUMN_STELLAR_MASS = "Mstar"
TREE_COLUMN_METALLICITY = "Z"
TREE_COLUMN_SMBH_MASS = "MBH"
TREE_COLUMN_NODE = "inode"
TREE_COLUMN_BRANCH = "br"

#: The complete set of merger-tree columns this interface ever reads.
REQUIRED_TREE_COLUMNS: tuple[str, ...] = (
    TREE_COLUMN_REDSHIFT,
    TREE_COLUMN_HALO_MASS,
    TREE_COLUMN_STELLAR_MASS,
    TREE_COLUMN_METALLICITY,
    TREE_COLUMN_SMBH_MASS,
    TREE_COLUMN_NODE,
    TREE_COLUMN_BRANCH,
)

#: Columns present in the placeholder schema that are intentionally not read.
IGNORED_TREE_COLUMNS: tuple[str, ...] = (
    "plc",
    "br_b4",
    "fc",
    "sf",
    "bh",
    "brf",
    "Mhot",
    "Mcold",
    "Mout",
    "mZ",
    "dM",
    "Q",
    "mlist",
    "dt",
    "gen",
    "MSF",
    "MBH2",
    "Mgas",
    "Mgas2",
    "Mtde",
    "Mtde2",
    "Macc2",
)

DEFAULT_CHUNK_SIZE = 50_000


class MergerTreeError(RuntimeError):
    """Raised when the merger-tree CSV cannot be used as a simulation input."""


@dataclass(frozen=True)
class HaloHost:
    """One merger-tree record reduced to the quantities the local model may use.

    Masses are in solar masses, ``metallicity_mass_fraction`` is the absolute
    (dimensionless) metal mass fraction carried by the tree, and
    ``record_index`` is the zero-based position of the row in the CSV.  The
    record index is assigned from a sequential counter during streaming, so it
    is independent of the chunk size used to read the file.
    """

    host_id: str
    redshift: float
    record_index: int
    node_id: int | str | None = None
    branch_id: int | str | None = None
    halo_mass_msun: float | None = None
    smbh_mass_msun: float | None = None
    stellar_mass_msun: float | None = None
    metallicity_mass_fraction: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, JSON/CSV friendly mapping of the host record."""
        return asdict(self)


@dataclass(frozen=True)
class HostSelection:
    """Declarative host filter applied while the CSV is streamed.

    ``None`` disables a bound.  ``max_hosts`` truncates the stream after the
    given number of accepted hosts; because records are streamed in file order
    the truncated set is independent of the chunk size.
    """

    nodes: tuple[float, ...] | None = None
    branches: tuple[float, ...] | None = None
    redshift_min: float | None = None
    redshift_max: float | None = None
    smbh_mass_min_msun: float | None = None
    smbh_mass_max_msun: float | None = None
    halo_mass_min_msun: float | None = None
    halo_mass_max_msun: float | None = None
    max_hosts: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReaderCounters:
    """Compact counters describing one streaming pass over the CSV."""

    rows_seen: int = 0
    rows_selected: int = 0
    missing_required_field: int = 0
    invalid_redshift: int = 0
    invalid_halo_mass: int = 0
    invalid_bh_mass: int = 0
    outside_selection: int = 0
    truncated_by_max_hosts: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _identity_token(value: float | int | str | None) -> str:
    """Return a stable, filesystem-safe token for a node or branch identity."""
    if value is None:
        return "na"
    if isinstance(value, str):
        cleaned = "".join(
            character if character.isalnum() else "-" for character in value
        )
        return cleaned or "na"
    numeric = float(value)
    if np.isfinite(numeric) and float(numeric).is_integer():
        return f"{int(numeric):06d}"
    return f"{numeric:.6g}".replace(".", "p").replace("+", "").replace("-", "m")


def _normalise_identity(value: float) -> int | float | None:
    """Return an int for integral identities, a float otherwise."""
    if value is None or not np.isfinite(value):
        return None
    return int(value) if float(value).is_integer() else float(value)


def make_host_id(
    node_id: int | float | str | None,
    branch_id: int | float | str | None,
    record_index: int,
) -> str:
    """Build the deterministic host identity used for jobs and output paths."""
    return (
        f"n{_identity_token(node_id)}"
        f"_b{_identity_token(branch_id)}"
        f"_r{record_index:07d}"
    )


def read_tree_columns(path: Path | str) -> list[str]:
    """Return the CSV header without loading any data rows."""
    tree_path = Path(path)
    if not tree_path.is_file():
        raise MergerTreeError(f"merger-tree file not found: {tree_path}")
    try:
        with tree_path.open("r", newline="") as handle:
            header = next(csv.reader(handle), None)
    except OSError as error:  # pragma: no cover - depends on the filesystem
        raise MergerTreeError(f"merger-tree file is not readable: {error}") from error
    if not header:
        raise MergerTreeError(f"merger-tree file has no header: {tree_path}")
    return [name.strip() for name in header]


def _matches(value: float | None, low: float | None, high: float | None) -> bool:
    if low is None and high is None:
        return True
    if value is None or not np.isfinite(value):
        return False
    if low is not None and value < low:
        return False
    if high is not None and value > high:
        return False
    return True


def _in_set(value: float | None, allowed: Sequence[float] | None) -> bool:
    if allowed is None:
        return True
    if value is None or not np.isfinite(value):
        return False
    return any(np.isclose(value, candidate, rtol=0.0, atol=1e-9) for candidate in allowed)


@dataclass
class MergerTreeReader:
    """Stream selected hosts out of a large, read-only merger-tree CSV.

    The reader loads only :data:`REQUIRED_TREE_COLUMNS` and iterates the file in
    chunks, so peak memory is bounded by ``chunk_size`` rather than by the file
    size.  It never writes to, copies, or reorders the input file.
    """

    path: Path
    selection: HostSelection = field(default_factory=HostSelection)
    chunk_size: int = DEFAULT_CHUNK_SIZE

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.chunk_size <= 0:
            raise MergerTreeError("chunk size must be positive")
        self.counters = ReaderCounters()
        header = read_tree_columns(self.path)
        missing = [name for name in REQUIRED_TREE_COLUMNS if name not in header]
        if missing:
            raise MergerTreeError(
                "merger-tree file is missing required columns: " + ", ".join(missing)
            )
        self.header = tuple(header)

    def iter_hosts(self) -> Iterator[HaloHost]:
        """Yield the selected hosts in file order, resetting the counters."""
        self.counters = ReaderCounters()
        record_index = -1
        max_hosts = self.selection.max_hosts

        reader = pd.read_csv(
            self.path,
            usecols=list(REQUIRED_TREE_COLUMNS),
            chunksize=self.chunk_size,
        )
        for chunk in reader:
            numeric = {
                name: pd.to_numeric(chunk[name], errors="coerce").to_numpy(dtype=float)
                for name in REQUIRED_TREE_COLUMNS
            }
            chunk_length = len(chunk.index)
            for offset in range(chunk_length):
                record_index += 1
                self.counters.rows_seen += 1
                host = self._build_host(numeric, offset, record_index)
                if host is None:
                    continue
                if not self._selected(host):
                    self.counters.outside_selection += 1
                    continue
                if max_hosts is not None and self.counters.rows_selected >= max_hosts:
                    self.counters.truncated_by_max_hosts = 1
                    return
                self.counters.rows_selected += 1
                yield host
            if (
                max_hosts is not None
                and self.counters.rows_selected >= max_hosts
            ):
                self.counters.truncated_by_max_hosts = 1
                return

    def _build_host(
        self,
        numeric: dict[str, np.ndarray],
        offset: int,
        record_index: int,
    ) -> HaloHost | None:
        redshift = float(numeric[TREE_COLUMN_REDSHIFT][offset])
        if not np.isfinite(redshift):
            self.counters.missing_required_field += 1
            return None
        if redshift < 0.0:
            self.counters.invalid_redshift += 1
            return None

        halo_mass = float(numeric[TREE_COLUMN_HALO_MASS][offset])
        if np.isfinite(halo_mass) and halo_mass < 0.0:
            self.counters.invalid_halo_mass += 1
            return None

        smbh_mass = float(numeric[TREE_COLUMN_SMBH_MASS][offset])
        if np.isfinite(smbh_mass) and smbh_mass < 0.0:
            self.counters.invalid_bh_mass += 1
            return None

        stellar_mass = float(numeric[TREE_COLUMN_STELLAR_MASS][offset])
        metallicity = float(numeric[TREE_COLUMN_METALLICITY][offset])
        node_id = _normalise_identity(float(numeric[TREE_COLUMN_NODE][offset]))
        branch_id = _normalise_identity(float(numeric[TREE_COLUMN_BRANCH][offset]))

        return HaloHost(
            host_id=make_host_id(node_id, branch_id, record_index),
            redshift=redshift,
            record_index=record_index,
            node_id=node_id,
            branch_id=branch_id,
            halo_mass_msun=halo_mass if np.isfinite(halo_mass) else None,
            smbh_mass_msun=smbh_mass if np.isfinite(smbh_mass) else None,
            stellar_mass_msun=stellar_mass if np.isfinite(stellar_mass) else None,
            metallicity_mass_fraction=(
                metallicity if np.isfinite(metallicity) else None
            ),
        )

    def _selected(self, host: HaloHost) -> bool:
        selection = self.selection
        if not _in_set(host.node_id, selection.nodes):
            return False
        if not _in_set(host.branch_id, selection.branches):
            return False
        if not _matches(
            host.redshift, selection.redshift_min, selection.redshift_max
        ):
            return False
        if selection.halo_mass_min_msun is not None or (
            selection.halo_mass_max_msun is not None
        ):
            if not _matches(
                host.halo_mass_msun,
                selection.halo_mass_min_msun,
                selection.halo_mass_max_msun,
            ):
                return False
        if selection.smbh_mass_min_msun is not None or (
            selection.smbh_mass_max_msun is not None
        ):
            if not _matches(
                host.smbh_mass_msun,
                selection.smbh_mass_min_msun,
                selection.smbh_mass_max_msun,
            ):
                return False
        return True


def hosts_to_frame(hosts: Sequence[HaloHost]) -> pd.DataFrame:
    """Return a compact table of selected hosts for output and provenance."""
    return pd.DataFrame.from_records(
        [host.to_dict() for host in hosts],
        columns=[
            "host_id",
            "record_index",
            "node_id",
            "branch_id",
            "redshift",
            "halo_mass_msun",
            "smbh_mass_msun",
            "stellar_mass_msun",
            "metallicity_mass_fraction",
        ],
    )
