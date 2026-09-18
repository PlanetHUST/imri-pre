"""Canonical migration-trap identification from a signed torque profile.

A migration trap is **derived**, never hard-coded.  It is a radius where the
total migration torque changes sign, so that migration converges onto it from
both sides:

``inner torque > 0`` (pushed out)  and  ``outer torque < 0`` (pushed in)
    -> **stable trap**: migrators accumulate here.

``inner torque < 0``  and  ``outer torque > 0``
    -> **anti-trap** (unstable zero): migrators are pushed away from it.

Both root types are located by linear interpolation of the torque between the
two bracketing radial samples.

Numerical artefacts
-------------------
A tabulated torque profile can change sign twice within a couple of grid cells
purely from interpolation noise.  Such pairs are not physical traps.
``min_separation_cells`` discards a group of opposite-type roots whose mutual
separation is smaller than that many local radial-grid intervals, which lets
callers distinguish

    torque zero -> stable trap / unstable zero / unresolved numerical wiggle

``boundary`` roots (a sign change in the first or last interval of the profile)
are reported through :attr:`TorqueTrapRadii.boundary_radius_rg` because they may
reflect the truncation of the disk solution rather than disk physics.

Units
-----
Radii are in gravitational radii ``r_g``; torque may be supplied in any
consistent unit because only its sign and linear interpolation are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_MIN_TRAP_SEPARATION_CELLS = 10
STABLE_TRAP_KIND = 1
ANTI_TRAP_KIND = -1


@dataclass(frozen=True)
class TorqueTrapRadii:
    """Torque-zero radii [r_g] separated by their stability interpretation."""

    stable_radius_rg: np.ndarray
    anti_trap_radius_rg: np.ndarray
    boundary_radius_rg: np.ndarray
    unresolved_radius_rg: np.ndarray

    @property
    def trap_count(self) -> int:
        return int(self.stable_radius_rg.size)

    @property
    def innermost_stable_radius_rg(self) -> float:
        return (
            float(np.min(self.stable_radius_rg))
            if self.stable_radius_rg.size
            else float("nan")
        )

    @property
    def outermost_stable_radius_rg(self) -> float:
        return (
            float(np.max(self.stable_radius_rg))
            if self.stable_radius_rg.size
            else float("nan")
        )


def _empty_torque_traps() -> TorqueTrapRadii:
    empty = np.array([], dtype=float)
    return TorqueTrapRadii(empty, empty.copy(), empty.copy(), empty.copy())


def _resolved_crossing_mask(
    root_radius_rg: np.ndarray,
    crossing_kind: np.ndarray,
    local_width_rg: np.ndarray,
    min_separation_cells: float,
) -> np.ndarray:
    """Return which roots survive the minimum-separation filter."""
    keep = np.ones(root_radius_rg.size, dtype=bool)
    if min_separation_cells <= 0.0 or root_radius_rg.size < 2:
        return keep

    group = [0]

    def flush_group() -> None:
        if len(group) < 2:
            return
        group_kind = crossing_kind[np.asarray(group, dtype=int)]
        if np.unique(group_kind).size < 2:
            return
        keep[group] = False

    for crossing_index in range(1, root_radius_rg.size):
        previous_index = group[-1]
        separation_rg = root_radius_rg[crossing_index] - root_radius_rg[previous_index]
        local_resolution_rg = min(
            local_width_rg[crossing_index],
            local_width_rg[previous_index],
        )
        if separation_rg < min_separation_cells * local_resolution_rg:
            group.append(crossing_index)
            continue
        flush_group()
        group = [crossing_index]
    flush_group()
    return keep


def classify_torque_traps(
    radius_rg: np.ndarray,
    torque_values: np.ndarray,
    *,
    min_separation_cells: float = DEFAULT_MIN_TRAP_SEPARATION_CELLS,
) -> TorqueTrapRadii:
    """Locate and classify the torque zeros of one migration torque profile.

    Parameters
    ----------
    radius_rg
        Radial grid [r_g]; need not be sorted.
    torque_values
        Signed total migration torque on the same grid.  Positive means
        outward.
    min_separation_cells
        Opposite-type roots closer than this many local grid intervals are
        reported as ``unresolved_radius_rg`` instead of traps.

    Returns
    -------
    TorqueTrapRadii
        Stable traps, anti-traps, boundary crossings and unresolved wiggles.
    """
    radius_rg = np.asarray(radius_rg, dtype=float)
    values = np.asarray(torque_values, dtype=float)
    min_separation_cells = float(min_separation_cells)

    finite = np.isfinite(radius_rg) & np.isfinite(values) & (values != 0.0)
    radius_rg = radius_rg[finite]
    values = values[finite]

    if (
        values.size < 2
        or not np.isfinite(min_separation_cells)
        or min_separation_cells < 0.0
    ):
        return _empty_torque_traps()

    order = np.argsort(radius_rg)
    radius_rg = radius_rg[order]
    values = values[order]

    candidate_roots_rg: list[float] = []
    candidate_kinds: list[int] = []
    local_width_rg: list[float] = []
    is_boundary: list[bool] = []
    last_interval = values.size - 2
    for index in np.where(np.sign(values[1:]) != np.sign(values[:-1]))[0]:
        inner_radius_rg, outer_radius_rg = radius_rg[index], radius_rg[index + 1]
        inner_torque_value, outer_torque_value = values[index], values[index + 1]

        root_rg = inner_radius_rg - inner_torque_value * (
            outer_radius_rg - inner_radius_rg
        ) / (outer_torque_value - inner_torque_value)

        if inner_torque_value > 0.0 and outer_torque_value < 0.0:
            kind = STABLE_TRAP_KIND
        elif inner_torque_value < 0.0 and outer_torque_value > 0.0:
            kind = ANTI_TRAP_KIND
        else:
            continue

        candidate_roots_rg.append(root_rg)
        candidate_kinds.append(kind)
        local_width_rg.append(outer_radius_rg - inner_radius_rg)
        is_boundary.append(index == 0 or index == last_interval)

    if not candidate_roots_rg:
        return _empty_torque_traps()

    roots = np.asarray(candidate_roots_rg, dtype=float)
    kinds = np.asarray(candidate_kinds, dtype=int)
    widths = np.asarray(local_width_rg, dtype=float)
    boundary = np.asarray(is_boundary, dtype=bool)
    root_order = np.argsort(roots)
    roots, kinds, widths, boundary = (
        roots[root_order],
        kinds[root_order],
        widths[root_order],
        boundary[root_order],
    )
    keep = _resolved_crossing_mask(roots, kinds, widths, min_separation_cells)

    return TorqueTrapRadii(
        stable_radius_rg=roots[keep & (kinds == STABLE_TRAP_KIND)],
        anti_trap_radius_rg=roots[keep & (kinds == ANTI_TRAP_KIND)],
        boundary_radius_rg=roots[boundary],
        unresolved_radius_rg=roots[~keep],
    )


def stable_blocking_radius_cm(
    radius_cm: np.ndarray,
    torque_cgs: np.ndarray,
) -> float | None:
    """Return the outermost stable trap [cm] blocking inward migration.

    Used by the migration integrator to report *where* a trap-bound migrator is
    held.  A cell whose torque is exactly zero counts as a stable trap when its
    neighbours bracket it with a positive-inside / negative-outside pattern.
    """
    stable_roots_cm: list[float] = []
    for index in range(radius_cm.size - 1):
        inner_torque = torque_cgs[index]
        outer_torque = torque_cgs[index + 1]
        if inner_torque > 0.0 and outer_torque < 0.0:
            inner_radius = radius_cm[index]
            outer_radius = radius_cm[index + 1]
            root_cm = inner_radius - inner_torque * (outer_radius - inner_radius) / (
                outer_torque - inner_torque
            )
            stable_roots_cm.append(float(root_cm))

    zero_indices = np.where(torque_cgs == 0.0)[0]
    for index in zero_indices:
        if (
            index > 0
            and index + 1 < torque_cgs.size
            and torque_cgs[index - 1] > 0.0
            and torque_cgs[index + 1] < 0.0
        ):
            stable_roots_cm.append(float(radius_cm[index]))

    return max(stable_roots_cm) if stable_roots_cm else None


__all__ = [
    "ANTI_TRAP_KIND",
    "DEFAULT_MIN_TRAP_SEPARATION_CELLS",
    "STABLE_TRAP_KIND",
    "TorqueTrapRadii",
    "classify_torque_traps",
    "stable_blocking_radius_cm",
]
