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
purely from interpolation noise -- the opacity table alone steps by up to 2.4
decades between adjacent radial cells.  Such pairs are not physical traps.
They are rejected by ``min_basin_dex``, which measures the width of the
outward-torque region terminating at each trap in *decades of radius*.  That
test is resolution independent: refining the grid leaves a real trap's width in
dex unchanged while a table artefact keeps a fixed cell count and shrinks.  A
cell-count criterion was previously used for this and has been removed -- it
called the same feature resolved or unresolved depending only on how finely the
disk happened to be sampled, which is not a property of the disk.

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

#: Minimum basin width, in decades of radius, for a torque zero to count as a
#: resolved trap.  Unlike a cell count this is resolution independent: a real
#: trap keeps its width in dex as the grid is refined, while an opacity-table
#: artefact stays a fixed *number of cells* wide and so shrinks in dex.
#: A value of 0.02 dex rejects opacity-table single-cell wiggles while retaining
#: zeros with a radially resolved convergent basin. Resolution checks should be
#: repeated whenever the pAGN grid or opacity interpolation changes.
DEFAULT_MIN_TRAP_BASIN_DEX = 0.02
STABLE_TRAP_KIND = 1
ANTI_TRAP_KIND = -1


@dataclass(frozen=True)
class TorqueTrapRadii:
    """Torque-zero radii [r_g] separated by their stability interpretation."""

    stable_radius_rg: np.ndarray
    anti_trap_radius_rg: np.ndarray
    boundary_radius_rg: np.ndarray
    unresolved_radius_rg: np.ndarray
    #: Basin width [decades of radius] for each entry of ``stable_radius_rg``.
    basin_dex: np.ndarray

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
    return TorqueTrapRadii(
        empty, empty.copy(), empty.copy(), empty.copy(), empty.copy()
    )


def classify_torque_traps(
    radius_rg: np.ndarray,
    torque_values: np.ndarray,
    *,
    min_basin_dex: float = DEFAULT_MIN_TRAP_BASIN_DEX,
) -> TorqueTrapRadii:
    """Locate and classify the torque zeros of one migration torque profile.

    Parameters
    ----------
    radius_rg
        Radial grid [r_g]; need not be sorted.
    torque_values
        Signed total migration torque on the same grid.  Positive means
        outward.
    min_basin_dex
        A stable trap whose basin -- the width of the outward-torque region
        terminating at it -- is narrower than this many decades of radius is
        reported as ``unresolved_radius_rg`` instead of a trap.  Resolution
        independent: refining the grid widens a genuine trap in cells but
        leaves its width in dex unchanged, while a table artefact keeps a fixed
        cell width and shrinks in dex.

    Returns
    -------
    TorqueTrapRadii
        Stable traps, anti-traps, boundary crossings and unresolved wiggles,
        with ``basin_dex`` giving each stable trap's measured basin width.
    """
    radius_rg = np.asarray(radius_rg, dtype=float)
    values = np.asarray(torque_values, dtype=float)

    finite = np.isfinite(radius_rg) & np.isfinite(values) & (values != 0.0)
    radius_rg = radius_rg[finite]
    values = values[finite]

    if (
        values.size < 2
    ):
        return _empty_torque_traps()

    order = np.argsort(radius_rg)
    radius_rg = radius_rg[order]
    values = values[order]

    candidate_roots_rg: list[float] = []
    candidate_kinds: list[int] = []
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
        is_boundary.append(index == 0 or index == last_interval)

    if not candidate_roots_rg:
        return _empty_torque_traps()

    roots = np.asarray(candidate_roots_rg, dtype=float)
    kinds = np.asarray(candidate_kinds, dtype=int)
    boundary = np.asarray(is_boundary, dtype=bool)
    root_order = np.argsort(roots)
    roots, kinds, boundary = (
        roots[root_order],
        kinds[root_order],
        boundary[root_order],
    )
    keep = np.ones(roots.size, dtype=bool)

    # Width, in decades of radius, of the outward-torque region each root
    # bounds: inward from a stable trap (which sits at that region's outer
    # edge) and outward from an anti-trap (which sits at its inner edge). A
    # spurious pair bounds the same narrow spike, so both are rejected
    # together.
    #
    # This is the resolution-independent test. A physical trap is the outer
    # edge of a radially extended outward-torque region, so refining the grid
    # adds cells and leaves the width in dex unchanged. A discontinuity in the
    # opacity table instead produces a spike that stays a fixed *number of
    # cells* wide, so its width in dex shrinks under refinement and it is
    # rejected. Measuring the outer catchment instead would not discriminate:
    # a one-cell spike sitting in an otherwise inward disk has an arbitrarily
    # large outer catchment and would pass.
    region_dex = np.zeros(roots.size, dtype=float)
    for position in range(roots.size):
        if kinds[position] == STABLE_TRAP_KIND:
            bound_rg = roots[position - 1] if position > 0 else float(radius_rg[0])
            inner_rg, outer_rg = bound_rg, roots[position]
        else:
            bound_rg = (
                roots[position + 1]
                if position + 1 < roots.size
                else float(radius_rg[-1])
            )
            inner_rg, outer_rg = roots[position], bound_rg
        if inner_rg > 0.0 and outer_rg > inner_rg:
            region_dex[position] = float(np.log10(outer_rg / inner_rg))

    min_basin_dex = float(min_basin_dex)
    if np.isfinite(min_basin_dex) and min_basin_dex > 0.0:
        keep = keep & (region_dex >= min_basin_dex)

    stable = keep & (kinds == STABLE_TRAP_KIND)
    basin_dex = region_dex

    stable = keep & (kinds == STABLE_TRAP_KIND)
    return TorqueTrapRadii(
        stable_radius_rg=roots[stable],
        anti_trap_radius_rg=roots[keep & (kinds == ANTI_TRAP_KIND)],
        boundary_radius_rg=roots[boundary],
        unresolved_radius_rg=roots[~keep],
        basin_dex=basin_dex[stable],
    )


def inward_delivery_blocked(
    radius_rg: np.ndarray,
    torque: np.ndarray,
    inner_radius_rg: float | None = None,
    outer_radius_rg: float | None = None,
    min_basin_dex: float = DEFAULT_MIN_TRAP_BASIN_DEX,
) -> tuple[bool, float | None]:
    """Return ``(blocked, blocking_radius_rg)`` for one migration path.

    This is the single definition of "gas cannot deliver this migrator inward",
    shared by the trap classifier and the migration integrator so the two can
    never disagree.  Previously the integrator blocked on a bare
    ``any(torque >= 0)``, which promoted a single spurious positive cell -- the
    opacity table steps by up to 2.4 decades between adjacent cells -- into a
    permanent ``time = inf``, while the classifier filtered exactly those out.

    Traps are classified on the **full** profile and only then tested against
    ``[inner_radius_rg, outer_radius_rg]``.  Classifying on a truncated path
    would measure each trap's basin against the path's own end rather than the
    next real torque zero, which narrows it artificially and would reject
    genuine traps that happen to sit near the start radius.

    Two outcomes are distinguished, because they are different physics:

    * a **resolved stable trap** inside the path returns its radius;
    * **outward torque with no resolved trap** -- the migrator is being pushed
      out of the domain, not held -- returns ``None`` as the radius.  Delivery
      is still blocked, but the object is not trap-bound.
    """
    traps = classify_torque_traps(
        radius_rg,
        torque,
        min_basin_dex=min_basin_dex,
    )
    stable = np.asarray(traps.stable_radius_rg, dtype=float)
    if stable.size:
        on_path = np.ones(stable.shape, dtype=bool)
        if inner_radius_rg is not None:
            on_path &= stable >= float(inner_radius_rg)
        if outer_radius_rg is not None:
            on_path &= stable <= float(outer_radius_rg)
        if on_path.any():
            return True, float(np.max(stable[on_path]))

    values = np.asarray(torque, dtype=float)
    radii = np.asarray(radius_rg, dtype=float)
    within = np.ones(values.shape, dtype=bool)
    if inner_radius_rg is not None:
        within &= radii >= float(inner_radius_rg)
    if outer_radius_rg is not None:
        within &= radii <= float(outer_radius_rg)
    finite = np.isfinite(values) & within
    if finite.any() and np.any(values[finite] >= 0.0):
        return True, None
    return False, None


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
    "STABLE_TRAP_KIND",
    "TorqueTrapRadii",
    "classify_torque_traps",
    "stable_blocking_radius_cm",
]
