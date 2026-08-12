"""Terminal reporting for the IMRI workflow."""

from .models import FloatArray, IMRIDynSingleResult
from .runtime import vprint


def vprint_population_ranges(
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
        f"{label} mass range is "
        f"{mass_msun.min():.3e} to {mass_msun.max():.3e} Msun"
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


def print_header(text: str, width: int = 72, char: str = "=") -> None:
    """Print a centered workflow heading."""
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}")


def debug_print_result(result: IMRIDynSingleResult) -> None:
    """Print detailed counters for one realization."""
    print("\n========== DEBUG INFO ==========")
    print(f"Sampled objects: {result.total_sampled_objects}")
    print(f"Disk crossing: {result.disk_crossing_objects}")
    print(f"Embedded stars: {result.embedded_star_count}")
    print(f"Embedded BHs: {result.embedded_bh_count}")
    print(f"Migration-formed BHs: {result.migration_formed_bh_count}")
    print(f"BH candidates: {result.bh_candidate_count}")
    print(f"IMRI candidates: {result.imri_count}")
    print(f"Detectable IMRIs: {result.detectable_imri_count}")
    print(f"IMRI candidate mass: {result.imri_mass_msun:.3e} Msun")
    print(f"Detectable IMRI mass: {result.detectable_imri_mass_msun:.3e} Msun")
    print("================================\n")


def print_result_summary(result: IMRIDynSingleResult) -> None:
    """Print the normal compact summary for one realization."""
    print(f"  total sampled objects : {result.total_sampled_objects:.3e}")
    print(f"  disk-crossing objects : {result.disk_crossing_objects:.3e}")
    print(f"  f_geo                 : {result.f_geo:.3e}")
    print(f"  embedded stars        : {result.embedded_star_count:.3e}")
    print(f"  embedded BHs          : {result.embedded_bh_count:.3e}")
    print(f"  migration-formed BHs  : {result.migration_formed_bh_count:.3e}")
    print(f"  BH candidates         : {result.bh_candidate_count:.3e}")
    print(f"  IMRI candidates       : {result.imri_count:.3e}")
    print(f"  no-IMRI objects       : {result.no_imri_count:.3e}")
    print(f"  detectable IMRIs      : {result.detectable_imri_count:.3e}")
    print(f"  non-detectable IMRIs  : {result.no_detectable_imri_count:.3e}")
    print(
        "  Torque migration      : "
        f"evaluated={result.torque_migration_evaluated_count}, "
        f"finite={result.finite_torque_migration_count}, "
        f"trap_bound={result.trap_bound_migration_count}, "
        f"errors={result.torque_migration_error_count}"
    )
    print(f"  Trap-bound mass       : {result.trap_bound_mass_msun:.6e} Msun")
    print(f"  IMRI candidate rate   : {result.imri_rate_per_myr:.3e} /Myr")
    print(f"  detectable IMRI rate  : {result.detectable_imri_rate_per_myr:.3e} /Myr")
    print(
        f"  candidate mass rate   : "
        f"{result.imri_mass_rate_msun_per_myr:.3e} Msun/Myr"
    )
    print(
        f"  detectable mass rate  : "
        f"{result.detectable_imri_mass_rate_msun_per_myr:.3e} Msun/Myr"
    )
