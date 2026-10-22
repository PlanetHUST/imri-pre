"""Correctness tests for the Round-1 merger-tree interface and job system.

These lock the Round-1 guarantees that Round-2 must not regress: bounded
minimal-column reading, chunk-size invariance, deterministic identities and
seeds, explicit unit conversion, and the separation of execution failure from
physical zero.
"""

from __future__ import annotations

import numpy as np

from physical_process.sampling import SOLAR_METALLICITY
from lisa.snr import DEFAULT_SNR_WINDOW_PLACEMENT
from workflows.config import IMRIDynConfig
from workflows.host_mapping import (
    HostMappingError,
    HostMappingPolicy,
    resolve_host_parameters,
)
from workflows.merger_tree import (
    REQUIRED_TREE_COLUMNS,
    HaloHost,
    HostSelection,
    MergerTreeReader,
    make_host_id,
)
from workflows.single_halo import JOB_STATUSES, STATUS_SUCCESS
from workflows.tree_batch import build_job, derive_host_seed

FIXTURE_HEADER = (
    ",plc,z,m,inode,br,br_b4,fc,sf,bh,brf,Mhot,Mcold,Mstar,Mout,mZ,Z,dM,Q,mlist,dt,"
    "gen,MSF,MBH,MBH2,Mgas,Mgas2,Mtde,Mtde2,Macc2"
)


def _fixture(tmp_path, rows: int = 12) -> "object":
    lines = [FIXTURE_HEADER]
    for index in range(rows):
        lines.append(
            ",".join(
                [
                    str(index + 1), "-1", f"{6.0 + 0.01 * index}", "1e12",
                    str(1 + index % 3), "1", "1", "1", "0", "-1", "-1",
                    "junk", "junk", f"{3.0e9 - 1e7 * index}", "junk", "junk",
                    "2.0e-4", "9", "0", "336631", "747", "junk", "6740",
                    "0.0", "0", "0", "0", "0", "0", "",
                ]
            )
        )
    path = tmp_path / "tree.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def _base_config(tmp_path) -> IMRIDynConfig:
    return IMRIDynConfig(
        smbh_masses_msun=(1.0e5,), nsc_masses_msun=(2.0e5,), num_repeats=1,
        duty_cycle_myr=10.0, output_dir=tmp_path / "out", metallicity=1.0e-5,
        initial_stellar_density_gamma=1.75, use_disk_inner_radius=True,
        lisa_min_gw_frequency_hz=1e-3, lisa_max_gw_frequency_hz=1e-1,
        max_gw_time_myr=13.6e3, gw_eccentricity=0.0, redshift=15.0,
        snr_threshold=20.0,
        lisa_mission_duration_years=4.0,
        snr_window_placement=DEFAULT_SNR_WINDOW_PLACEMENT,
        detector_hubble_constant_km_s_mpc=67.4, detector_omega_m=0.315,
        cosmic_merge_time_myr=np.inf, cosmic_merge_smbh_mass_factor=2.0,
        cosmic_merge_radius_factor=2.0, damping_cpu_count=1,
        checkpoint_grid_interval=5, print_debug_results=False, base_seed=0,
    )


def test_reader_loads_only_the_required_columns() -> None:
    assert set(REQUIRED_TREE_COLUMNS) == {"z", "m", "Mstar", "Z", "MBH", "inode", "br"}


def test_unused_columns_are_never_parsed(tmp_path_factory=None) -> None:
    """Non-numeric junk in ignored columns must not break the reader."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = _fixture(Path(directory))
        hosts = list(MergerTreeReader(path).iter_hosts())
        assert len(hosts) == 12
        assert all(host.stellar_mass_msun is not None for host in hosts)


def test_chunk_size_does_not_change_the_selected_hosts() -> None:
    """Chunk size is an engineering parameter, never a physical one."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = _fixture(Path(directory))
        selection = HostSelection(nodes=(1.0, 2.0))
        reference = [
            (h.host_id, h.redshift, h.stellar_mass_msun)
            for h in MergerTreeReader(path, selection, chunk_size=1).iter_hosts()
        ]
        for chunk_size in (2, 3, 5, 7, 1000):
            other = [
                (h.host_id, h.redshift, h.stellar_mass_msun)
                for h in MergerTreeReader(path, selection, chunk_size).iter_hosts()
            ]
            assert other == reference, f"chunk_size={chunk_size} changed the host set"


def test_host_identity_is_deterministic_and_unique() -> None:
    first = make_host_id(3, 1, 42)
    second = make_host_id(3, 1, 42)
    assert first == second == "n000003_b000001_r0000042"
    assert make_host_id(3, 1, 43) != first


def test_host_seed_is_deterministic_and_identity_derived() -> None:
    host = HaloHost(host_id="h", redshift=6.0, record_index=7, node_id=3, branch_id=1)
    assert derive_host_seed(2026, host) == derive_host_seed(2026, host)
    assert derive_host_seed(2026, host) != derive_host_seed(2027, host)
    other = HaloHost(host_id="h", redshift=6.0, record_index=8, node_id=3, branch_id=1)
    assert derive_host_seed(2026, host) != derive_host_seed(2026, other)


def test_job_output_namespaces_are_unique() -> None:
    hosts = [
        HaloHost(host_id=make_host_id(1, 1, i), redshift=6.0, record_index=i,
                 node_id=1, branch_id=1)
        for i in range(8)
    ]
    jobs = [build_job(host, 0, "/tmp/sim") for host in hosts]
    assert len({job.job_id for job in jobs}) == len(jobs)
    assert len({job.output_dir for job in jobs}) == len(jobs)
    assert len({job.seed for job in jobs}) == len(jobs)


def test_tree_metallicity_conversion_is_explicit() -> None:
    """Tree Z is an absolute mass fraction; the local model wants solar units."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        config = _base_config(Path(directory))
        host = HaloHost(host_id="h", redshift=6.0, record_index=0,
                        stellar_mass_msun=3.0e9, metallicity_mass_fraction=2.0e-4)
        policy = HostMappingPolicy(smbh_mass_source="stellar-scaling",
                                   nsc_mass_source="stellar-scaling",
                                   metallicity_source="tree")
        resolved = resolve_host_parameters(host, config, policy)
        assert np.isclose(resolved.metallicity_zsun, 2.0e-4 / SOLAR_METALLICITY)
        assert resolved.sources["metallicity_zsun"] == "tree:Z/Zsun"


def test_empty_tree_bh_mass_is_reported_not_substituted() -> None:
    """The supplied tree's empty MBH column must fail loudly under the default."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        config = _base_config(Path(directory))
        host = HaloHost(host_id="h", redshift=6.0, record_index=0,
                        smbh_mass_msun=0.0, stellar_mass_msun=3.0e9)
        try:
            resolve_host_parameters(host, config, HostMappingPolicy())
        except HostMappingError as error:
            assert error.status == "invalid_host_input"
            assert "MBH" in error.message
            return
        raise AssertionError("a zero tree MBH was silently accepted")


def test_derived_masses_are_labelled_as_derived() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        config = _base_config(Path(directory))
        host = HaloHost(host_id="h", redshift=6.0, record_index=0,
                        stellar_mass_msun=3.0e9)
        policy = HostMappingPolicy(smbh_mass_source="stellar-scaling",
                                   nsc_mass_source="stellar-scaling")
        resolved = resolve_host_parameters(host, config, policy)
        assert resolved.sources["smbh_mass_msun"].startswith("derived:")
        assert resolved.sources["nsc_mass_msun"].startswith("derived:")
        assert resolved.sources["redshift"] == "tree:z"
        assert np.isclose(resolved.smbh_mass_msun,
                          policy.smbh_stellar_mass_fraction * 3.0e9)


def test_out_of_domain_host_is_not_a_physical_zero() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        config = _base_config(Path(directory))
        host = HaloHost(host_id="h", redshift=6.0, record_index=0,
                        stellar_mass_msun=1.0e14)  # -> M_smbh far above the domain
        policy = HostMappingPolicy(smbh_mass_source="stellar-scaling")
        try:
            resolve_host_parameters(host, config, policy)
        except HostMappingError as error:
            assert error.status == "outside_local_model_domain"
            assert error.status in JOB_STATUSES
            assert error.status != STATUS_SUCCESS
            return
        raise AssertionError("an out-of-domain host was accepted")
