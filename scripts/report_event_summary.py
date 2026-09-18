"""Generate human-readable summaries and figures for one IMRI workflow run."""

import argparse
import sys
from pathlib import Path
from typing import Sequence, cast

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (
    SOURCE_ROOT / "physical_process",
    SOURCE_ROOT / "disk_quantity",
    SOURCE_ROOT,
    PROJECT_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from workflows.config import DISK_MODELS, IMRI_EVENTS_FILENAME
from viz.rate_plots import (
    build_rate_summary_data,
    generate_run_figures,
    generate_scan_figures,
    load_run_results,
    shared_rate_axis_limits,
)

DEFAULT_RESULTS_DIR = Path("output/dyn_results")


def _print_event_summary(events_csv: Path) -> None:
    """Print compact mass statistics for IMRI candidates and detections."""
    frame = pd.read_csv(events_csv)
    if frame.empty or "is_imri_candidate" not in frame:
        print("No IMRI event rows found")
        return

    unsupported_count = 0
    if "band_entry_status" in frame:
        unsupported_count = int(
            frame["band_entry_status"].eq("band_entry_inside_isco").sum()
        )

    candidates = frame[frame["is_imri_candidate"].eq(True)]
    if candidates.empty:
        print("No classified IMRI candidates found")
        if unsupported_count:
            print(f"Unsupported circular band-entry rows = {unsupported_count}")
        return

    detected_count = 0
    if "is_lisa_detectable" in frame:
        detected_count = int(frame["is_lisa_detectable"].eq(True).sum())

    mass = candidates["mass_msun"]
    print(f"IMRI candidate count = {len(candidates)}")
    print(f"Detectable IMRI count = {detected_count}")
    if unsupported_count:
        print(f"Unsupported circular band-entry rows = {unsupported_count}")
    print(
        f"max mass = {mass.max():.3f} Msun\n"
        f"min mass = {mass.min():.3f} Msun\n"
        f"mean mass = {mass.mean():.3f} Msun"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate persisted IMRI results and generate rate figures."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="workflow directory containing the persisted CSV and metadata files",
    )
    parser.add_argument(
        "--disk-model",
        choices=DISK_MODELS,
        default=None,
        help=(
            "required when RESULTS_DIR is a paired disk-model scan root; "
            "never inferred from directory order"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="figure directory; defaults to RESULTS_DIR/figures",
    )
    parser.add_argument(
        "--nsc-mass-msun",
        type=float,
        default=None,
        help="NSC mass to present when a result contains more than one NSC mass",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Print the existing summary, validate tables, and create rate figures."""
    arguments = build_parser().parse_args(argv)
    direct_events = arguments.results_dir / IMRI_EVENTS_FILENAME
    scan_summary = arguments.results_dir / "disk_model_scan_summary.csv"
    if direct_events.is_file():
        run = load_run_results(arguments.results_dir)
        actual_model = str(run.repeats["disk_model"].iloc[0])
        if arguments.disk_model is not None and arguments.disk_model != actual_model:
            raise ValueError(
                f"requested disk model {arguments.disk_model} does not match "
                f"scalar run model {actual_model}"
            )
        _print_event_summary(direct_events)
        outputs = generate_run_figures(
            arguments.results_dir,
            arguments.output_dir,
            arguments.nsc_mass_msun,
        )
    elif scan_summary.is_file():
        if arguments.disk_model is None:
            raise ValueError(
                "paired scan root requires an explicit --disk-model; "
                f"choose one of {DISK_MODELS}"
            )
        repeats = pd.read_csv(scan_summary)
        model_repeats = cast(
            pd.DataFrame,
            repeats.loc[
                repeats["disk_model"].astype(str).eq(arguments.disk_model)
            ].copy(),
        )
        if model_repeats.empty:
            raise ValueError(
                f"paired scan has no rows for disk model {arguments.disk_model}"
            )
        run_dirs = sorted(
            path
            for path in (arguments.results_dir / arguments.disk_model).iterdir()
            if path.is_dir() and (path / IMRI_EVENTS_FILENAME).is_file()
        )
        if not run_dirs:
            raise FileNotFoundError(
                f"paired scan has no scalar runs for {arguments.disk_model}"
            )
        reference_run = load_run_results(run_dirs[0])
        _print_event_summary(run_dirs[0] / IMRI_EVENTS_FILENAME)
        figure_dir = arguments.output_dir or (
            arguments.results_dir / "figures" / arguments.disk_model
        )
        outputs = generate_scan_figures(
            model_repeats,
            figure_dir,
            reference_run,
            arguments.nsc_mass_msun,
            shared_rate_axis_limits(build_rate_summary_data(repeats)),
        )
    else:
        raise FileNotFoundError(
            "results directory is neither a scalar run nor a paired disk-model scan"
        )
    for name, paths in outputs.items():
        if isinstance(paths, list):
            for path in paths:
                print(f"saved {name}: {path}")
        else:
            print(f"saved {name}: {paths}")


if __name__ == "__main__":
    main()
