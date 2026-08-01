"""Small command-line summary for IMRI event outputs."""

from pathlib import Path

import pandas as pd

DEFAULT_EVENTS_CSV = Path("imri_dyn_results/imri_events.csv")


def main(events_csv: Path = DEFAULT_EVENTS_CSV) -> None:
    """Print compact mass statistics for IMRI candidates and detections."""
    frame = pd.read_csv(events_csv)
    if frame.empty or "is_imri_candidate" not in frame:
        print("No IMRI event rows found")
        return

    candidates = frame[frame["is_imri_candidate"].astype(bool)]
    if candidates.empty:
        print("No IMRI candidates found")
        return

    detected_count = 0
    if "is_lisa_detectable" in frame:
        detected_count = int(frame["is_lisa_detectable"].astype(bool).sum())

    mass = candidates["mass_msun"]
    print(f"IMRI candidate count = {len(candidates)}")
    print(f"Detectable IMRI count = {detected_count}")
    print(
        f"max mass = {mass.max():.3f} Msun\n"
        f"min mass = {mass.min():.3f} Msun\n"
        f"mean mass = {mass.mean():.3f} Msun"
    )


if __name__ == "__main__":
    main()
