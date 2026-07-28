"""Small command-line summary for IMRI event outputs."""

from pathlib import Path

import pandas as pd

DEFAULT_EVENTS_CSV = Path("imri_dyn_results/imri_events.csv")


def main(events_csv: Path = DEFAULT_EVENTS_CSV) -> None:
    """Print compact mass statistics for counted IMRI events."""
    frame = pd.read_csv(events_csv)
    if frame.empty or "counted_as_imri" not in frame:
        print("No IMRI event rows found")
        return
    selected = frame[frame["counted_as_imri"].astype(bool)]
    if selected.empty:
        print("No counted IMRI events found")
        return
    mass = selected["mass_msun"]
    print(f"IMRI count = {len(selected)}")
    print(
        f"max mass = {mass.max():.3f} Msun\n"
        f"min mass = {mass.min():.3f} Msun\n"
        f"mean mass = {mass.mean():.3f} Msun"
    )


if __name__ == "__main__":
    main()
