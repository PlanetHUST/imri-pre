"""Runtime logging controls shared by workflow modules."""

from typing import Any

VERBOSE = True


def set_verbose(enabled: bool) -> None:
    """Enable or disable workflow progress messages."""
    global VERBOSE
    VERBOSE = enabled


def vprint(*args: Any, **kwargs: Any) -> None:
    """Print only when workflow verbosity is enabled."""
    if VERBOSE:
        print(*args, **kwargs)
