"""Event-rate classification for gas-assisted IMRI formation."""

from dataclasses import dataclass

import numpy as np

BH_ORIGIN_EMBEDDED = "ebh"
BH_ORIGIN_MIGRATION_FORMED = "mfbh"
BH_ORIGIN_NOT_FORMED = "no_bh"

EVENT_RATE_IMRI_CANDIDATE = "imri candi"
EVENT_RATE_NO_IMRI = "no-IMRI"


@dataclass(frozen=True)
class EventRateDecision:
    """Classification result for the event-rate stage."""

    bh_origin: str
    bh_candidate: bool
    is_imri_candidate: bool
    bh_embedding_time_myr: float
    arrival_time_myr: float
    event_rate_class: str
    event_rate_reason: str


def classify_event_rate(
    *,
    object_type: int,
    formation_time_myr: float,
    damping_time_myr: float,
    final_time_myr: float,
    migration_time_myr: float,
    duty_cycle_myr: float,
) -> EventRateDecision:
    """Classify whether gas migration delivers a BH to the IMRI radius in time."""
    if not np.isfinite(migration_time_myr):
        return EventRateDecision(
            bh_origin=BH_ORIGIN_EMBEDDED if object_type == 1 else BH_ORIGIN_NOT_FORMED,
            bh_candidate=object_type == 1,
            is_imri_candidate=False,
            bh_embedding_time_myr=final_time_myr if object_type == 1 else np.nan,
            arrival_time_myr=np.inf,
            event_rate_class=EVENT_RATE_NO_IMRI,
            event_rate_reason="migration_unavailable",
        )

    arrival_time_myr = final_time_myr + migration_time_myr
    embedded_bh = object_type == 1
    migration_formed_bh = bool(
        object_type == 0
        and np.isfinite(formation_time_myr)
        and np.isfinite(damping_time_myr)
        and formation_time_myr < damping_time_myr + migration_time_myr
    )
    bh_candidate = bool(embedded_bh or migration_formed_bh)

    if embedded_bh:
        bh_origin = BH_ORIGIN_EMBEDDED
        bh_embedding_time_myr = final_time_myr
    elif migration_formed_bh:
        bh_origin = BH_ORIGIN_MIGRATION_FORMED
        bh_embedding_time_myr = formation_time_myr
    else:
        bh_origin = BH_ORIGIN_NOT_FORMED
        bh_embedding_time_myr = np.nan

    if not bh_candidate:
        return EventRateDecision(
            bh_origin=bh_origin,
            bh_candidate=False,
            is_imri_candidate=False,
            bh_embedding_time_myr=bh_embedding_time_myr,
            arrival_time_myr=arrival_time_myr,
            event_rate_class=EVENT_RATE_NO_IMRI,
            event_rate_reason="not_bh_before_r_imri",
        )

    if arrival_time_myr > duty_cycle_myr:
        return EventRateDecision(
            bh_origin=bh_origin,
            bh_candidate=True,
            is_imri_candidate=False,
            bh_embedding_time_myr=bh_embedding_time_myr,
            arrival_time_myr=arrival_time_myr,
            event_rate_class=EVENT_RATE_NO_IMRI,
            event_rate_reason="misses_duty_cycle",
        )

    return EventRateDecision(
        bh_origin=bh_origin,
        bh_candidate=True,
        is_imri_candidate=True,
        bh_embedding_time_myr=bh_embedding_time_myr,
        arrival_time_myr=arrival_time_myr,
        event_rate_class=EVENT_RATE_IMRI_CANDIDATE,
        event_rate_reason=EVENT_RATE_IMRI_CANDIDATE,
    )
