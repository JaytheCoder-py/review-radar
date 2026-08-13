"""What an index calculator must do about an event.

A pure function of the event type and the confidence. No I/O, no model, no state - so it
can be read and argued with by someone who does not run it.

The confidence override is the important line: below threshold, every treatment collapses
to `MANUAL_REVIEW` regardless of how confident the *type* looked. Acting on a
low-confidence split is worse than queueing it, because the divisor adjustment is applied
before anyone notices and unwinding it is a recalculation event.
"""

from __future__ import annotations

from typing import Final

from reviewradar.types import EventType, Treatment

#: The treatment each event type implies, at sufficient confidence.
#:
#: SPINOFF is DIVISOR_ADJUST rather than SHARES_UPDATE because the parent's price steps
#: down on the distribution: the index has to absorb the value that left, and only a
#: divisor change does that without a phantom return.
TREATMENT_BY_TYPE: Final[dict[EventType, Treatment]] = {
    EventType.SPLIT_FORWARD: Treatment.DIVISOR_ADJUST,
    EventType.SPLIT_REVERSE: Treatment.DIVISOR_ADJUST,
    EventType.SPECIAL_DIVIDEND: Treatment.DIVISOR_ADJUST,
    EventType.SPINOFF: Treatment.DIVISOR_ADJUST,
    EventType.RIGHTS_ISSUE: Treatment.DIVISOR_ADJUST,
    EventType.MERGER_COMPLETED: Treatment.REMOVE_CONSTITUENT,
    EventType.DELISTING: Treatment.REMOVE_CONSTITUENT,
    EventType.BANKRUPTCY: Treatment.REMOVE_CONSTITUENT,
    EventType.TICKER_CHANGE: Treatment.SHARES_UPDATE,
    EventType.NAME_CHANGE: Treatment.SHARES_UPDATE,
    EventType.NO_INDEX_ACTION: Treatment.NO_ACTION,
    EventType.UNRESOLVED: Treatment.MANUAL_REVIEW,
}

#: Default routing threshold. Calibrated against the random stratum of the gold set
#: rather than chosen in advance - see `evals.score.calibrate_threshold` and D-007.
DEFAULT_THRESHOLD: Final[float] = 0.70


def decide(
    event_type: EventType, *, confidence: float, threshold: float = DEFAULT_THRESHOLD
) -> Treatment:
    """Map an event type to a treatment, with a confidence floor.

    `NO_INDEX_ACTION` is exempt from the floor. A low-confidence "nothing happened" is
    still a filing nobody needs to look at, and routing those to review would drown the
    queue in exactly the filings the baseline exists to eliminate.
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1]; got {confidence}")
    treatment = TREATMENT_BY_TYPE[event_type]
    if treatment is Treatment.NO_ACTION:
        return treatment
    if confidence < threshold:
        return Treatment.MANUAL_REVIEW
    return treatment
