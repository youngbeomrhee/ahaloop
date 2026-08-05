"""Structured learning-event log.

Every signal the loop produces — quiz results, gate outcomes, gaps opening and
closing, aha moments — is written here as a schema'd, append-only record.

Two properties are deliberate and load-bearing:

1. The log is the training data for later personalization (recommendation,
   per-user forgetting curves). Such data cannot be reconstructed after the
   fact, so it is captured from the first session onward.
2. Every event carries a `visibility` flag, defaulting to private. A learning
   graph can only be shared publicly if the choice was recorded per event from
   the start; retrofitting it would mean guessing at intent.

`occurred_at` is injected by the caller rather than defaulted to "now" so that
backfills, imports, and tests record the time the learning actually happened.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """The learning signals the loop emits."""

    QUIZ_RESULT = "quiz_result"
    GATE_PASS = "gate_pass"
    GATE_FAIL = "gate_fail"
    GAP_OPENED = "gap_opened"
    GAP_RESOLVED = "gap_resolved"
    AHA = "aha"


class Visibility(StrEnum):
    """Who may see an event. Private unless the learner opts in."""

    PRIVATE = "private"
    PUBLIC = "public"


class LearningEvent(BaseModel):
    """One thing that happened while learning."""

    event_type: EventType
    concept: str
    source_ref: str
    occurred_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    visibility: Visibility = Visibility.PRIVATE


def append_event(event: LearningEvent, path: Path) -> None:
    """Append one event to a JSONL log, creating the file and parents if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n")


def read_events(path: Path) -> list[LearningEvent]:
    """Read every event from a JSONL log. Returns empty if the log does not exist."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [LearningEvent.model_validate_json(line) for line in f if line.strip()]
