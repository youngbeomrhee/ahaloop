"""Learning pack extraction — turning transcript segments into cited concepts.

This is the first place in ahaloop where a model is allowed to speak, and the
only place its output is treated as untrusted input.

The product promise is that every concept in a pack is one click from the moment
it was said. A model asked for that citation will sometimes return a timestamp
that appears nowhere in the transcript. That failure is silent: the response
parses, the types check, and the number looks reasonable. Nothing downstream can
tell it apart from a real citation.

The transcript makes it catchable anyway, because the segment list *is* the
complete set of legal timestamps. Membership in that set is a local, cheap,
model-free check — which is why citations are grounded on timestamps here rather
than on anything requiring outside knowledge to verify.

A candidate that fails the check is discarded, not repaired. Snapping it to a
nearby segment would treat an invented timestamp as a damaged field, when it is
better read as evidence about the whole item: a model that fabricated the
citation was not reading the transcript for that concept, so the concept it
shipped in the same breath has not earned trust either.

Discards are counted. The share of candidates that fail grounding is the quality
signal for everything upstream — prompt wording, chunk size, model choice — and
it cannot be recovered later if the failures are only dropped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

from pydantic import BaseModel, ValidationError

from ahaloop.llm import LLMClient
from ahaloop.transcript import Transcript, TranscriptSegment, source_ref

CHUNK_SIZE = 40
"""Segments per model call.

Smaller chunks shrink the set of timestamps the model is choosing among, which
both reduces fabrication and makes the grounding check sharper. Long contexts
buy nothing here: concepts are local to the passage that explains them.
"""

EXTRACTION_TEMPERATURE = 0.0
"""Extraction, not composition — the answer is already in the source.

Zero also keeps the discard rate comparable between runs, without which the
metric could not be used to judge a prompt change.
"""

_START_TOLERANCE = 1e-3
"""Slack for float round-tripping through JSON, and nothing else.

This is deliberately orders of magnitude below the spacing between real
segments. Widening it into a "close enough" match would reintroduce the repair
this module refuses to do.
"""

_PROMPT = """\
You are extracting concepts from a video transcript.

Each line below is one caption segment, prefixed by its start time in seconds:

{segments}

Return a JSON array. Each element must have exactly these keys:
  "concept": a short noun phrase naming one idea explained in the transcript
  "summary": one sentence explaining it, drawn only from the text above
  "source_start": the start time of the segment where the explanation begins

Rules:
- "source_start" MUST be copied verbatim from one of the start times listed
  above. Do not compute, average, or estimate a time.
- If you cannot point to a specific segment for a concept, omit that concept.
  A shorter list is correct; an invented citation is not.
- Return only the JSON array, with no commentary.
"""


class ConceptCandidate(BaseModel):
    """One concept as the model returned it, before grounding is checked."""

    concept: str
    summary: str
    source_start: float


class Concept(BaseModel):
    """A concept whose citation was found in the transcript."""

    concept: str
    summary: str
    source_ref: str


class DiscardedCandidate(BaseModel):
    """A candidate that failed grounding, kept only so it can be counted."""

    candidate: ConceptCandidate
    reason: str


class ExtractionReport(BaseModel):
    """Everything one extraction produced, including what it threw away."""

    concepts: list[Concept]
    discarded: list[DiscardedCandidate]

    @property
    def candidate_count(self) -> int:
        """How many concepts the model proposed, kept and discarded together."""
        return len(self.concepts) + len(self.discarded)

    @property
    def ungrounded_rate(self) -> float:
        """Share of proposed concepts whose citation was not in the transcript.

        Zero when the model proposed nothing, so that an empty extraction reads
        as "no claims made" rather than as a total failure.
        """
        if self.candidate_count == 0:
            return 0.0
        return len(self.discarded) / self.candidate_count


def chunk_segments(
    segments: list[TranscriptSegment], size: int = CHUNK_SIZE
) -> Iterator[list[TranscriptSegment]]:
    """Split segments into consecutive groups of at most `size`."""
    for start in range(0, len(segments), size):
        yield segments[start : start + size]


def build_prompt(segments: list[TranscriptSegment]) -> str:
    """Render the extraction prompt for one chunk.

    Start times are printed exactly as they will be compared, so that copying
    one verbatim is the easiest thing the model can do.
    """
    lines = "\n".join(f"[{segment.start}] {segment.text}" for segment in segments)
    return _PROMPT.format(segments=lines)


def parse_candidates(response: str) -> list[ConceptCandidate]:
    """Read the model's reply into candidates.

    Tolerates a fenced code block around the array, because that wrapping is
    presentation rather than content. Anything else that fails to parse yields
    no candidates: a reply that is not the requested shape is not evidence of
    anything, and guessing at its intent would be the fabrication this module
    exists to catch.
    """
    text = response.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    candidates = []
    for item in raw:
        try:
            candidates.append(ConceptCandidate.model_validate(item))
        except ValidationError:
            continue
    return candidates


def _cited_segment(
    candidate: ConceptCandidate, segments: list[TranscriptSegment]
) -> TranscriptSegment | None:
    """The segment the candidate cites, or None if it cites no real segment.

    This is a lookup, not a nearest-neighbour search: a candidate either names a
    segment that exists or it names nothing.
    """
    for segment in segments:
        if abs(candidate.source_start - segment.start) < _START_TOLERANCE:
            return segment
    return None


def is_grounded(candidate: ConceptCandidate, segments: list[TranscriptSegment]) -> bool:
    """Whether the candidate's citation names a segment that actually exists."""
    return _cited_segment(candidate, segments) is not None


def verify_candidates(
    candidates: list[ConceptCandidate],
    segments: list[TranscriptSegment],
    video_id: str,
) -> ExtractionReport:
    """Split candidates into grounded concepts and counted discards."""
    concepts: list[Concept] = []
    discarded: list[DiscardedCandidate] = []
    for candidate in candidates:
        segment = _cited_segment(candidate, segments)
        if segment is None:
            discarded.append(DiscardedCandidate(candidate=candidate, reason="ungrounded_timestamp"))
            continue
        concepts.append(
            Concept(
                concept=candidate.concept,
                summary=candidate.summary,
                source_ref=source_ref(video_id, segment),
            )
        )
    return ExtractionReport(concepts=concepts, discarded=discarded)


def extract_pack(
    client: LLMClient, transcript: Transcript, chunk_size: int = CHUNK_SIZE
) -> ExtractionReport:
    """Run the whole transcript through the model, chunk by chunk.

    Each chunk is verified against its own segments rather than the full
    transcript, so a citation borrowed from a passage the model was not shown
    still counts as ungrounded.

    `chunk_size` is a parameter because it is one of the few knobs that moves
    the ungrounded rate, and comparing settings is the point of measuring it.
    """
    concepts: list[Concept] = []
    discarded: list[DiscardedCandidate] = []
    for chunk in chunk_segments(transcript.segments, chunk_size):
        response = client.complete(build_prompt(chunk), temperature=EXTRACTION_TEMPERATURE)
        report = verify_candidates(parse_candidates(response), chunk, transcript.video_id)
        concepts.extend(report.concepts)
        discarded.extend(report.discarded)
    return ExtractionReport(concepts=concepts, discarded=discarded)
