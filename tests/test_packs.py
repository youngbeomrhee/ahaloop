"""Tests for learning pack extraction, and above all for what it refuses."""

from __future__ import annotations

import json
from typing import Any

import pytest

from ahaloop.llm import LLMClient
from ahaloop.packs import (
    ConceptCandidate,
    build_prompt,
    chunk_segments,
    extract_pack,
    is_grounded,
    parse_candidates,
    verify_candidates,
)
from ahaloop.transcript import Transcript, TranscriptSegment

SEGMENTS = [
    TranscriptSegment(start=81.5, duration=3.7, text="so the model has to weigh every token"),
    TranscriptSegment(start=85.2, duration=5.6, text="and that weighting is self-attention"),
    TranscriptSegment(start=90.8, duration=3.3, text="which is quadratic in sequence length"),
    TranscriptSegment(start=94.1, duration=4.0, text="so long contexts get expensive fast"),
]


def candidate(**overrides) -> ConceptCandidate:
    fields = {
        "concept": "self-attention",
        "summary": "Each token is weighed against every other token.",
        "source_start": 85.2,
    }
    fields.update(overrides)
    return ConceptCandidate(**fields)


class FakeClient(LLMClient):
    """Returns canned replies in order, and records what it was asked."""

    name = "fake"

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.opts: list[dict[str, Any]] = []

    def complete(self, prompt: str, **opts: Any) -> str:
        self.prompts.append(prompt)
        self.opts.append(opts)
        return self.replies.pop(0)


class TestGrounding:
    def test_timestamp_from_the_transcript_is_grounded(self):
        assert is_grounded(candidate(source_start=90.8), SEGMENTS)

    def test_invented_timestamp_is_not_grounded(self):
        assert not is_grounded(candidate(source_start=87.3), SEGMENTS)

    def test_tolerance_covers_float_noise(self):
        assert is_grounded(candidate(source_start=85.20001), SEGMENTS)

    @pytest.mark.parametrize("start", [85.9, 87.3, 90.0])
    def test_tolerance_does_not_cover_proximity(self, start: float):
        """Being near a real segment is not being on one.

        87.3 is closer to 85.2 than to anything else, which is exactly the
        reasoning that would justify snapping it there. Snapping is the repair
        this module refuses to do, so nearness must not be enough.
        """
        assert not is_grounded(candidate(source_start=start), SEGMENTS)

    def test_float_round_trip_survives(self):
        restored = json.loads(json.dumps(85.2))

        assert is_grounded(candidate(source_start=restored), SEGMENTS)


class TestVerification:
    def test_grounded_candidate_becomes_a_cited_concept(self):
        report = verify_candidates([candidate()], SEGMENTS, "vq5WhoPCWQ8")

        assert report.discarded == []
        assert report.concepts[0].source_ref == "https://youtu.be/vq5WhoPCWQ8?t=85"

    def test_ungrounded_candidate_is_discarded_not_repaired(self):
        report = verify_candidates([candidate(source_start=87.3)], SEGMENTS, "vq5WhoPCWQ8")

        assert report.concepts == []
        assert report.discarded[0].reason == "ungrounded_timestamp"

    def test_discarding_one_does_not_discard_the_rest(self):
        report = verify_candidates(
            [candidate(), candidate(concept="quadratic cost", source_start=87.3)],
            SEGMENTS,
            "vq5WhoPCWQ8",
        )

        assert [c.concept for c in report.concepts] == ["self-attention"]
        assert len(report.discarded) == 1

    def test_ungrounded_rate_counts_both_sides(self):
        report = verify_candidates(
            [candidate(), candidate(source_start=87.3), candidate(source_start=1000.0)],
            SEGMENTS,
            "vq5WhoPCWQ8",
        )

        assert report.candidate_count == 3
        assert report.ungrounded_rate == pytest.approx(2 / 3)

    def test_no_candidates_is_not_a_failure(self):
        report = verify_candidates([], SEGMENTS, "vq5WhoPCWQ8")

        assert report.ungrounded_rate == 0.0


class TestParsing:
    def test_plain_json_array(self):
        reply = json.dumps([{"concept": "c", "summary": "s", "source_start": 85.2}])

        assert parse_candidates(reply)[0].source_start == 85.2

    def test_fenced_json_is_unwrapped(self):
        reply = '```json\n[{"concept": "c", "summary": "s", "source_start": 85.2}]\n```'

        assert len(parse_candidates(reply)) == 1

    def test_prose_yields_nothing(self):
        assert parse_candidates("Sure! Here are the concepts I found:") == []

    def test_object_instead_of_array_yields_nothing(self):
        assert parse_candidates('{"concept": "c"}') == []

    def test_malformed_element_is_dropped_without_losing_the_rest(self):
        reply = json.dumps(
            [
                {"concept": "kept", "summary": "s", "source_start": 85.2},
                {"concept": "no timestamp", "summary": "s"},
            ]
        )

        assert [c.concept for c in parse_candidates(reply)] == ["kept"]


class TestChunking:
    def test_chunks_are_consecutive_and_complete(self):
        chunks = list(chunk_segments(SEGMENTS, size=3))

        assert [len(c) for c in chunks] == [3, 1]
        assert [s for c in chunks for s in c] == SEGMENTS

    def test_prompt_lists_every_start_time(self):
        prompt = build_prompt(SEGMENTS)

        for segment in SEGMENTS:
            assert f"[{segment.start}]" in prompt


class TestExtractPack:
    def transcript(self) -> Transcript:
        return Transcript(
            video_id="vq5WhoPCWQ8",
            language_code="en",
            is_generated=True,
            segments=SEGMENTS,
        )

    def test_runs_every_chunk_at_zero_temperature(self):
        client = FakeClient(["[]"])

        extract_pack(client, self.transcript())

        assert client.opts[0]["temperature"] == 0.0

    def test_grounded_and_ungrounded_are_reported_together(self):
        client = FakeClient(
            [
                json.dumps(
                    [
                        {"concept": "self-attention", "summary": "s", "source_start": 85.2},
                        {"concept": "hallucinated", "summary": "s", "source_start": 87.3},
                    ]
                )
            ]
        )

        report = extract_pack(client, self.transcript())

        assert [c.concept for c in report.concepts] == ["self-attention"]
        assert report.ungrounded_rate == pytest.approx(0.5)

    def test_citation_borrowed_from_another_chunk_is_ungrounded(self):
        """Each chunk is checked against the segments the model was shown.

        A timestamp that is real elsewhere in the video was still not available
        to the model for this call, so it cannot have been read off the source.
        """
        segments = [
            TranscriptSegment(start=10.0, duration=2.0, text="first"),
            TranscriptSegment(start=20.0, duration=2.0, text="second"),
        ]
        transcript = Transcript(
            video_id="vq5WhoPCWQ8", language_code="en", is_generated=True, segments=segments
        )
        client = FakeClient(
            [
                json.dumps([{"concept": "a", "summary": "s", "source_start": 20.0}]),
                json.dumps([{"concept": "b", "summary": "s", "source_start": 10.0}]),
            ]
        )

        report = extract_pack(client, transcript, chunk_size=1)

        assert len(report.concepts) == 0
        assert len(report.discarded) == 2
