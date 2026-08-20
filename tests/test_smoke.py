"""Smoke tests for the Session 0 scaffold."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ahaloop.events import EventType, LearningEvent, Visibility, append_event, read_events
from ahaloop.llm import PROVIDER_ENV_VAR, LLMConfigurationError, OllamaClient, get_client
from ahaloop.transcript import TranscriptError, TranscriptSegment, parse_video_id, source_ref


def make_event(**overrides) -> LearningEvent:
    fields = {
        "event_type": EventType.AHA,
        "concept": "residual connection",
        "source_ref": "https://youtu.be/vq5WhoPCWQ8?t=612",
        "occurred_at": datetime(2026, 8, 6, 9, 30, tzinfo=UTC),
        "payload": {"note": "gradients skip the block entirely"},
    }
    fields.update(overrides)
    return LearningEvent(**fields)


class TestEvents:
    def test_defaults_to_private(self):
        assert make_event().visibility is Visibility.PRIVATE

    def test_jsonl_roundtrip(self, tmp_path: Path):
        log = tmp_path / "nested" / "events.jsonl"
        written = [
            make_event(),
            make_event(
                event_type=EventType.QUIZ_RESULT,
                payload={"correct": False},
                visibility=Visibility.PUBLIC,
            ),
        ]
        for event in written:
            append_event(event, log)

        assert read_events(log) == written

    def test_append_only(self, tmp_path: Path):
        log = tmp_path / "events.jsonl"
        append_event(make_event(), log)
        append_event(make_event(event_type=EventType.GAP_RESOLVED), log)

        assert len(log.read_text(encoding="utf-8").splitlines()) == 2

    def test_missing_log_reads_empty(self, tmp_path: Path):
        assert read_events(tmp_path / "absent.jsonl") == []

    def test_occurred_at_is_required(self):
        with pytest.raises(ValueError):
            LearningEvent(event_type=EventType.AHA, concept="c", source_ref="s")


class TestLLM:
    def test_unconfigured_provider_names_the_env_var(self, monkeypatch):
        monkeypatch.delenv(PROVIDER_ENV_VAR, raising=False)
        with pytest.raises(LLMConfigurationError, match=PROVIDER_ENV_VAR):
            get_client()

    def test_unknown_provider_lists_known_ones(self, monkeypatch):
        monkeypatch.setenv(PROVIDER_ENV_VAR, "definitely-not-a-provider")
        with pytest.raises(LLMConfigurationError, match="ollama"):
            get_client()

    def test_env_var_selects_provider(self, monkeypatch):
        monkeypatch.setenv(PROVIDER_ENV_VAR, "ollama")
        assert isinstance(get_client(), OllamaClient)


class TestTranscript:
    @pytest.mark.parametrize(
        "url",
        [
            "https://youtu.be/vq5WhoPCWQ8",
            "https://www.youtube.com/watch?v=vq5WhoPCWQ8",
            "https://www.youtube.com/watch?v=vq5WhoPCWQ8&t=612s",
        ],
    )
    def test_parses_video_id_from_url_shapes(self, url: str):
        assert parse_video_id(url) == "vq5WhoPCWQ8"

    def test_url_without_video_id_is_rejected(self):
        with pytest.raises(TranscriptError):
            parse_video_id("https://www.youtube.com/results?search_query=openai")

    def test_source_ref_truncates_backwards(self):
        segment = TranscriptSegment(start=612.94, duration=4.48, text="...")

        assert source_ref("vq5WhoPCWQ8", segment) == "https://youtu.be/vq5WhoPCWQ8?t=612"


def test_cli_exposes_ingest():
    from ahaloop.cli import app, ingest

    assert app is not None
    assert callable(ingest)
