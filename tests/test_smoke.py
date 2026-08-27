"""Smoke tests for the Session 0 scaffold."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ahaloop.events import EventType, LearningEvent, Visibility, append_event, read_events
from ahaloop.llm import (
    CLI_COMMAND_ENV_VAR,
    PROVIDER_ENV_VAR,
    CLIClient,
    LLMConfigurationError,
    OllamaClient,
    get_client,
)
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

    def test_cli_is_selectable(self, monkeypatch):
        monkeypatch.setenv(PROVIDER_ENV_VAR, "cli")
        assert isinstance(get_client(), CLIClient)


class TestCLIClient:
    def run_stub(self, monkeypatch, **result):
        calls = {}

        def fake_run(command, **kwargs):
            calls["command"] = command
            calls["input"] = kwargs.get("input")
            if "raises" in result:
                raise result["raises"]
            return subprocess.CompletedProcess(
                command,
                result.get("returncode", 0),
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", ""),
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        return calls

    def test_prompt_goes_over_stdin(self, monkeypatch):
        monkeypatch.setenv(CLI_COMMAND_ENV_VAR, "claude -p")
        calls = self.run_stub(monkeypatch, stdout="[]")

        assert CLIClient().complete("extract concepts") == "[]"
        assert calls["input"] == "extract concepts"
        assert calls["command"] == ["claude", "-p"]

    @pytest.mark.parametrize(
        ("configured", "expected"),
        [
            ("claude -p", ["claude", "-p"]),
            ("codex exec", ["codex", "exec"]),
            ("ollama run llama3.2", ["ollama", "run", "llama3.2"]),
            ("llm -m 'gpt-4o mini'", ["llm", "-m", "gpt-4o mini"]),
        ],
    )
    def test_any_agent_cli_can_be_configured(self, monkeypatch, configured, expected):
        monkeypatch.setenv(CLI_COMMAND_ENV_VAR, configured)
        calls = self.run_stub(monkeypatch, stdout="[]")

        CLIClient().complete("p")

        assert calls["command"] == expected

    def test_unconfigured_command_names_the_env_var(self, monkeypatch):
        monkeypatch.delenv(CLI_COMMAND_ENV_VAR, raising=False)

        with pytest.raises(LLMConfigurationError, match=CLI_COMMAND_ENV_VAR):
            CLIClient().complete("p")

    def test_options_are_not_forwarded(self, monkeypatch):
        """Flags belong in the configured command, not in guessed arguments.

        `extract_pack` always sends `temperature=0.0`. Inventing a flag for it
        would break on most CLIs, so the option is dropped rather than guessed —
        a limitation the class docstring states outright.
        """
        monkeypatch.setenv(CLI_COMMAND_ENV_VAR, "claude -p")
        calls = self.run_stub(monkeypatch, stdout="[]")

        CLIClient().complete("p", temperature=0.0, model="whatever")

        assert calls["command"] == ["claude", "-p"]

    def test_missing_binary_names_the_command(self, monkeypatch):
        monkeypatch.setenv(CLI_COMMAND_ENV_VAR, "nope --run")
        self.run_stub(monkeypatch, raises=FileNotFoundError())

        with pytest.raises(LLMConfigurationError, match="nope"):
            CLIClient().complete("p")

    def test_failed_run_surfaces_stderr(self, monkeypatch):
        monkeypatch.setenv(CLI_COMMAND_ENV_VAR, "claude -p")
        self.run_stub(monkeypatch, returncode=1, stderr="not logged in")

        with pytest.raises(LLMConfigurationError, match="not logged in"):
            CLIClient().complete("p")

    def test_timeout_is_reported(self, monkeypatch):
        monkeypatch.setenv(CLI_COMMAND_ENV_VAR, "claude -p")
        self.run_stub(monkeypatch, raises=subprocess.TimeoutExpired("claude", 300))

        with pytest.raises(LLMConfigurationError, match="did not answer"):
            CLIClient().complete("p")


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
