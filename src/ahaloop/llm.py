"""Provider-neutral LLM adapter.

ahaloop is bring-your-own-LLM: inference runs on a model the user owns — their
API key, their subscription, or a local runtime. The project provides the loop,
never the inference bill.

That principle only survives if it is structural, so every model call in this
codebase goes through `LLMClient`. No module outside this one may import a
provider SDK, name a provider, or read a provider API key. Adding a provider
means adding a subclass here and nothing else.

Providers are selected at runtime via the `AHALOOP_LLM_PROVIDER` environment
variable; there is no default provider, because guessing one would quietly send
a user's text somewhere they did not choose.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

PROVIDER_ENV_VAR = "AHALOOP_LLM_PROVIDER"
OLLAMA_MODEL_ENV_VAR = "AHALOOP_OLLAMA_MODEL"
OLLAMA_HOST_ENV_VAR = "AHALOOP_OLLAMA_HOST"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
CLI_COMMAND_ENV_VAR = "AHALOOP_CLI_COMMAND"
CLI_TIMEOUT_SECONDS = 300


class LLMConfigurationError(RuntimeError):
    """Raised when no usable provider is configured."""


class LLMClient(ABC):
    """The single interface through which ahaloop talks to any model."""

    name: str

    @abstractmethod
    def complete(self, prompt: str, **opts: Any) -> str:
        """Return the model's completion for `prompt`.

        Args:
            prompt: The full prompt to send.
            **opts: Provider-specific options (model, temperature, max_tokens, ...).

        Returns:
            The completion text.
        """


class AnthropicClient(LLMClient):
    """Anthropic-backed client. Implemented in M1."""

    name = "anthropic"

    def complete(self, prompt: str, **opts: Any) -> str:
        raise NotImplementedError("AnthropicClient lands in M1")


class OpenAIClient(LLMClient):
    """OpenAI-backed client. Implemented in M1."""

    name = "openai"

    def complete(self, prompt: str, **opts: Any) -> str:
        raise NotImplementedError("OpenAIClient lands in M1")


class OllamaClient(LLMClient):
    """Local Ollama-backed client.

    Spoken to over plain HTTP rather than through the vendor SDK, because the
    generate endpoint is two fields wide and a dependency would buy nothing.

    The model is never guessed. Ollama serves whatever the user has pulled, and
    silently picking one would make a run's results depend on an invisible
    choice — which the extraction metric could not survive.
    """

    name = "ollama"

    def complete(self, prompt: str, **opts: Any) -> str:
        model = opts.pop("model", None) or os.environ.get(OLLAMA_MODEL_ENV_VAR)
        if not model:
            raise LLMConfigurationError(
                f"No Ollama model configured. Set {OLLAMA_MODEL_ENV_VAR} to a model "
                "you have pulled (see `ollama list`)."
            )
        host = os.environ.get(OLLAMA_HOST_ENV_VAR, DEFAULT_OLLAMA_HOST).rstrip("/")
        body = json.dumps(
            {"model": model, "prompt": prompt, "stream": False, "options": opts}
        ).encode()
        request = urllib.request.Request(
            f"{host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request) as response:
                payload = json.load(response)
        except urllib.error.URLError as exc:
            raise LLMConfigurationError(
                f"Could not reach Ollama at {host}: {exc}. Is `ollama serve` running?"
            ) from exc
        return payload["response"]


class CLIClient(LLMClient):
    """Runs inference through whatever agent CLI the user already has.

    Most people arriving here are paying for a coding agent rather than an API
    key — Claude Code, Codex, Gemini CLI, `llm`, `ollama run`. A subscription is
    one of the models a user owns, so this client treats any command that reads
    a prompt on stdin and prints a completion on stdout as a provider. That
    covers a whole class of tools without a subclass each.

    The command comes from `AHALOOP_CLI_COMMAND` and is never guessed. This
    client executes it, so a default would run a program the user did not
    choose — a sharper version of the reason there is no default provider.

    Options are deliberately not forwarded. Every CLI spells its flags
    differently and ahaloop cannot know which one it is talking to, so guessing
    a `--model` or `--temperature` flag would either error or, worse, be
    silently dropped. Put the flags in the command instead:

        AHALOOP_CLI_COMMAND='claude -p'
        AHALOOP_CLI_COMMAND='codex exec'
        AHALOOP_CLI_COMMAND='ollama run llama3.2'

    Known limitation: because nothing is forwarded, the `temperature=0` that
    extraction asks for is not honoured unless the command itself pins it.
    Ungrounded-rate comparisons across runs carry that sampling noise, and a
    command with no temperature control cannot be fully reproducible.
    """

    name = "cli"

    def complete(self, prompt: str, **opts: Any) -> str:
        configured = os.environ.get(CLI_COMMAND_ENV_VAR, "").strip()
        if not configured:
            raise LLMConfigurationError(
                f"No CLI command configured. Set {CLI_COMMAND_ENV_VAR} to a command that "
                "reads a prompt on stdin and prints the completion on stdout, "
                "for example 'claude -p' or 'ollama run llama3.2'."
            )
        command = shlex.split(configured)
        try:
            finished = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=CLI_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            raise LLMConfigurationError(
                f"{command[0]!r} is not on PATH (from {CLI_COMMAND_ENV_VAR})."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise LLMConfigurationError(
                f"{configured!r} did not answer within {CLI_TIMEOUT_SECONDS}s."
            ) from exc
        if finished.returncode != 0:
            raise LLMConfigurationError(
                f"{configured!r} exited {finished.returncode}: {finished.stderr.strip()}"
            )
        return finished.stdout


_PROVIDERS: dict[str, type[LLMClient]] = {
    AnthropicClient.name: AnthropicClient,
    OpenAIClient.name: OpenAIClient,
    OllamaClient.name: OllamaClient,
    CLIClient.name: CLIClient,
}


def get_client(provider: str | None = None) -> LLMClient:
    """Build the configured client.

    Args:
        provider: Provider name; falls back to the `AHALOOP_LLM_PROVIDER` env var.

    Raises:
        LLMConfigurationError: If no provider is set, or the name is unknown.
    """
    chosen = provider or os.environ.get(PROVIDER_ENV_VAR)
    known = ", ".join(sorted(_PROVIDERS))
    if not chosen:
        raise LLMConfigurationError(
            f"No LLM provider configured. Set {PROVIDER_ENV_VAR} to one of: {known}."
        )
    try:
        return _PROVIDERS[chosen]()
    except KeyError:
        raise LLMConfigurationError(
            f"Unknown LLM provider {chosen!r}. Expected one of: {known}."
        ) from None
