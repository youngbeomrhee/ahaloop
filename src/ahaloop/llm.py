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

import os
from abc import ABC, abstractmethod
from typing import Any

PROVIDER_ENV_VAR = "AHALOOP_LLM_PROVIDER"


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
    """Local Ollama-backed client. Implemented in M1."""

    name = "ollama"

    def complete(self, prompt: str, **opts: Any) -> str:
        raise NotImplementedError("OllamaClient lands in M1")


_PROVIDERS: dict[str, type[LLMClient]] = {
    AnthropicClient.name: AnthropicClient,
    OpenAIClient.name: OpenAIClient,
    OllamaClient.name: OllamaClient,
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
