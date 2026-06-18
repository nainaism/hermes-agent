"""Ollama Cloud provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class OllamaCloudProfile(ProviderProfile):
    """Ollama Cloud provider — think parameter support.

    Ollama Cloud supports ``think: true|false`` (toggle only, no effort
    levels).  Reasoning content is returned in the ``reasoning`` field.
    """

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}

        if reasoning_config and isinstance(reasoning_config, dict):
            _enabled = reasoning_config.get("enabled", True)
            if _enabled:
                extra_body["think"] = True
            else:
                extra_body["think"] = False

        return extra_body, {}


ollama_cloud = OllamaCloudProfile(
    name="ollama-cloud",
    aliases=("ollama_cloud",),
    default_aux_model="nemotron-3-nano:30b",
    env_vars=("OLLAMA_API_KEY",),
    base_url="https://ollama.com/v1",
)

register_provider(ollama_cloud)