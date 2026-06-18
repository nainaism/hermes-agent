"""ZAI / GLM provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class ZAIProfile(ProviderProfile):
    """Z.AI / GLM provider — thinking + reasoning_effort support.

    GLM-5.x models support two parameters:
    - ``thinking: {"type": "enabled"|"disabled"}`` — toggle thinking on/off
    - ``reasoning_effort: str`` — control thinking depth (GLM-5.2+ only)
      Values: max, xhigh, high, medium, low, minimal, none
      Z.AI internally maps: xhigh→max, medium/low→high, none/minimal→skip

    Reasoning content is returned in the ``reasoning_content`` field.
    """

    # Map Hermes reasoning effort levels to ZAI reasoning_effort values
    _EFFORT_MAP = {
        "off": None,        # → thinking disabled
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
        "max": "max",
    }

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}

        if reasoning_config and isinstance(reasoning_config, dict):
            _enabled = reasoning_config.get("enabled", True)
            _effort = (reasoning_config.get("effort") or "medium").strip().lower()

            if not _enabled or _effort == "off" or _effort == "none":
                # Disable thinking entirely
                extra_body["thinking"] = {"type": "disabled"}
            else:
                # Enable thinking + set reasoning_effort
                extra_body["thinking"] = {"type": "enabled"}
                zai_effort = self._EFFORT_MAP.get(_effort)
                if zai_effort:
                    # Always send reasoning_effort explicitly so the API
                    # doesn't fall back to its default (max) when we want
                    # a lower level like low/medium/high.
                    extra_body["reasoning_effort"] = zai_effort

        return extra_body, {}


zai = ZAIProfile(
    name="zai",
    aliases=("glm", "z-ai", "z.ai", "zhipu"),
    env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
    display_name="Z.AI (GLM)",
    description="Z.AI / GLM — Zhipu AI models",
    signup_url="https://z.ai/",
    fallback_models=(
        "glm-5.2",
        "glm-5",
        "glm-4-9b",
    ),
    base_url="https://api.z.ai/api/paas/v4",
    default_aux_model="glm-4.5-flash",
)

register_provider(zai)