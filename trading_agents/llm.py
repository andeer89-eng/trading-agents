"""
Universal LLM client — Anthropic, OpenAI, and any OpenAI-compatible provider.

Usage::

    client = LLMClient("anthropic", api_key="sk-ant-...", model="claude-opus-4-6")
    text   = client.chat("You are an analyst.", "Analyze AAPL.")

    for chunk in client.stream("You are an analyst.", "Synthesize this..."):
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

from typing import Generator


# ── Provider registry ─────────────────────────────────────────────────────────
# Each entry defines the label shown in the UI, the base URL (None = SDK default),
# the model list, default model, and the env-var to check for a stored key.

PROVIDERS: dict[str, dict] = {
    "anthropic": {
        "label": "Anthropic (Claude)",
        "base_url": None,
        "models": [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ],
        "default": "claude-opus-4-6",
        "key_env": "ANTHROPIC_API_KEY",
        "key_hint": "sk-ant-…  ·  console.anthropic.com",
    },
    "openai": {
        "label": "OpenAI (ChatGPT)",
        "base_url": None,
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "default": "gpt-4o",
        "key_env": "OPENAI_API_KEY",
        "key_hint": "sk-…  ·  platform.openai.com",
    },
    "groq": {
        "label": "Groq  (Llama — fast & free tier)",
        "base_url": "https://api.groq.com/openai/v1",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "llama-3.1-8b-instant",
        ],
        "default": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
        "key_hint": "gsk_…  ·  console.groq.com",
    },
    "openrouter": {
        "label": "OpenRouter  (50 + models)",
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            "anthropic/claude-opus-4-6",
            "anthropic/claude-sonnet-4-6",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "meta-llama/llama-3.3-70b-instruct",
            "google/gemini-pro-1.5",
        ],
        "default": "anthropic/claude-opus-4-6",
        "key_env": "OPENROUTER_API_KEY",
        "key_hint": "sk-or-…  ·  openrouter.ai/keys",
    },
    "custom": {
        "label": "Custom / Self-hosted  (OpenAI-compatible)",
        "base_url": None,   # user supplies
        "models": [],       # user supplies
        "default": "",
        "key_env": "CUSTOM_API_KEY",
        "key_hint": "Your API key",
    },
}


class LLMClient:
    """
    Thin unified wrapper around the Anthropic and OpenAI Python SDKs.

    - ``provider="anthropic"`` uses :pypi:`anthropic` SDK.
    - All other providers use :pypi:`openai` SDK pointed at the provider's
      base URL (or a user-supplied ``base_url``).
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        base_url: str | None = None,
    ) -> None:
        self.provider = provider.lower()
        self.model = model

        if self.provider == "anthropic":
            import anthropic  # noqa: PLC0415
            self._client = anthropic.Anthropic(api_key=api_key)
        else:
            import openai  # noqa: PLC0415
            url = base_url or PROVIDERS.get(self.provider, {}).get("base_url")
            self._client = openai.OpenAI(api_key=api_key, base_url=url)

    # ── Public interface ──────────────────────────────────────────────────────

    def chat(self, system: str, user: str, max_tokens: int = 3000) -> str:
        """Single-turn chat; returns the full response text."""
        if self.provider == "anthropic":
            r = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return r.content[0].text

        r = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return r.choices[0].message.content or ""

    def stream(
        self, system: str, user: str, max_tokens: int = 4000
    ) -> Generator[str, None, None]:
        """Streaming chat; yields text chunks as they arrive."""
        if self.provider == "anthropic":
            with self._client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as s:
                yield from s.text_stream
        else:
            stream = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                stream=True,
            )
            for chunk in stream:
                text = chunk.choices[0].delta.content or ""
                if text:
                    yield text
