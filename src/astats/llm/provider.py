"""LLM provider abstraction layer.

Supports Gemini Flash 2.5 (primary) and Groq (secondary) — both free tier.
No OpenAI dependency. Uses litellm as a fallback unified router.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from tenacity import retry, stop_after_attempt, wait_exponential

from astats.config import AStatsConfig, LLMProviderType
from astats.logging import get_logger

logger = get_logger("llm.provider")


class LLMMessage:
    """A single message in a conversation."""

    def __init__(self, role: str, content: str) -> None:
        self.role = role  # "system", "user", "assistant"
        self.content = content

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    def __repr__(self) -> str:
        preview = self.content[:80] + "..." if len(self.content) > 80 else self.content
        return f"LLMMessage(role={self.role!r}, content={preview!r})"


class LLMResponse:
    """Response from an LLM provider."""

    def __init__(
        self,
        content: str,
        model: str,
        provider: str,
        usage: dict[str, int] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        raw: Any = None,
    ) -> None:
        self.content = content
        self.model = model
        self.provider = provider
        self.usage = usage or {}
        self.tool_calls = tool_calls
        self.raw = raw

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)

    def __repr__(self) -> str:
        return (
            f"LLMResponse(provider={self.provider!r}, model={self.model!r}, "
            f"tokens={self.total_tokens})"
        )


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: AStatsConfig) -> None:
        self.config = config

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: Conversation history.
            tools: Optional tool/function definitions for function calling.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            LLMResponse with the model's reply.
        """
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response."""
        ...


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider using google-genai SDK.

    Primary provider — Gemini Flash 2.5 is free tier.
    """

    def __init__(self, config: AStatsConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self) -> Any:
        """Lazy-init the Gemini client."""
        if self._client is None:
            from google import genai

            api_key = self.config.llm.get_api_key()
            self._client = genai.Client(api_key=api_key)
            logger.info(f"[agent]Gemini[/agent] client initialized (model: {self.config.llm.model})")
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def chat(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat request to Gemini."""
        from google.genai import types

        client = self._get_client()
        temp = temperature if temperature is not None else self.config.llm.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.llm.max_tokens

        # Build contents from messages
        # Gemini uses a different format: system instruction is separate
        system_instruction = None
        contents = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg.content)],
                ))

        # Build generation config
        gen_config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=max_tok,
            system_instruction=system_instruction,
        )

        # Build tool declarations if provided
        if tools:
            gemini_tools = self._convert_tools(tools)
            gen_config.tools = gemini_tools

        response = client.models.generate_content(
            model=self.config.llm.model,
            contents=contents,
            config=gen_config,
        )

        # Extract content
        content = ""
        tool_calls = None
        if response.candidates:
            candidate = response.candidates[0]
            parts = candidate.content.parts if (candidate.content and candidate.content.parts) else []
            text_parts = []
            fc_parts = []
            for part in parts:
                if part.text:
                    text_parts.append(part.text)
                if part.function_call:
                    fc_parts.append({
                        "name": part.function_call.name,
                        "arguments": dict(part.function_call.args) if part.function_call.args else {},
                    })
            content = "\n".join(text_parts)
            if fc_parts:
                tool_calls = fc_parts

        # Extract usage
        usage = {}
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                "total_tokens": response.usage_metadata.total_token_count or 0,
            }

        logger.debug(
            f"Gemini response: {usage.get('total_tokens', '?')} tokens"
        )

        return LLMResponse(
            content=content,
            model=self.config.llm.model,
            provider="gemini",
            usage=usage,
            tool_calls=tool_calls,
            raw=response,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream response from Gemini."""
        from google.genai import types

        client = self._get_client()
        temp = temperature if temperature is not None else self.config.llm.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.llm.max_tokens

        system_instruction = None
        contents = []
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg.content)],
                ))

        gen_config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=max_tok,
            system_instruction=system_instruction,
        )

        response_stream = client.models.generate_content_stream(
            model=self.config.llm.model,
            contents=contents,
            config=gen_config,
        )

        for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[Any]:
        """Convert OpenAI-style tool definitions to Gemini format."""
        from google.genai import types

        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                fd = types.FunctionDeclaration(
                    name=func["name"],
                    description=func.get("description", ""),
                    parameters=func.get("parameters"),
                )
                function_declarations.append(fd)

        return [types.Tool(function_declarations=function_declarations)]


class GroqProvider(BaseLLMProvider):
    """Groq provider for fast inference on open-weight models.

    Secondary provider — free tier with rate limits.
    Models: llama-3.3-70b-versatile, mixtral-8x7b-32768, etc.
    """

    def __init__(self, config: AStatsConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            from groq import Groq

            api_key = self.config.llm.get_api_key()
            self._client = Groq(api_key=api_key)
            logger.info(f"[agent]Groq[/agent] client initialized (model: {self.config.llm.model})")
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def chat(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat request to Groq."""
        client = self._get_client()
        temp = temperature if temperature is not None else self.config.llm.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.llm.max_tokens

        msg_dicts = [m.to_dict() for m in messages]

        kwargs: dict[str, Any] = {
            "model": self.config.llm.model,
            "messages": msg_dicts,
            "temperature": temp,
            "max_tokens": max_tok,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        content = choice.message.content or ""

        # Extract tool calls if present
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments)
                    if tc.function.arguments
                    else {},
                }
                for tc in choice.message.tool_calls
            ]

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=content,
            model=self.config.llm.model,
            provider="groq",
            usage=usage,
            tool_calls=tool_calls,
            raw=response,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream response from Groq."""
        client = self._get_client()
        temp = temperature if temperature is not None else self.config.llm.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.llm.max_tokens

        msg_dicts = [m.to_dict() for m in messages]

        response_stream = client.chat.completions.create(
            model=self.config.llm.model,
            messages=msg_dicts,
            temperature=temp,
            max_tokens=max_tok,
            stream=True,
        )

        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def create_provider(config: AStatsConfig) -> BaseLLMProvider:
    """Factory function to create the appropriate LLM provider.

    Args:
        config: AStats configuration.

    Returns:
        An initialized LLM provider.
    """
    providers = {
        LLMProviderType.GEMINI: GeminiProvider,
        LLMProviderType.GROQ: GroqProvider,
    }

    provider_cls = providers.get(config.llm.provider)
    if provider_cls is None:
        raise ValueError(
            f"Unsupported LLM provider: {config.llm.provider}. "
            f"Supported: {', '.join(p.value for p in LLMProviderType)}"
        )

    return provider_cls(config)
