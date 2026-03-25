"""Conversation memory management with token-aware truncation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from astats.llm.provider import LLMMessage
from astats.logging import get_logger

logger = get_logger("llm.memory")


class ConversationMemory:
    """Manages conversation history with intelligent truncation.

    Supports:
    - Sliding window (keep last N messages)
    - Character-budget truncation (approximate token control)
    - Session persistence to JSON
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        max_messages: int = 50,
        max_chars: int = 100_000,
    ) -> None:
        """Initialize conversation memory.

        Args:
            system_prompt: System prompt to always include.
            max_messages: Maximum messages to retain (excluding system).
            max_chars: Maximum total characters (approximate token budget).
        """
        self.system_prompt = system_prompt
        self.max_messages = max_messages
        self.max_chars = max_chars
        self._messages: list[LLMMessage] = []
        self._metadata: dict[str, Any] = {}

    @property
    def messages(self) -> list[LLMMessage]:
        """Get all messages including system prompt."""
        result = []
        if self.system_prompt:
            result.append(LLMMessage("system", self.system_prompt))
        result.extend(self._messages)
        return result

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def add_user(self, content: str) -> None:
        """Add a user message."""
        self._messages.append(LLMMessage("user", content))
        self._truncate()

    def add_assistant(self, content: str) -> None:
        """Add an assistant message."""
        self._messages.append(LLMMessage("assistant", content))
        self._truncate()

    def add_tool_result(self, tool_name: str, result: str) -> None:
        """Add a tool execution result as a user message."""
        formatted = f"[Tool: {tool_name}]\n{result}"
        self._messages.append(LLMMessage("user", formatted))
        self._truncate()

    def inject_context(self, context: str) -> None:
        """Inject contextual information into the conversation.

        This is added as a system-like user message that provides
        data context without being a "user query".
        """
        self._messages.append(
            LLMMessage("user", f"[Context Update]\n{context}")
        )
        self._truncate()

    def clear(self) -> None:
        """Clear all messages (keeps system prompt)."""
        self._messages.clear()

    def _truncate(self) -> None:
        """Apply truncation strategies to keep memory within bounds."""
        # Sliding window
        if len(self._messages) > self.max_messages:
            excess = len(self._messages) - self.max_messages
            self._messages = self._messages[excess:]
            logger.debug(f"Truncated {excess} old messages (sliding window)")

        # Character budget
        total_chars = sum(len(m.content) for m in self._messages)
        while total_chars > self.max_chars and len(self._messages) > 2:
            removed = self._messages.pop(0)
            total_chars -= len(removed.content)
            logger.debug(f"Truncated message (char budget): {removed.role}")

    def get_summary(self) -> str:
        """Generate a brief summary of the conversation so far."""
        if not self._messages:
            return "No conversation history."

        user_msgs = [m for m in self._messages if m.role == "user" and not m.content.startswith("[")]
        n_total = len(self._messages)
        n_user = len(user_msgs)

        lines = [f"Conversation: {n_total} messages ({n_user} user queries)"]
        if user_msgs:
            last = user_msgs[-1].content[:100]
            lines.append(f"Last query: {last}")

        return "\n".join(lines)

    # ---- Persistence ----

    def save(self, path: Path) -> None:
        """Save conversation to a JSON file."""
        data = {
            "system_prompt": self.system_prompt,
            "messages": [m.to_dict() for m in self._messages],
            "metadata": self._metadata,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved {len(self._messages)} messages to {path}")

    @classmethod
    def load(cls, path: Path) -> ConversationMemory:
        """Load conversation from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        memory = cls(system_prompt=data.get("system_prompt"))
        for msg_data in data.get("messages", []):
            memory._messages.append(
                LLMMessage(msg_data["role"], msg_data["content"])
            )
        memory._metadata = data.get("metadata", {})
        logger.debug(f"Loaded {len(memory._messages)} messages from {path}")
        return memory

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "message_count": len(self._messages),
            "total_chars": sum(len(m.content) for m in self._messages),
            "has_system_prompt": self.system_prompt is not None,
        }
