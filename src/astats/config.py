"""AStats configuration management.

Supports layered configuration from:
1. Default values
2. Config file (~/.astats/config.toml)
3. Environment variables (ASTATS_*)
4. CLI overrides
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class LLMProviderType(str, Enum):
    """Supported LLM providers."""

    GEMINI = "gemini"
    GROQ = "groq"
    CLAUDE = "claude"
    OPENAI = "openai"


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: LLMProviderType = Field(
        default=LLMProviderType.GEMINI,
        description="LLM provider to use",
    )
    model: str = Field(
        default="gemini-2.5-flash",
        description="Model name for the selected provider",
    )
    api_key: str | None = Field(
        default=None,
        description="API key (falls back to env var)",
    )
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8192, gt=0)
    timeout: int = Field(default=120, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0)

    def get_api_key(self) -> str:
        """Resolve API key from config or environment."""
        if self.api_key:
            return self.api_key
        env_map = {
            LLMProviderType.GEMINI: "GEMINI_API_KEY",
            LLMProviderType.GROQ: "GROQ_API_KEY",
            LLMProviderType.CLAUDE: "ANTHROPIC_API_KEY",
            LLMProviderType.OPENAI: "OPENAI_API_KEY",
        }
        env_var = env_map.get(self.provider, "")
        key = os.environ.get(env_var, "")
        if not key:
            raise ValueError(
                f"No API key found for {self.provider.value}. "
                f"Set {env_var} environment variable or pass --api-key."
            )
        return key


class AnalysisConfig(BaseModel):
    """Analysis behavior configuration."""

    autonomy_level: str = Field(
        default="semi-auto",
        description="Autonomy level: 'full-auto', 'semi-auto', 'step-by-step'",
    )
    max_iterations: int = Field(
        default=10,
        gt=0,
        description="Max agentic loop iterations",
    )
    output_dir: Path = Field(
        default=Path("./output"),
        description="Directory for generated outputs",
    )
    save_plots: bool = Field(default=True)
    plot_format: str = Field(default="png", description="Plot format: png, svg, pdf")
    plot_dpi: int = Field(default=150, gt=0)


class LogConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    file: Path | None = Field(default=None, description="Log file path")
    rich_console: bool = Field(default=True)


class AStatsConfig(BaseModel):
    """Root configuration for AStats."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    log: LogConfig = Field(default_factory=LogConfig)

    @classmethod
    def load(cls, config_path: Path | None = None) -> AStatsConfig:
        """Load configuration from file, env vars, and defaults.

        Priority: CLI overrides > env vars > config file > defaults.
        """
        data: dict[str, Any] = {}

        # Try loading from config file
        path = config_path or Path.home() / ".astats" / "config.toml"
        if path.exists():
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore[no-redef]
            with open(path, "rb") as f:
                data = tomllib.load(f)

        # Build config with env var overrides
        config = cls(**data)

        # Override from environment
        if env_provider := os.environ.get("ASTATS_LLM_PROVIDER"):
            config.llm.provider = LLMProviderType(env_provider)
        if env_model := os.environ.get("ASTATS_LLM_MODEL"):
            config.llm.model = env_model
        if env_level := os.environ.get("ASTATS_LOG_LEVEL"):
            config.log.level = env_level

        return config

    def ensure_dirs(self) -> None:
        """Create required directories."""
        self.analysis.output_dir.mkdir(parents=True, exist_ok=True)
        if self.log.file:
            self.log.file.parent.mkdir(parents=True, exist_ok=True)
