# Contributing to AStats

Thank you for your interest in contributing to AStats! This guide will help you get set up.

## Development Setup

### Prerequisites
- Python 3.10+
- Git

### Getting Started

```bash
# Fork and clone
git clone https://github.com/<your-username>/AStats.git
cd AStats

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Set up API keys for testing
export GEMINI_API_KEY="your-key"
```

## Code Style

- **Formatter**: We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting
- **Type hints**: All public functions must have type annotations
- **Docstrings**: Use Google-style docstrings for all public APIs
- **Line length**: 100 characters max

```bash
# Check for issues
ruff check src/

# Auto-fix
ruff check src/ --fix

# Type checking
mypy src/
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=astats --cov-report=html

# Run specific test file
pytest tests/test_profiler.py -v
```

## Pull Request Process

1. **Fork** the repository and create your branch from `main`
2. **Write tests** for any new functionality
3. **Update docs** if you changed public APIs
4. **Run the full test suite** before submitting
5. **Open a PR** with a clear description of changes

## Project Structure

See the [README](README.md#project-structure) for the full project layout.

## Adding a New Agent

1. Create `src/astats/agents/your_agent.py`
2. Inherit from `BaseAgent`
3. Set `agent_name` and `system_prompt`
4. Register in `WorkflowOrchestrator.AGENT_MAP`
5. Add tests in `tests/`

## Adding a New Tool

1. Create the tool function in `src/astats/agents/tools.py`
2. Register it in `ToolRegistry._register_builtins()`
3. Add the tool definition to `src/astats/llm/prompts.py`
4. Add tests

## Questions?

Open an issue or reach out to the mentors listed in the README.
