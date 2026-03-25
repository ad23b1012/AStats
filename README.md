<div align="center">

# 📊 AStats

**Agentic AI for Applied Statistical Workflows**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GSoC 2025](https://img.shields.io/badge/GSoC-2025-orange.svg)](https://summerofcode.withgoogle.com/)

*An intelligent, LLM-powered system that helps statistical practitioners explore, analyze, and report on datasets with scientific rigor*

[Quick Start](#-quick-start) · [Features](#-features) · [Architecture](#-architecture) · [Usage](#-usage) · [Contributing](#-contributing)

</div>

---

## 🚀 What is AStats?

AStats is an **agentic AI system** that augments statistical practitioners by automating the tedious parts of data analysis while maintaining full transparency and human oversight. It uses a **Plan → Execute → Reflect** loop powered by free LLMs (Gemini Flash 2.5 and Groq) to:

- 🔍 **Auto-discover** dataset structure, types, and quality issues
- 📊 **Generate** comprehensive statistical profiles with one command
- 🤖 **Deploy specialized AI agents** for EDA, hypothesis testing, regression, and time-series analysis
- 📈 **Create** publication-quality visualizations automatically
- 📋 **Produce** professional HTML/Markdown analysis reports

> **The "A" in AStats stands for:** Autonomous, Augmented, Automatic, Applied — your choice.

## ✨ Features

### 🧠 Agentic AI Engine
- **Plan → Execute → Reflect** loop with configurable autonomy (full-auto, semi-auto, step-by-step)
- **4 specialized agents**: EDA, Hypothesis Testing, Regression, Time-Series
- **Automatic agent selection** — LLM routes your query to the right specialist
- **Tool calling** — agents invoke statistical operations via function calling

### 📂 Universal Data Ingestion
- Auto-detect and load: **CSV, TSV, Excel, Parquet, JSON, Feather, Stata, SPSS**
- Intelligent encoding detection and delimiter sniffing
- Large file sampling support

### 📊 Auto-Profiling
- Column type classification (continuous, discrete, categorical, binary, datetime, text)
- Univariate statistics (mean, std, skew, kurtosis)
- Missing data analysis with pattern detection
- Correlation analysis with top pairs
- Outlier detection (IQR method)

### 🤖 Free LLM Integration
- **Gemini Flash 2.5** (primary) — Google's free-tier model
- **Groq** (secondary) — Fast inference on Llama/Mixtral
- **No OpenAI dependency** — 100% free providers
- Streaming, function calling, retry with exponential backoff

### 📈 Smart Visualizations
- Auto-selects appropriate plot types based on data
- Professional styling with custom color palette
- Histograms, box plots, scatter, heatmaps, pair plots, Q-Q plots, violin plots

### 📋 Report Generation
- Professional HTML reports with embedded visualizations
- Markdown export for documentation
- Executive summaries powered by LLM narratives

### 💻 Rich CLI
- Interactive REPL with natural language querying
- Tab completion and command history
- Pretty-printed tables and panels via Rich

---

## 📦 Quick Start

### Prerequisites
- Python 3.10+
- A free API key from [Gemini](https://aistudio.google.com/apikey) or [Groq](https://console.groq.com/keys)

### Installation

```bash
# Clone the repository
git clone https://github.com/m2b3/AStats.git
cd AStats

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Set your API key
export GEMINI_API_KEY="your-key-here"
# OR
export GROQ_API_KEY="your-key-here"
```

### Verify Installation

```bash
astats --version
astats config --show
```

---

## 🎯 Usage

### Quick Profile

```bash
# Get a comprehensive statistical profile of any dataset
astats profile examples/data/sample_sales.csv
```

### Full Auto-Analysis

```bash
# Run a complete AI-powered analysis pipeline
astats analyze examples/data/sample_sales.csv

# With a specific question
astats analyze data.csv -q "What factors drive customer satisfaction?"

# With a specific agent
astats analyze data.csv --agent regression
```

### Interactive Mode

```bash
# Start an interactive exploration session
astats explore examples/data/sample_sales.csv
```

Then use natural language:
```
🔍 AStats> What's the distribution of revenue across regions?
🔍 AStats> Are there significant differences in satisfaction by product category?
🔍 AStats> Run a regression predicting profit from the other variables
🔍 AStats> /plots
🔍 AStats> /report
```

### Using Different Providers

```bash
# Use Groq instead of Gemini
astats analyze data.csv --provider groq --model llama-3.3-70b-versatile

# Use Gemini Flash (default)
astats analyze data.csv --provider gemini --model gemini-2.5-flash
```

### Python API

```python
from astats.config import AStatsConfig
from astats.agents.orchestrator import WorkflowOrchestrator

# Initialize
config = AStatsConfig.load()
orchestrator = WorkflowOrchestrator(config)

# Load and profile data
dataset = orchestrator.load_dataset("data.csv")

# Run analysis
results = orchestrator.run("What factors predict high revenue?")

# Or chat interactively
response = orchestrator.chat("Show me the correlation between age and satisfaction")
```

---

## 🏗️ Architecture

```
                          ┌─────────────────┐
                          │   CLI / REPL    │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  Orchestrator   │
                          │ Plan→Exec→Refl  │
                          └────────┬────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
     ┌────────▼──────┐  ┌────────▼──────┐  ┌────────▼──────┐
     │   EDA Agent   │  │ Hypothesis    │  │  Regression   │
     │               │  │ Agent         │  │  Agent        │
     └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
              │                  │                   │
              └──────────────────┼───────────────────┘
                                 │
                       ┌─────────▼─────────┐
                       │  Tool Registry    │
                       │ (run_code, plot,  │
                       │  test, fit_model) │
                       └─────────┬─────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
     ┌────────▼──────┐  ┌──────▼────────┐  ┌──────▼──────┐
     │ Data Engine   │  │ Viz Engine    │  │ LLM Layer   │
     │ (ingestion +  │  │ (matplotlib + │  │ (Gemini +   │
     │  profiler)    │  │  seaborn)     │  │  Groq)      │
     └───────────────┘  └───────────────┘  └─────────────┘
```

### Project Structure

```
AStats/
├── src/astats/
│   ├── config.py          # Pydantic settings with layered overrides
│   ├── logging.py         # Rich-formatted structured logging
│   ├── data/
│   │   ├── ingestion.py   # Universal data loader (10+ formats)
│   │   ├── profiler.py    # Auto statistical profiling engine
│   │   └── models.py      # Dataset, DataProfile, ColumnProfile
│   ├── llm/
│   │   ├── provider.py    # Gemini + Groq abstraction (free only)
│   │   ├── prompts.py     # System prompts, templates, tool defs
│   │   └── memory.py      # Conversation memory with persistence
│   ├── agents/
│   │   ├── orchestrator.py # Central workflow coordinator
│   │   ├── base.py        # Plan → Execute → Reflect base class
│   │   ├── tools.py       # Tool registry (code, plots, tests)
│   │   ├── eda.py         # Exploratory data analysis agent
│   │   ├── hypothesis.py  # Hypothesis testing agent
│   │   ├── regression.py  # Regression modeling agent
│   │   └── timeseries.py  # Time series analysis agent
│   ├── viz/
│   │   └── engine.py      # Professional auto-visualization
│   ├── reports/
│   │   └── generator.py   # HTML + Markdown report generation
│   └── cli/
│       ├── main.py        # Click CLI commands
│       └── repl.py        # Interactive REPL
├── tests/
├── examples/
├── pyproject.toml
└── CONTRIBUTING.md
```

---

## 🧪 Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/

# Type check
mypy src/
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR process.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.


---

<div align="center">
<em>Built with ❤️ for the statistical research community</em>
</div>
