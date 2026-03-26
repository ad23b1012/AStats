<div align="center">

# 📊 AStats

**Agentic AI for Applied Statistical Workflows**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GSoC 2026](https://img.shields.io/badge/GSoC-2026-orange.svg)](https://summerofcode.withgoogle.com/)

*An intelligent, LLM-powered framework that helps statistical practitioners explore, analyze, and report on datasets with scientific rigor — using state-of-the-art open-weight and proprietary AI.*

</div>

---

## 🚀 What is AStats?

AStats is an **agentic AI system** that augments statistical practitioners by automating the tedious parts of data analysis while maintaining full transparency and human oversight. It uses a **Plan → Execute → Reflect** loop powered by multiple LLM providers to:

- 🔍 **Auto-discover** dataset structure, types, and quality issues
- 📊 **Generate** comprehensive statistical profiles with one command
- 🤖 **Deploy specialized AI agents** for EDA, hypothesis testing, regression, and time-series analysis
- 📈 **Create** publication-quality visualizations automatically
- 📋 **Produce** professional HTML/Markdown analysis reports

> **The "A" in AStats stands for:** Autonomous, Augmented, Automatic, Applied — your choice.

---

## ✨ Key Features

### 🧠 Agentic AI Engine
- **Plan → Execute → Reflect** loop with configurable autonomy levels
- **4 specialized agents**: EDA, Hypothesis Testing, Regression, Time-Series
- **Automatic agent selection** — LLM routes queries to the right specialist
- **Tool calling** — agents invoke statistical operations via structured function calling

### 🤖 Multi-Provider LLM Abstraction
| Provider | Models | Type |
|----------|--------|------|
| **Google Gemini** | Gemini 2.5 Flash | Free-tier cloud API |
| **Groq** | Llama 3 70B, Mixtral | Free-tier cloud API |
| **Anthropic** | Claude 3.5 Sonnet/Opus/Haiku | Proprietary cloud API |
| **OpenAI** | GPT-4o, Codex | Proprietary cloud API |

All providers support streaming, function calling, and automatic retry with exponential backoff.

### 📂 Universal Data Ingestion
- Auto-detect and load: **CSV, TSV, Excel, Parquet, JSON, Feather, Stata, SPSS**
- Intelligent encoding detection and delimiter sniffing

### 📊 Auto-Profiling
- Column type classification (continuous, discrete, categorical, binary, datetime, text)
- Univariate statistics (mean, std, skew, kurtosis)
- Missing data analysis with pattern detection
- Correlation analysis with top pairs
- Outlier detection (IQR method)

### 📈 Smart Visualizations
- Auto-selects appropriate plot types based on data characteristics
- Professional styling with custom color palette
- Distribution plots, box plots, correlation heatmaps, missing data charts

### 📋 Report Generation
- Professional HTML reports with embedded base64 visualizations
- Markdown export for documentation
- Executive summaries powered by LLM analysis narratives

---

## 🧪 Validated on Real-World Datasets

AStats has been tested end-to-end on standard scientific datasets:

| Dataset | Rows | Task | What It Demonstrates |
|---------|------|------|---------------------|
| **Fisher's Iris** | 150 | Exploratory Data Analysis | Auto-profiling, species grouping, correlation discovery |
| **Diabetes** | 442 | Regression Modeling | OLS fitting, VIF multicollinearity check, Ridge/Lasso regularization |
| **Titanic** | 100 | Hypothesis Testing | Chi-Square, t-test, Mann-Whitney U, One-Way ANOVA |

Each dataset has a fully documented example script in the `examples/` directory.

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
              ┌─────────────────--─┼────────────────┐
              │                    │                │
     ┌────────▼──────┐  ┌────────--▼────┐  ┌────────▼──────┐
     │   EDA Agent   │  │ Hypothesis    │  │  Regression   │
     │               │  │ Agent         │  │  Agent        │
     └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
             │                  │                  │
             └────────────-─────┼────────────────-─┘
                                │
                      ┌─────────▼─────────┐
                      │  Tool Registry    │
                      │ (run_code, plot,  │
                      │  test, fit_model) │
                      └─────────┬─────────┘
                                │
              ┌─────────────────┼──────────────────┐
              │                 │                  │
     ┌────────▼──────┐  ┌────--─▼───────┐  ┌──────-▼─────┐
     │ Data Engine   │  │ Viz Engine    │  │ LLM Layer   │
     │ (ingestion +  │  │ (matplotlib + │  │ (Gemini,    │
     │  profiler)    │  │  seaborn)     │  │ Groq,Claude │
     └───────────────┘  └───────────────┘  │ OpenAI)     │
                                           └─────────────┘
```

### Project Structure

```
AStats/
├── src/astats/
│   ├── config.py          # Pydantic settings & provider configuration
│   ├── logging.py         # Rich-formatted structured logging
│   ├── data/
│   │   ├── ingestion.py   # Universal data loader (10+ formats)
│   │   ├── profiler.py    # Auto statistical profiling engine
│   │   └── models.py      # Dataset, DataProfile, ColumnProfile
│   ├── llm/
│   │   ├── provider.py    # Multi-provider abstraction (Gemini, Groq, Claude, OpenAI)
│   │   ├── prompts.py     # System prompts and tool definitions
│   │   └── memory.py      # Conversation memory with persistence
│   ├── agents/
│   │   ├── orchestrator.py # Central workflow coordinator
│   │   ├── base.py        # Plan → Execute → Reflect base agent
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
│       ├── main.py        # Click CLI entry point
│       └── repl.py        # Interactive REPL
├── examples/
│   ├── data/              # Real-world datasets (Iris, Diabetes, Titanic)
│   ├── eda_iris_example.py
│   ├── regression_diabetes_example.py
│   └── hypothesis_titanic_example.py
├── METHODOLOGY.md
├── CONTRIBUTING.md
├── pyproject.toml
└── LICENSE
```

---

## 📈 Methodology

Read [METHODOLOGY.md](METHODOLOGY.md) for a comprehensive deep-dive into the statistical tests, assumption checks, and pipelines implemented in the AStats agents, including:

- **EDA**: Data profiling, normality screening, outlier detection, bivariate associations
- **Hypothesis Testing**: Parametric (t-test, ANOVA) and non-parametric (Mann-Whitney, Kruskal-Wallis, Chi-Square) with effect sizes
- **Regression**: OLS diagnostics (Durbin-Watson, Breusch-Pagan, VIF), Ridge/Lasso regularization

---

## 📦 Getting Started

### Prerequisites
- Python 3.10+
- At least one API key from: [Gemini](https://aistudio.google.com/apikey), [Groq](https://console.groq.com/keys), [Anthropic](https://console.anthropic.com/), or [OpenAI](https://platform.openai.com/api-keys)

### Installation

```bash
git clone https://github.com/ad23b1012/AStats.git
cd AStats
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Set your API key(s) in your environment or a `.env` file (see `.env.example`).

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
