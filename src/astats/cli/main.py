"""AStats CLI — command-line interface for statistical analysis.

Commands:
    astats analyze <file>   — Full auto-analysis pipeline
    astats explore <file>   — Interactive EDA session
    astats chat             — Open-ended conversation with data
    astats profile <file>   — Quick data profiling
    astats config           — View/edit configuration
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from astats import __version__

console = Console()

BANNER = r"""
    _    ____  _        _
   / \  / ___|| |_ __ _| |_ ___
  / _ \ \___ \| __/ _` | __/ __|
 / ___ \ ___) | || (_| | |_\__ \
/_/   \_\____/ \__\__,_|\__|___/

  Agentic AI for Statistical Workflows
"""


@click.group()
@click.version_option(version=__version__, prog_name="AStats")
@click.option("--provider", type=click.Choice(["gemini", "groq"]), default=None, help="LLM provider")
@click.option("--model", type=str, default=None, help="Model name")
@click.option("--api-key", type=str, default=None, envvar="ASTATS_API_KEY", help="API key")
@click.option("--output-dir", type=click.Path(), default="./output", help="Output directory")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default="INFO")
@click.pass_context
def cli(
    ctx: click.Context,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    output_dir: str,
    log_level: str,
) -> None:
    """AStats — Agentic AI for Applied Statistical Workflows.

    Explore, analyze, and report on datasets with AI-powered statistical agents.
    Uses Gemini Flash 2.5 (default) or Groq for free LLM inference.
    """
    from astats.config import AStatsConfig, LLMProviderType
    from astats.logging import setup_logging
    
    # Load environment variables from .env
    from dotenv import load_dotenv
    load_dotenv()

    # Load configuration
    config = AStatsConfig.load()

    # Apply CLI overrides
    if provider:
        config.llm.provider = LLMProviderType(provider)
    if model:
        config.llm.model = model
    if api_key:
        config.llm.api_key = api_key
    config.analysis.output_dir = Path(output_dir)
    config.log.level = log_level

    # Setup logging
    setup_logging(level=log_level)

    # Ensure output dirs
    config.ensure_dirs()

    # Store in context
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--query", "-q", type=str, default=None, help="Specific analysis question")
@click.option("--agent", type=click.Choice(["eda", "hypothesis", "regression", "timeseries"]),
              default=None, help="Specific agent to use")
@click.option("--report/--no-report", default=True, help="Generate HTML report")
@click.pass_context
def analyze(
    ctx: click.Context,
    file: str,
    query: str | None,
    agent: str | None,
    report: bool,
) -> None:
    """Run a full automated analysis pipeline on a dataset."""
    config = ctx.obj["config"]
    _show_banner()

    from astats.agents.orchestrator import WorkflowOrchestrator
    from astats.reports.generator import ReportGenerator
    from astats.viz.engine import VizEngine

    orchestrator = WorkflowOrchestrator(config)

    # Load and profile
    console.print("\n[bold cyan]📂 Loading dataset...[/bold cyan]")
    dataset = orchestrator.load_dataset(file)

    # Auto-visualize
    console.print("\n[bold cyan]📈 Generating visualizations...[/bold cyan]")
    viz = VizEngine(output_dir=str(config.analysis.output_dir / "plots"))
    plots = viz.auto_visualize(dataset)

    # Run analysis
    default_query = query or f"Perform a comprehensive exploratory analysis of the {dataset.name} dataset. Identify key patterns, distributions, correlations, anomalies, and actionable insights."

    console.print("\n[bold cyan]🤖 Running analysis...[/bold cyan]")
    results = orchestrator.run(default_query, agent_type=agent)

    # Generate report
    if report:
        console.print("\n[bold cyan]📋 Generating report...[/bold cyan]")
        gen = ReportGenerator(output_dir=str(config.analysis.output_dir))
        report_path = gen.generate(dataset, results, plots=plots)
        md_path = gen.generate_markdown(dataset, results)
        console.print(f"\n[bold green]✅ Report saved:[/bold green] {report_path}")
        console.print(f"[bold green]✅ Markdown:[/bold green] {md_path}")

    console.print("\n[bold green]✨ Analysis complete![/bold green]")


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.pass_context
def profile(ctx: click.Context, file: str) -> None:
    """Quick data profiling — load and display dataset statistics."""
    config = ctx.obj["config"]
    _show_banner()

    from astats.data.ingestion import DataLoader
    from astats.data.profiler import AutoProfiler

    loader = DataLoader()
    dataset = loader.load(file)

    profiler = AutoProfiler()
    prof = profiler.profile(dataset)

    # Display as rich table
    table = Table(title=f"📊 Profile: {dataset.name}", show_lines=True)
    table.add_column("Column", style="bold cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Missing", style="yellow")
    table.add_column("Unique")
    table.add_column("Key Stats", style="green")

    for cp in prof.column_profiles:
        stats = ""
        if cp.mean is not None:
            stats = f"μ={cp.mean:.2f}, σ={cp.std:.2f}, range=[{cp.min:.2f}, {cp.max:.2f}]"
        elif cp.top_values:
            stats = ", ".join(f"{v['value']} ({v['pct']:.0%})" for v in cp.top_values[:3])

        table.add_row(
            cp.name,
            cp.semantic_type.value,
            f"{cp.missing_pct:.1%}" if cp.missing_pct > 0 else "✓",
            str(cp.unique_count),
            stats,
        )

    console.print(table)

    # Summary panel
    console.print(Panel(
        f"Rows: {prof.n_rows:,} | Columns: {prof.n_cols} | "
        f"Missing: {prof.missing_report.missing_pct:.1%} | "
        f"Memory: {prof.memory_usage_mb:.1f} MB | "
        f"Time: {prof.profiling_time_seconds:.2f}s",
        title="Summary",
        border_style="blue",
    ))


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.pass_context
def explore(ctx: click.Context, file: str) -> None:
    """Interactive EDA session — explore your dataset with AI assistance."""
    config = ctx.obj["config"]
    _show_banner()

    from astats.cli.repl import start_repl

    start_repl(config, file)


@cli.command()
@click.argument("file", type=click.Path(exists=True), required=False)
@click.pass_context
def chat(ctx: click.Context, file: str | None) -> None:
    """Open-ended AI chat about data."""
    config = ctx.obj["config"]
    _show_banner()

    from astats.cli.repl import start_repl

    start_repl(config, file)


@cli.command("config")
@click.option("--show", is_flag=True, help="Show current config")
@click.pass_context
def config_cmd(ctx: click.Context, show: bool) -> None:
    """View or edit AStats configuration."""
    config = ctx.obj["config"]

    if show:
        table = Table(title="⚙️ AStats Configuration")
        table.add_column("Setting", style="bold cyan")
        table.add_column("Value", style="green")

        table.add_row("LLM Provider", config.llm.provider.value)
        table.add_row("Model", config.llm.model)
        table.add_row("Temperature", str(config.llm.temperature))
        table.add_row("Max Tokens", str(config.llm.max_tokens))
        table.add_row("Autonomy", config.analysis.autonomy_level)
        table.add_row("Output Dir", str(config.analysis.output_dir))
        table.add_row("Log Level", config.log.level)

        console.print(table)
    else:
        console.print("Use [bold]astats config --show[/bold] to view current settings")
        console.print("Edit [bold]~/.astats/config.toml[/bold] to change defaults")


def _show_banner() -> None:
    """Display the AStats banner."""
    console.print(f"[bold magenta]{BANNER}[/bold magenta]")
    console.print(f"  [dim]v{__version__} | Free-tier AI (Gemini + Groq)[/dim]\n")


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
