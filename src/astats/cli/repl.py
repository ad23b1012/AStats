"""Interactive REPL for AStats — explore datasets conversationally."""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from astats.config import AStatsConfig
from astats.logging import get_logger

logger = get_logger("cli.repl")
console = Console()

HELP_TEXT = """
[bold cyan]AStats Interactive Mode[/bold cyan]

Commands:
  [bold]/help[/bold]         — Show this help
  [bold]/profile[/bold]      — Show dataset profile
  [bold]/columns[/bold]      — List columns
  [bold]/head[/bold] [N]     — Show first N rows
  [bold]/agent[/bold] <type> — Switch agent (eda, hypothesis, regression, timeseries)
  [bold]/auto[/bold]         — Run auto-analysis
  [bold]/plots[/bold]        — Generate auto-visualizations
  [bold]/report[/bold]       — Generate analysis report
  [bold]/save[/bold]         — Save session
  [bold]/quit[/bold]         — Exit

Type any question or analysis request in natural language.
"""


def start_repl(config: AStatsConfig, file_path: str | None = None) -> None:
    """Start the interactive REPL.

    Args:
        config: AStats configuration.
        file_path: Optional path to a dataset to load on start.
    """
    from astats.agents.orchestrator import WorkflowOrchestrator
    from astats.reports.generator import ReportGenerator
    from astats.viz.engine import VizEngine

    orchestrator = WorkflowOrchestrator(config)

    # Load dataset if provided
    if file_path:
        try:
            orchestrator.load_dataset(file_path)
        except Exception as e:
            console.print(f"[bold red]Error loading file:[/bold red] {e}")
            return

    console.print(HELP_TEXT)

    # Setup prompt with history
    history_path = Path.home() / ".astats" / "repl_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_path)),
    )

    current_agent: str | None = None

    while True:
        try:
            user_input = session.prompt("\n🔍 AStats> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Command handling
        if user_input.startswith("/"):
            _handle_command(user_input, orchestrator, config, current_agent)
            if user_input.lower() in ("/quit", "/exit", "/q"):
                break
            if user_input.lower().startswith("/agent"):
                parts = user_input.split()
                if len(parts) > 1:
                    current_agent = parts[1]
                    console.print(f"[bold magenta]Agent set to: {current_agent}[/bold magenta]")
            continue

        # Natural language query
        if orchestrator.dataset is None:
            console.print(
                "[yellow]No dataset loaded. Use:[/yellow] "
                "[bold]astats explore <file>[/bold]"
            )
            continue

        try:
            with console.status("[bold magenta]Thinking...[/bold magenta]"):
                response = orchestrator.chat(user_input)

            console.print(Panel(
                Markdown(response),
                title="🤖 AStats",
                border_style="magenta",
            ))
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            logger.exception("REPL error")


def _handle_command(
    cmd: str,
    orchestrator: Any,
    config: AStatsConfig,
    current_agent: str | None,
) -> None:
    """Handle slash commands."""
    from astats.reports.generator import ReportGenerator
    from astats.viz.engine import VizEngine

    parts = cmd.strip().split()
    command = parts[0].lower()

    if command in ("/help", "/h"):
        console.print(HELP_TEXT)

    elif command == "/profile":
        if orchestrator.dataset and orchestrator.dataset.profile:
            console.print(Panel(
                orchestrator.dataset.profile.summary_for_llm(),
                title="📊 Data Profile",
                border_style="blue",
            ))
        else:
            console.print("[yellow]No dataset profiled yet.[/yellow]")

    elif command == "/columns":
        if orchestrator.dataset:
            for col in orchestrator.dataset.df.columns:
                dtype = str(orchestrator.dataset.df[col].dtype)
                console.print(f"  [cyan]{col}[/cyan] ({dtype})")
        else:
            console.print("[yellow]No dataset loaded.[/yellow]")

    elif command == "/head":
        if orchestrator.dataset:
            n = int(parts[1]) if len(parts) > 1 else 5
            console.print(orchestrator.dataset.df.head(n).to_string())
        else:
            console.print("[yellow]No dataset loaded.[/yellow]")

    elif command == "/auto":
        if orchestrator.dataset:
            query = " ".join(parts[1:]) if len(parts) > 1 else \
                f"Perform comprehensive EDA on the {orchestrator.dataset.name} dataset"
            with console.status("[bold magenta]Running auto-analysis...[/bold magenta]"):
                results = orchestrator.run(query, agent_type=current_agent)
            if results.get("summary"):
                console.print(Panel(
                    Markdown(results["summary"]),
                    title="📋 Analysis Results",
                    border_style="green",
                ))

    elif command == "/plots":
        if orchestrator.dataset:
            viz = VizEngine(output_dir=str(config.analysis.output_dir / "plots"))
            with console.status("[bold cyan]Generating plots...[/bold cyan]"):
                plots = viz.auto_visualize(orchestrator.dataset)
            console.print(f"[green]Generated {len(plots)} plots in {config.analysis.output_dir / 'plots'}[/green]")

    elif command == "/report":
        if orchestrator.dataset:
            gen = ReportGenerator(output_dir=str(config.analysis.output_dir))
            path = gen.generate(orchestrator.dataset)
            console.print(f"[green]Report saved: {path}[/green]")

    elif command == "/save":
        path = orchestrator.save_session()
        console.print(f"[green]Session saved: {path}[/green]")

    elif command in ("/quit", "/exit", "/q"):
        console.print("[dim]Goodbye![/dim]")

    else:
        console.print(f"[yellow]Unknown command: {command}. Type /help for commands.[/yellow]")
