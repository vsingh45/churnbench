"""ChurnBench command-line interface."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from rich import print

from churnbench.generator.timeline import TimelineConfig, TimelineSimulator
from churnbench.ledger.ledger import Ledger

app = typer.Typer(help="ChurnBench — drift-aware benchmark for agentic AI grounding.")


@app.command()
def generate(
    days: int = typer.Option(180, help="Timeline length in days."),
    seed: int = typer.Option(42, help="Deterministic seed."),
    out: Path = typer.Option(..., help="Output directory."),
    start: str = typer.Option("2024-01-01", help="Timeline start date (ISO)."),
) -> None:
    """Run the timeline simulator and persist the ground-truth ledger."""
    cfg = TimelineConfig(start=date.fromisoformat(start), days=days, seed=seed)
    ledger = Ledger()
    print(f"[cyan]Simulating {days} days from {cfg.start}, seed={seed}…[/cyan]")
    TimelineSimulator(cfg, ledger).run()
    ledger_path = out / "ledger.jsonl"
    ledger.save(ledger_path)
    print(f"[green]Wrote {len(ledger)} ledger events to {ledger_path}[/green]")


@app.command()
def snapshot(run: Path, at: str) -> None:
    """Project the ledger onto Postgres+Mongo+API+docs at timestamp T. (TODO)"""
    print(f"[yellow]snapshot not yet implemented — will project {run} at {at}[/yellow]")


@app.command()
def tasks(run: Path, n: int = 180) -> None:
    """Generate n evaluation tasks from the ledger. (TODO)"""
    print(f"[yellow]tasks not yet implemented — will emit {n} tasks for {run}[/yellow]")


@app.command()
def run(arm: str, run: Path) -> None:  # noqa: A002 — CLI verb
    """Run one experimental arm against the current snapshot. (TODO)"""
    print(f"[yellow]run not yet implemented — arm={arm} run={run}[/yellow]")


@app.command()
def score(run: Path) -> None:
    """Score results against the ledger. (TODO)"""
    print(f"[yellow]score not yet implemented for {run}[/yellow]")


if __name__ == "__main__":
    app()
