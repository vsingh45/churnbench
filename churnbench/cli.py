"""ChurnBench command-line interface."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from rich import print

from churnbench.generator.timeline import TimelineConfig, TimelineSimulator
from churnbench.ledger.ledger import Ledger

app = typer.Typer(help="ChurnBench — drift-aware benchmark for agentic AI grounding.")

_ARM_NAMES = ["naive", "classic_rag", "hierarchical", "grounding"]
_ARM_FLAGS: dict[str, dict[str, bool]] = {
    "grounding_no_freshness_tiers": {"no_freshness_tiers": True},
    "grounding_no_semantic_model": {"no_semantic_model": True},
    "grounding_no_source_routing": {"no_source_routing": True},
}


def _build_arm(arm_name: str) -> object:
    """Instantiate an arm by name (including ablation variants)."""
    from churnbench.arms import (
        ClassicRagArm,
        GroundingArm,
        HierarchicalArm,
        NaiveArm,
    )

    flags = _ARM_FLAGS.get(arm_name, {})
    match arm_name.split("_no_")[0]:
        case "naive":
            return NaiveArm()
        case "classic_rag":
            return ClassicRagArm()
        case "hierarchical":
            return HierarchicalArm()
        case "grounding":
            return GroundingArm(**flags)
        case _:
            print(f"[red]Unknown arm: {arm_name}[/red]")
            raise typer.Exit(code=1)


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
def snapshot(
    run: Path = typer.Argument(..., help="Run directory containing ledger.jsonl."),
    at: str = typer.Argument(..., help="Freeze timestamp ISO date, e.g. 2024-06-30."),
    docs_out: Path = typer.Option(
        None,
        help="Directory for rendered contract docs. Defaults to <run>/docs.",
    ),
) -> None:
    """Project the ledger onto Postgres, Mongo, and docs at timestamp T."""
    from churnbench.fabric.projector import Projector

    T = date.fromisoformat(at)
    ledger_path = run / "ledger.jsonl"
    if not ledger_path.exists():
        print(f"[red]Ledger not found: {ledger_path}[/red]")
        raise typer.Exit(code=1)

    print(f"[cyan]Loading ledger from {ledger_path}…[/cyan]")
    ledger = Ledger.load(ledger_path)
    print(f"[cyan]{len(ledger)} events loaded. Projecting at T={T}…[/cyan]")

    proj = Projector()
    docs_dir = docs_out if docs_out is not None else run / "docs"

    print("[cyan]→ Postgres…[/cyan]")
    pg_counts = proj.project_postgres(ledger, T)
    for table, n in pg_counts.items():
        print(f"   {table}: {n} rows")

    print("[cyan]→ MongoDB…[/cyan]")
    mongo_counts = proj.project_mongo(ledger, T)
    for coll, n in mongo_counts.items():
        print(f"   {coll}: {n} docs")

    print(f"[cyan]→ Docs → {docs_dir}…[/cyan]")
    n_docs = proj.project_docs(ledger, T, docs_dir)
    print(f"   contracts rendered: {n_docs}")

    print(f"[green]Snapshot complete. Frozen at T={T}.[/green]")


@app.command()
def tasks(
    run: Path = typer.Option(..., help="Run directory containing ledger.jsonl."),
    at: str = typer.Option(..., help="Freeze timestamp ISO date, e.g. 2024-06-15."),
    n: int = typer.Option(180, help="Number of tasks to generate."),
    seed: int = typer.Option(7, help="Deterministic seed."),
) -> None:
    """Generate evaluation tasks from the ledger frozen at T."""
    from churnbench.tasks.generator import TaskSetGenerator

    T = date.fromisoformat(at)
    ledger_path = run / "ledger.jsonl"
    if not ledger_path.exists():
        print(f"[red]Ledger not found: {ledger_path}[/red]")
        raise typer.Exit(code=1)

    ledger = Ledger.load(ledger_path)
    print(f"[cyan]{len(ledger)} events loaded. Generating {n} tasks at T={T}…[/cyan]")

    gen = TaskSetGenerator(ledger=ledger, T=T, seed=seed, n=n)
    task_list = gen.generate_and_save(run)

    tier_counts: dict[int, int] = {}
    for t in task_list:
        tier_counts[t.tier] = tier_counts.get(t.tier, 0) + 1

    print(f"[green]Wrote {len(task_list)} tasks to {run / 'tasks.jsonl'}[/green]")
    for tier, count in sorted(tier_counts.items()):
        print(f"   tier-{tier}: {count} tasks")


@app.command()
def run(  # noqa: A002 — CLI verb, shadows builtin
    arm_name: str = typer.Argument(..., help="Arm to run (naive, classic_rag, hierarchical, grounding, grounding_no_*)."),
    run_dir: Path = typer.Argument(..., help="Run directory (must contain ledger.jsonl and tasks.jsonl)."),
    t_prime: str = typer.Option(..., "--t-prime", help="Cache-build timestamp ISO date."),
    t: str = typer.Option(..., "--t", help="Evaluation timestamp ISO date."),
    seed: int = typer.Option(42, help="Deterministic seed for config hash."),
) -> None:
    """Run one experimental arm against the frozen snapshot and score results."""
    from churnbench.arms.base import FabricConfig
    from churnbench.eval.harness import RunHarness
    from churnbench.tasks.resolver import LedgerResolver
    from churnbench.tasks.schema import load_tasks

    ledger_path = run_dir / "ledger.jsonl"
    tasks_path = run_dir / "tasks.jsonl"

    for p in (ledger_path, tasks_path):
        if not p.exists():
            print(f"[red]Not found: {p}[/red]")
            raise typer.Exit(code=1)

    T_prime = date.fromisoformat(t_prime)
    T_eval = date.fromisoformat(t)

    print(f"[cyan]Loading ledger ({ledger_path})…[/cyan]")
    ledger = Ledger.load(ledger_path)
    print(f"[cyan]{len(ledger)} events. T′={T_prime}, T={T_eval}[/cyan]")

    task_list = load_tasks(tasks_path)
    print(f"[cyan]{len(task_list)} tasks loaded.[/cyan]")

    resolver = LedgerResolver(ledger)
    arm = _build_arm(arm_name)

    results_dir = run_dir / "results"
    harness = RunHarness(run_dir=results_dir, fabric_config=FabricConfig())

    print(f"[cyan]Running arm '{arm_name}'…[/cyan]")
    from churnbench.arms.base import BaseArm
    assert isinstance(arm, BaseArm)

    all_results = harness.run(
        arm=arm,
        arm_name=arm_name,
        T_prime=T_prime,
        T=T_eval,
        ledger=ledger,
        tasks=task_list,
        resolver=resolver,
        seed=seed,
    )

    from churnbench.eval.scoring import compute_summary
    summary = compute_summary(all_results)
    print(
        f"[green]Done. {summary.n_tasks} tasks: "
        f"{summary.n_correct} correct ({summary.accuracy*100:.1f}%), "
        f"{summary.n_freshness_error} freshness-error, "
        f"{summary.n_reasoning_error} reasoning-error, "
        f"{summary.n_parse_failure} parse-failure. "
        f"Total cost: ${summary.total_cost_usd:.4f}[/green]"
    )


@app.command()
def score(
    run_dir: Path = typer.Argument(..., help="Run directory containing results/*.json."),
    t_prime: str = typer.Option(..., "--t-prime", help="Cache-build timestamp ISO date."),
    t: str = typer.Option(..., "--t", help="Evaluation timestamp ISO date."),
    report_out: Path = typer.Option(
        None, "--report-out", help="Path for the Markdown report. Default: <run_dir>/report.md."
    ),
) -> None:
    """Load all results from a run directory and generate the paper tables."""
    from churnbench.eval.report import generate_report, load_arm_results

    results_dir = run_dir / "results"
    if not results_dir.exists():
        print(f"[red]Results directory not found: {results_dir}[/red]")
        raise typer.Exit(code=1)

    by_arm = load_arm_results(results_dir)
    if not by_arm:
        print(f"[yellow]No result files found in {results_dir}[/yellow]")
        raise typer.Exit(code=1)

    out = report_out if report_out is not None else run_dir / "report.md"
    generate_report(by_arm, T_prime=t_prime, T=t, out_path=out)
    print(f"[green]Report written to {out}[/green]")

    from churnbench.eval.scoring import compute_summary
    for arm_name, results in sorted(by_arm.items()):
        m = compute_summary(results)
        print(
            f"  {arm_name}: acc={m.accuracy*100:.1f}% "
            f"fresh_err={m.freshness_error_rate*100:.1f}% "
            f"cost=${m.total_cost_usd:.4f}"
        )


@app.command()
def smoke(
    arm_name: str = typer.Option("grounding", "--arm", help="Arm to smoke-test."),
    seed: int = typer.Option(42, help="Deterministic seed."),
    n_tasks: int = typer.Option(5, "--n", help="Number of tasks to run."),
) -> None:
    """Run a 5-task smoke test with real API calls to verify the stack end-to-end.

    Generates a fresh 30-day ledger in a temp directory, builds caches at day 10,
    evaluates at day 24, and prints verdicts. Calls the real LLM API — intended
    for pre-experiment validation, not CI.
    """
    import tempfile

    from churnbench.arms.base import FabricConfig
    from churnbench.eval.harness import RunHarness
    from churnbench.tasks.generator import TaskSetGenerator
    from churnbench.tasks.resolver import LedgerResolver

    start = date(2024, 3, 1)
    T_prime = date(2024, 3, 10)
    T_eval = date(2024, 3, 24)

    print("[cyan]Smoke test: generating 30-day ledger…[/cyan]")
    cfg = TimelineConfig(start=start, days=30, seed=seed)
    ledger = Ledger()
    TimelineSimulator(cfg, ledger).run()
    print(f"[cyan]{len(ledger)} events generated.[/cyan]")

    task_list = TaskSetGenerator(ledger=ledger, T=T_eval, seed=seed, n=n_tasks).generate()
    print(f"[cyan]{len(task_list)} tasks generated.[/cyan]")

    resolver = LedgerResolver(ledger)
    arm = _build_arm(arm_name)

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        harness = RunHarness(run_dir=run_dir, fabric_config=FabricConfig())
        from churnbench.arms.base import BaseArm
        assert isinstance(arm, BaseArm)
        results = harness.run(
            arm=arm,
            arm_name=arm_name,
            T_prime=T_prime,
            T=T_eval,
            ledger=ledger,
            tasks=task_list,
            resolver=resolver,
            seed=seed,
        )

    print(f"\n[bold]Smoke results ({arm_name}):[/bold]")
    for r in results:
        mark = {
            "correct": "[green]✓[/green]",
            "freshness_error": "[yellow]~[/yellow]",
            "reasoning_error": "[red]✗[/red]",
            "parse_failure": "[red]?[/red]",
        }.get(r.verdict, "?")
        print(
            f"  {mark} [{r.verdict}] {r.question_text[:60]!r} "
            f"→ {r.answer_raw[:30]!r}"
        )

    from churnbench.eval.scoring import compute_summary
    m = compute_summary(results)
    print(
        f"\n[bold]Summary:[/bold] "
        f"{m.n_correct}/{m.n_tasks} correct, "
        f"{m.n_freshness_error} freshness-err, "
        f"cost=${m.total_cost_usd:.6f}"
    )


if __name__ == "__main__":
    app()
