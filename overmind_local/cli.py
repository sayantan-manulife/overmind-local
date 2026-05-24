"""CLI: overmind-local <command>"""
import json
import sqlite3
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()
DEFAULT_DIR = ".overmind"


def _db(dir_: str) -> Path:
    return Path(dir_) / "traces.db"


@click.group()
@click.version_option("0.1.0", prog_name="overmind-local")
def cli():
    """Overmind Local — self-hosted LLM agent observability and optimization.\n
    Zero cloud. Everything stays in .overmind/traces.db (SQLite).
    """
    pass


# ── init ───────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--dir", "dir_", default=DEFAULT_DIR, show_default=True,
              help="Directory for local storage")
def init(dir_):
    """Initialise local storage (SQLite, no network needed)."""
    from overmind_local.storage import init as _init
    path = _init(dir_)
    console.print(f"[green]✓[/green] Initialised at [bold]{path}[/bold]")
    console.print("  No API key required. All data stays local.")


# ── traces ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--agent", default=None, help="Filter by agent name")
@click.option("--limit", default=30, show_default=True)
@click.option("--dir", "dir_", default=DEFAULT_DIR, show_default=True)
@click.option("--errors-only", is_flag=True, default=False)
def traces(agent, limit, dir_, errors_only):
    """View recent spans from local SQLite."""
    db = _db(dir_)
    if not db.exists():
        console.print("[red]No traces DB found — run `overmind-local init` first.[/red]")
        return

    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        q = "SELECT * FROM spans WHERE 1=1"
        p: list = []
        if agent:
            q += " AND agent_name = ?"
            p.append(agent)
        if errors_only:
            q += " AND error IS NOT NULL"
        q += " ORDER BY start_time DESC LIMIT ?"
        p.append(limit)
        rows = conn.execute(q, p).fetchall()

    if not rows:
        console.print("[yellow]No spans found.[/yellow]")
        return

    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Name", max_width=36)
    t.add_column("Type", max_width=12)
    t.add_column("Agent", max_width=18)
    t.add_column("Duration", justify="right")
    t.add_column("Status")

    for r in rows:
        t.add_row(
            r["name"],
            r["span_type"] or "-",
            r["agent_name"] or "-",
            f"{r['duration_ms']:.0f}ms" if r["duration_ms"] else "-",
            "[red]✗ " + (r["error"] or "")[:35] if r["error"] else "[green]✓[/green]",
        )

    console.print(t)
    console.print(f"[dim]{len(rows)} span(s) shown[/dim]")


# ── policies ───────────────────────────────────────────────────────────────────

@cli.group()
def policy():
    """Manage optimization policies for an agent."""
    pass


@policy.command("add")
@click.argument("agent_name")
@click.option("--name", required=True, help="Short policy name")
@click.option("--description", required=True, help="What the policy enforces")
@click.option("--dir", "dir_", default=DEFAULT_DIR, show_default=True)
def policy_add(agent_name, name, description, dir_):
    """Add a policy that the optimizer must respect."""
    from overmind_local.storage import add_policy, init as _init
    _init(dir_)
    add_policy(agent_name, name, description, db_path=_db(dir_))
    console.print(f"[green]✓[/green] Policy '[bold]{name}[/bold]' added for agent '{agent_name}'")


@policy.command("list")
@click.argument("agent_name")
@click.option("--dir", "dir_", default=DEFAULT_DIR, show_default=True)
def policy_list(agent_name, dir_):
    """List policies for an agent."""
    from overmind_local.storage import get_policies
    rows = get_policies(agent_name, db_path=_db(dir_))
    if not rows:
        console.print(f"[yellow]No policies for '{agent_name}'.[/yellow]")
        return
    for r in rows:
        console.print(f"  [cyan]{r['name']}[/cyan]: {r['description']}")


# ── dataset ────────────────────────────────────────────────────────────────────

@cli.group()
def dataset():
    """Manage test cases for an agent."""
    pass


@dataset.command("add")
@click.argument("agent_name")
@click.option("--input", "input_json", required=True,
              help="JSON string of input (e.g. '{\"query\":\"...\"}' )")
@click.option("--expected", default=None, help="Expected output or behaviour description")
@click.option("--dir", "dir_", default=DEFAULT_DIR, show_default=True)
def dataset_add(agent_name, input_json, expected, dir_):
    """Add a test case to the dataset."""
    from overmind_local.storage import add_dataset_item, init as _init
    _init(dir_)
    try:
        input_data = json.loads(input_json)
    except json.JSONDecodeError:
        input_data = {"raw": input_json}
    add_dataset_item(agent_name, input_data, expected, db_path=_db(dir_))
    console.print(f"[green]✓[/green] Dataset item added for '{agent_name}'")


@dataset.command("list")
@click.argument("agent_name")
@click.option("--dir", "dir_", default=DEFAULT_DIR, show_default=True)
def dataset_list(agent_name, dir_):
    """List test cases for an agent."""
    from overmind_local.storage import get_dataset
    rows = get_dataset(agent_name, db_path=_db(dir_))
    if not rows:
        console.print(f"[yellow]No dataset items for '{agent_name}'.[/yellow]")
        return
    for r in rows:
        inp = r.get("input", "")
        exp = r.get("expected_output", "")
        console.print(f"  Input: [cyan]{inp[:80]}[/cyan]" +
                      (f"\n  Expected: {exp[:80]}" if exp else ""))


# ── optimize ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("agent_name")
@click.option("--model", default="gpt-4o", show_default=True,
              help="LLM model for analysis (uses your own API key via litellm)")
@click.option("--traces-limit", default=50, show_default=True)
@click.option("--dir", "dir_", default=DEFAULT_DIR, show_default=True)
def optimize(agent_name, model, traces_limit, dir_):
    """Analyse traces and suggest concrete agent improvements."""
    from overmind_local.optimize import run_optimization
    db = _db(dir_)
    if not db.exists():
        console.print("[red]No traces DB found — run `overmind-local init` first.[/red]")
        return

    console.print(f"[blue]Analysing {traces_limit} traces for '{agent_name}' "
                  f"using [bold]{model}[/bold]...[/blue]")
    result = run_optimization(agent_name, model=model, trace_limit=traces_limit, db_path=db)
    console.print(Panel(Markdown(result), title=f"Optimization Report — {agent_name}",
                        border_style="green"))


# ── stats ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--dir", "dir_", default=DEFAULT_DIR, show_default=True)
def stats(dir_):
    """Show summary statistics across all agents."""
    db = _db(dir_)
    if not db.exists():
        console.print("[red]No traces DB found.[/red]")
        return

    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute("""
            SELECT agent_name,
                   COUNT(*) as total,
                   SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as errors,
                   AVG(duration_ms) as avg_ms,
                   MAX(created_at) as last_seen
            FROM spans
            GROUP BY agent_name
            ORDER BY total DESC
        """).fetchall()

    if not rows:
        console.print("[yellow]No data yet.[/yellow]")
        return

    t = Table(header_style="bold cyan")
    t.add_column("Agent")
    t.add_column("Spans", justify="right")
    t.add_column("Errors", justify="right")
    t.add_column("Error %", justify="right")
    t.add_column("Avg ms", justify="right")
    t.add_column("Last Seen")

    for r in rows:
        total, errors = r[1], r[2]
        pct = f"{100*errors/total:.1f}%" if total else "0%"
        t.add_row(
            r[0] or "(unnamed)",
            str(total),
            str(errors),
            f"[red]{pct}[/red]" if errors else pct,
            f"{r[3]:.0f}" if r[3] else "-",
            str(r[4] or "-")[:19],
        )

    console.print(t)


def main():
    cli()
