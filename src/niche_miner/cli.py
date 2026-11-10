"""niche-miner: collect → extract → cluster → score → report. See SPEC.md."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated, Any

import typer
from dotenv import load_dotenv

from . import autoreview
from . import cluster as clustering
from . import db
from . import report as reporting
from . import score as scoring
from .collectors import REGISTRY
from .extract import Extractor
from .llm import LLM, has_credentials
from .settings import Paths, load_niches, load_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
log = logging.getLogger("niche_miner")

Niche = Annotated[str, typer.Option("--niche", "-n", help="Niche tag from config/niches.yaml.")]
NicheFilter = Annotated[
    str | None, typer.Option("--niche", "-n", help="Only this niche (default: all).")]
Limit = Annotated[int | None, typer.Option(
    min=1, help="Test runs: max signals kept per source, split across its targets.")]


class Context:
    def __init__(self) -> None:
        self.paths = Paths()
        load_dotenv(self.paths.env)
        self.settings = load_settings(self.paths.config)
        self.niches = load_niches(self.paths.config)
        self.conn = db.connect(self.paths.db)
        self.run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._llm: LLM | None = None

    def niche(self, tag: str) -> dict[str, Any]:
        if tag not in self.niches:
            known = ", ".join(self.niches) or "none"
            raise typer.BadParameter(f"unknown niche {tag!r} (known: {known})",
                                     param_hint="--niche")
        return self.niches[tag]

    def llm(self) -> LLM:
        if not has_credentials():
            typer.secho("ANTHROPIC_API_KEY is not set: add it to .env (see .env.example).",
                        fg="red", err=True)
            raise typer.Exit(1)
        if self._llm is None:
            self._llm = LLM(self.conn, self.settings.llm, self.run_id)
        return self._llm


@app.callback()
def main(ctx: typer.Context,
         verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False
         ) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)-7s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "anthropic", "sentence_transformers", "urllib3",
                  "filelock", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ctx.obj = Context()


def _collect(c: Context, niche: str, source: str | None, limit: int | None) -> None:
    cfg = c.niche(niche)
    if source and source not in REGISTRY:
        raise typer.BadParameter(f"unknown source {source!r} (available: {', '.join(REGISTRY)})",
                                 param_hint="--source")
    for name, collector_cls in REGISTRY.items():
        if source and name != source:
            continue
        collector = collector_cls(c.settings, limit=limit,
                                  known_ids=db.external_ids(c.conn, name))
        signals = collector.collect(cfg)
        new = db.insert_signals(c.conn, signals)
        typer.echo(f"collect {name}: {len(signals)} signals kept, {new} new "
                   f"({len(signals) - new} already in the DB)")


def _usage_line(c: Context, purposes: tuple[str, ...], n_items: int | None = None) -> str:
    usage = [u for u in db.llm_usage(c.conn, c.run_id) if u["purpose"] in purposes]
    if not usage:
        return "no LLM calls"
    cost = sum(u["cost_usd"] for u in usage)
    line = (f"{sum(u['calls'] for u in usage)} calls, "
            f"{sum(u['input_tokens'] for u in usage):,} in / "
            f"{sum(u['output_tokens'] for u in usage):,} out tokens, ${cost:.4f}")
    if n_items:
        line += f" ≈ ${1000 * cost / n_items:.2f} per 1,000 signals"
    return line + (" (some calls unpriced)" if any(u["unpriced"] for u in usage) else "")


def _extract(c: Context, niche: str | None, limit: int | None = None, model: str | None = None,
             retry_failed: bool = False, redo: bool = False) -> None:
    if niche:
        c.niche(niche)
    if not (retry_failed or redo) and not db.pending_signals(c.conn, niche, 1):
        typer.echo("extract: nothing new to extract")
        return
    stats = Extractor(c.conn, c.llm(), c.settings.llm, model).run(niche, limit, retry_failed, redo)
    typer.echo(f"extract: {stats.signals} signals → {stats.complaints} complaints, "
               f"{stats.failed} failed ({stats.retried} retried, {stats.spend_dropped} "
               f"non-verbatim spend/cost quotes dropped, {stats.country_reset} Maps/Play "
               f"countries set to unknown)")
    typer.echo(f"extract cost: {_usage_line(c, ('extract',), stats.signals)}")
    if redo:
        typer.echo("extract: previous clusters lost their members; run `cluster` next")


METHODS = ("auto", "hdbscan", "llm")


def _cluster(c: Context, niche: str | None, method: str | None = None) -> None:
    if niche:
        c.niche(niche)
    if method and method not in METHODS:
        raise typer.BadParameter(f"choose one of {', '.join(METHODS)}", param_hint="--method")
    tags = [niche] if niche else db.niches_with_complaints(c.conn)
    if not tags:
        typer.echo("cluster: no complaint pains yet; run `extract` first")
        return
    llm = c.llm()
    for tag in tags:
        used, created, noise = clustering.cluster_niche(
            c.conn, llm, tag, c.run_id, c.settings.clustering, c.settings.llm.label_model, method)
        typer.echo(f"cluster {tag}: {created} clusters, {noise} pains left as noise "
                   f"(method {used}, run {c.run_id})")
    typer.echo(f"clustering cost: {_usage_line(c, ('label', 'themes', 'assign'))}")


def _review(c: Context, niche: str | None = None, refresh: bool = False) -> None:
    if niche:
        c.niche(niche)
    stats = autoreview.Reviewer(c.conn, c.settings, c.paths.config, c.paths.review,
                                c.llm(), c.niches).run(niche, refresh)
    typer.echo(f"review: competition for {stats.reviewed} clusters ({stats.cached} from the "
               f"cache, {stats.failed} failed); reachability for {stats.niches_reviewed} niches "
               f"({stats.niches_cached} from the cache); Google search about "
               f"${stats.search_usd:.4f} (final figure in the Apify console)")
    typer.echo(f"review cost: {_usage_line(c, ('review',))}")


def _score(c: Context) -> None:
    scored = scoring.score_latest(c.conn, c.paths.config, c.settings.scoring, c.paths.review)
    typer.echo(f"score: {len(scored)} clusters scored")
    for s in [s for s in scored if s["status"] != "not_a_niche"][:5]:
        co = f"{s['co_only']:5.1f}" if s["co_only"] is not None else "    —"
        typer.echo(f"  {s['score']:5.1f} (old frequency {s['old']:5.1f}, co-only {co})  "
                   f"#{s['id']} [{s['niche']}] {s['label']}")


def _report(c: Context, niche: str | None = None, top: int | None = None) -> None:
    path = reporting.write_report(c.conn, c.settings, c.paths.reports, niche=niche, top=top,
                                  review_dir=c.paths.review)
    typer.echo(f"report: {path}")


@app.command()
def collect(ctx: typer.Context, niche: Niche,
            source: Annotated[str | None, typer.Option(help="Only this source.")] = None,
            limit: Limit = None) -> None:
    """Fetch public signals for a niche into data/signals.db (deduplicated)."""
    _collect(ctx.obj, niche, source, limit)


@app.command()
def extract(ctx: typer.Context, niche: NicheFilter = None,
            limit: Annotated[int | None, typer.Option(min=1, help="Max signals this run.")] = None,
            model: Annotated[str | None, typer.Option(help="Override llm.extract_model.")] = None,
            retry_failed: Annotated[bool, typer.Option(help="Re-run failed extractions.")] = False,
            redo: Annotated[bool, typer.Option(
                help="Re-extract every signal (after a prompt change); re-run `cluster` after.")
            ] = False) -> None:
    """Extract pains from unprocessed signals with Claude (batches of 20)."""
    _extract(ctx.obj, niche, limit, model, retry_failed, redo)


@app.command()
def cluster(ctx: typer.Context, niche: NicheFilter = None,
            method: Annotated[str | None, typer.Option(
                help="auto | hdbscan | llm (default: clustering.method, auto = llm below "
                     "clustering.llm_max_pains pains).")] = None) -> None:
    """Group complaint pains into labeled clusters (writes a new run)."""
    _cluster(ctx.obj, niche, method)


@app.command()
def review(ctx: typer.Context, niche: NicheFilter = None,
           refresh: Annotated[bool, typer.Option(
               help="Ignore the cache in data/review and review every active cluster again.")
           ] = False) -> None:
    """Automated competition per cluster and reachability per niche (Apify Google search,
    Play Store, Computrabajo, YouTube, Claude); cached in data/review/."""
    _review(ctx.obj, niche, refresh)


@app.command()
def score(ctx: typer.Context) -> None:
    """Score the latest clusters (merges.yaml, reachability.yaml overrides, data/review)."""
    _score(ctx.obj)


@app.command()
def report(ctx: typer.Context, niche: NicheFilter = None,
           top: Annotated[int | None, typer.Option(min=1, help="Clusters to show.")] = None
           ) -> None:
    """Write data/reports/YYYY-MM-DD.md."""
    _report(ctx.obj, niche, top)


@app.command("run-all")
def run_all(ctx: typer.Context, niche: Niche, limit: Limit = None) -> None:
    """collect → extract → cluster → review → score → report for one niche."""
    c: Context = ctx.obj
    _collect(c, niche, None, limit)
    _extract(c, niche)
    _cluster(c, niche)
    _review(c, niche)
    _score(c)
    _report(c)
    typer.echo(f"LLM cost this run: ${sum(u['cost_usd'] for u in db.llm_usage(c.conn, c.run_id)):.4f}")
