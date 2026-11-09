# niche-miner

Personal CLI research pipeline: public complaints from sources Colombians use →
LLM-extracted pains → clusters → scores → weekly Markdown report. Design: [SPEC.md](SPEC.md).

## Setup

```bash
uv sync
cp .env.example .env   # then set ANTHROPIC_API_KEY
```

## Run

```bash
uv run niche-miner run-all --niche contadores_pymes --limit 50   # quick test run
uv run niche-miner run-all --niche contadores_pymes              # full run
```

Or step by step: `collect --niche X [--source google_play] [--limit N]`, `extract`,
`cluster`, `score`, `report`. Reports land in `data/reports/YYYY-MM-DD.md`; everything else
in `data/signals.db`. Re-running `collect` only adds new signals; `extract` only processes
signals it hasn't seen (`--retry-failed` re-runs failures).

## Review loop

1. Read the report. Each cluster has an id, score breakdown, quotes and a blank `Hypothesis:`.
2. Fill `config/reachability.yaml` (reachability/competition 0–10 per cluster id; the report
   ends with a paste-ready stub) and `config/merges.yaml` (`target_id: [ids]`).
3. `uv run niche-miner score && uv run niche-miner report` — no re-clustering needed.

Cluster ids are global: a new `cluster` run creates new ids, so redo step 2 after re-clustering.

## Config

- `config/niches.yaml` — seeds per niche; each collector reads only its own key.
- `config/settings.yaml` — model, effort, batch size, pricing (cost logging), filters,
  HDBSCAN params, scoring weights.

## Tests

```bash
uv run pytest            # offline: parsers on fixtures, schema validation, scoring, dedupe
uv run pytest -m live    # hits the API with tests/fixtures/extraction_cases.json (< 1 cent)
```

Pick the extraction model by running the live suite with
`NICHE_MINER_TEST_MODEL=<model>` and keeping the cheapest one that passes.
