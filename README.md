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

Or step by step: `collect --niche X [--source google_play|computrabajo|youtube] [--limit N]`, `extract`,
`cluster`, `score`, `report`. Reports land in `data/reports/YYYY-MM-DD.md`; everything else
in `data/signals.db`. Re-running `collect` only adds new signals; `extract` only processes
signals it hasn't seen (`--retry-failed` re-runs failures).

## Sources

- `google_play` — 1–2 star reviews of the `google_play_apps` in the Colombian store.
- `computrabajo` — job posts for the `computrabajo_queries` (search pages, then each offer's
  page). Computrabajo's legal terms forbid scripts and crawlers; it runs at low volume by the
  owner's decision (2026-11-09): 3 s between requests (2 s minimum), robots.txt checked before
  every request, offers already in the DB never re-fetched, at most 30 offer pages per query per
  run, and HTTP 403/429 stops the source. Company names, emails, phones and links are never
  stored. Delete `computrabajo_queries` from a niche to turn it off.
- `youtube` — comments on the `youtube_video_ids` (ids or URLs) via the Data API v3. Needs
  `YOUTUBE_API_KEY` in `.env` (Google Cloud project with "YouTube Data API v3" enabled, then an
  API key restricted to that API); without it the source is skipped. Each video costs 1 quota unit
  plus 1 per 100 comments (free default: 10,000 units a day). Comments by the video's own channel
  are left out; commenters are only hashed, and @mentions, emails, phones and links are scrubbed.
  The video title is added as the text's first line (`Video: …`) after filtering.
- Mercado Libre — parked (2026-11-09), no collector. Tested with a registered app, using both an
  app token (client credentials) and a user token: search, catalog, items and reviews all return
  403 (`PA_UNAUTHORIZED_RESULT_FROM_POLICIES`). Only `/questions/search?item=` answers, and that
  gives few, short pre-sale questions. Pages can't be scraped instead: the spec allows only the API
  for this site.

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
