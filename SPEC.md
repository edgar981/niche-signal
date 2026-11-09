# Niche Signal Mining Pipeline — Build Spec

## Goal
A personal, CLI-only Python pipeline that collects public complaints from
sources Colombians actually use, extracts the underlying pain with an LLM,
clusters pains into candidate niche problems, scores them by willingness to
pay and reachability, and emits a weekly Markdown report. Output is read by
one person (me). This is a research tool, not a product.

## Non-goals (do not build)
- No web UI, dashboard, auth, or multi-user anything.
- No hand-rolled scrapers for Facebook, TikTok, or Google Maps — use Apify
  actors via the `apify-client` package.
- No custom payment/delivery code. Out of scope entirely.
- No scheduling/cron inside the project; I run it manually.
- No Reddit collector in v1.

## Stack
Python 3.12, `uv` for deps, SQLite (`data/signals.db`), `typer` CLI,
`anthropic` SDK for extraction, `sentence-transformers`
(`paraphrase-multilingual-MiniLM-L12-v2`) for embeddings, `scikit-learn`
HDBSCAN for clustering, `pydantic` for schemas, `python-dotenv` for secrets.
All secrets in `.env` (gitignored); provide `.env.example`.

## Repo layout
```
niche-miner/
  SPEC.md
  pyproject.toml
  .env.example
  config/niches.yaml          # seed lists per niche (see below)
  src/niche_miner/
    cli.py                    # typer app: collect, extract, cluster, score, report, run-all
    db.py                     # SQLite schema + helpers
    models.py                 # pydantic: RawSignal, ExtractedPain, Cluster
    collectors/
      base.py                 # Collector ABC -> list[RawSignal]
      google_play.py          # google-play-scraper, country=co, lang=es, score<=2
      mercadolibre.py         # public REST API: item questions + reviews (site MCO)
      youtube.py              # Data API v3 commentThreads for listed video IDs
      computrabajo.py         # HTML scrape of search results + job detail text
      apify_facebook.py       # Apify FB groups posts/comments actor
      apify_gmaps.py          # Apify Google Maps reviews actor
      apify_tiktok.py         # Apify TikTok comments actor
    extract.py                # LLM pain extraction
    cluster.py                # embeddings + HDBSCAN + manual merge file
    score.py                  # scoring formula
    report.py                 # Markdown report generation
  data/                       # gitignored: signals.db, reports/
  tests/                      # unit tests for parsers, scoring, schema validation
```

## Config: `config/niches.yaml`
One entry per niche I want to explore. Example:
```yaml
niches:
  - tag: contadores_pymes
    google_play_apps: [co.com.siigo.app, com.alegra.app]
    mercadolibre_queries: ["software contable", "facturación electrónica"]
    youtube_video_ids: [..]
    computrabajo_queries: ["auxiliar contable excel", "conciliación bancaria"]
    facebook_groups: ["https://www.facebook.com/groups/..."]
    gmaps_queries: ["software contable Bogotá"]
    tiktok_hashtags: ["contadorescolombia"]
```
Every collector reads only its own keys; missing keys = skip, not crash.

## Data model
`raw_signals` table (dedupe on `(source, external_id)`):
`id, niche_tag, source, external_id, url, author_hash, author_type
(business|consumer|unknown), text, lang, posted_at, collected_at`.

`extracted_pains`: `id, raw_signal_id, is_complaint, pain, who, current_tool,
spend_signal, severity (1-5), model, extracted_at`.

`clusters`: `id, label, niche_tag, run_id, size, distinct_authors,
spend_count, b2b_ratio, reachability, competition, score, representative_ids`.

Hash authors (SHA-256 of source+author id). Never store names or emails.

## Collectors
- Shared ABC with `collect(niche_cfg) -> list[RawSignal]`, built-in retry with
  backoff, per-source rate limit, and a `--limit` flag for test runs.
- Respect each site's robots/ToS; Computrabajo: polite delays (≥2s), realistic
  UA, stop on 403. Mercado Libre: use the public API, not HTML.
- Apify collectors: actor IDs and input shapes go in config, not code, so I can
  swap actors. Fail soft if `APIFY_TOKEN` is absent (log + skip).
- Keep only Spanish text (`langdetect`), min 40 chars, drop obvious spam.

## Extraction (`extract.py`)
For each unprocessed raw signal, call Claude with a system prompt that says:
analyze a Colombian Spanish text; return strict JSON matching:
```json
{"is_complaint": bool,
 "pain": "one sentence, in Spanish, describing the unmet need",
 "who": "role or business type, e.g. 'dueño de restaurante', 'contador independiente'",
 "current_tool": "named tool, or 'manual/Excel/WhatsApp', or null",
 "spend_signal": "verbatim quote showing money/time spent or hiring, or null",
 "severity": 1-5}
```
Batch 20 texts per request to cut cost; validate with pydantic; on parse
failure retry once then mark failed. Use the cheapest Claude model that passes
the test fixtures. Make the model name configurable.

## Clustering (`cluster.py`)
Embed `pain` for `is_complaint=true`. HDBSCAN (`min_cluster_size=5`).
Label each cluster by asking Claude for a 6-word Spanish label from its top
10 pains. Support `config/merges.yaml` where I manually map cluster ids to
merge after review. Noise cluster is kept but excluded from scoring.

## Scoring (`score.py`) — per cluster, 0–100
- frequency: log-scaled distinct authors (not messages)
- spend: share of pains with non-null `spend_signal` × 30
- b2b: share of `who` classified as business × 20
- reachability: manual field in `config/reachability.yaml` (0–10) — "can I
  name a channel with 100+ of these people?" Default 0 → cluster capped at 20.
- competition: manual field 0–10 (10 = strong incumbents); subtract.
Print the formula in the report so I can tune it. Weights in config.

## Report (`report.py`)
`data/reports/YYYY-MM-DD.md`: top 10 clusters with score breakdown, 3
representative quotes (with source URLs), `who` distribution, `current_tool`
distribution, and a blank "Hypothesis:" line per cluster for me to fill in
with: "[Who] pays [$X] for [outcome] because [current alternative] fails at
[specific thing]."

## CLI
```
niche-miner collect --niche contadores_pymes [--source google_play] [--limit 50]
niche-miner extract [--niche ...]
niche-miner cluster [--niche ...]
niche-miner score
niche-miner report
niche-miner run-all --niche contadores_pymes
```

## Acceptance criteria
1. `uv run niche-miner run-all --niche <tag> --limit 50` completes end-to-end
   on a fresh clone with only `ANTHROPIC_API_KEY` set (Apify/YouTube skipped
   gracefully) and produces a report.
2. Re-running `collect` does not create duplicates.
3. Tests cover: each parser on saved HTML/JSON fixtures, extraction schema
   validation, scoring with known inputs, dedupe.
4. Total LLM cost for 1,000 signals is logged and printed.
5. No file in the repo exceeds ~300 lines; no UI code anywhere.

## Order of work
1. Scaffold, db, models, CLI stubs, one collector (google_play) end-to-end
   with extraction → report. Show me the report before continuing.
2. Add mercadolibre, youtube, computrabajo.
3. Add Apify collectors.
4. Clustering/scoring refinements only after I've reviewed real output.

Ask me before adding any dependency not listed here or expanding scope.
