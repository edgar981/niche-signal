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

Or step by step: `collect --niche X [--source google_play|computrabajo|youtube|apify_gmaps|apify_tiktok|apify_facebook] [--limit N]`, `extract`,
`cluster`, `review [--refresh]`, `score`, `report`. Reports land in `data/reports/YYYY-MM-DD.md`;
automated reviews in `data/review/` (`<cluster_id>.json`, `niche_<tag>.json`); DB backups in
`data/backups/`; everything else in `data/signals.db`. Re-running `collect` only adds new signals; `extract` only processes
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
- `apify_gmaps`, `apify_tiktok`, `apify_facebook` — Google Maps reviews for the `gmaps_queries`,
  TikTok comments on videos found through the `tiktok_hashtags`, and posts plus top comments
  from the public `facebook_groups`. All three run Apify actors and need `APIFY_TOKEN` in
  `.env`; without it they're skipped.
  - Actors and inputs are set under `collectors.apify` in `config/settings.yaml`. A replacement
    actor needs the same output shape, or a parser change.
  - Apify charges per item returned, including items already in the DB. Every run has a cost
    cap and an item cap, and `--limit` lowers the item cap. Each run's cost is logged. The
    defaults cost about $0.15 per Maps query, $0.30 per hashtag and $0.15 per group at
    free-plan rates.
  - Maps reviews are kept at 3 stars or fewer. Reviewer personal data is never fetched.
  - TikTok comments by the video's author are left out.
  - Authors are only hashed, and texts are scrubbed. The place name, video caption, group name
    or commented post is added as the first line after filtering.

## Review loop

`review` (part of `run-all`) fills in what you don't override, cached 30 days:

- **competition** (0–10): Google results for "<theme label> Colombia / app / software" (Apify
  `apify/google-search-scraper`, about $0.0045 per query), a Play Store search (country co) and
  the cluster's current tools go to Claude. Claude lists incumbents (name, country, exact or
  adjacent fit) and applies the rubric: 0 = nothing specific, 5 = generic tools partly cover
  it, 8–10 = a Colombian incumbent does exactly this.
- **reachability_proxy** (0–10), per niche, by channel *type*: `job_board` (Computrabajo posts
  in the last 30 days for the niche's own `computrabajo_queries`, in Bogotá, Medellín and Cali),
  `fb_groups` (listed groups; member counts by hand in `facebook_group_members`), `gremio`
  (Google "<niche> asociación Colombia", names extracted by Claude) and `whatsapp_community`
  (by hand). 0 = no type with signal, 3 = one, 5 = two, 7 = three; +1 per type with strong
  volume (≥ 50 posts, ≥ 20k FB members, a gremio with a public member directory). YouTube /
  TikTok creators whose videos drew the comments add at most +2 and never count as a type.
  Capped at 10.
- A cluster's reachability is the niche value × **who_match**: the share of its pains whose
  `who` starts with one of the niche's `buyer_roles` (niches.yaml), an unknown `who` counting
  0.7. Under 3 with no override, the report flags "no visible audience" but still ranks it.

Then:

1. Read the report: a top-10 table (score, old-frequency score, co-only score, documents,
   flags), and per cluster the breakdown, incumbents, audience, quotes and a blank `Hypothesis:`.
2. Override what you disagree with in `config/reachability.yaml`: `reachability` /
   `competition` (null keeps the automated value), `status` (`active`, `not_a_niche`,
   `merged_into:<id>`) and `note`. The report ends with a paste-ready stub for the clusters
   not reviewed yet. `config/merges.yaml` (`target_id: [ids]`) still works too.
3. `uv run niche-miner score && uv run niche-miner report`. No re-clustering needed.

Cluster ids are global: a new `cluster` run creates new ids, so overrides and the review cache
stop matching after re-clustering (`review` runs again for the new ids).

## Config

- `config/niches.yaml` — seeds per niche; each collector reads only its own key.
- `config/settings.yaml` — model, effort, batch size, pricing (cost logging), filters,
  HDBSCAN params, scoring weights, report flags, and `review` (search actor, cache days).

## Tests

```bash
uv run pytest            # offline: parsers on fixtures, schema validation, scoring, dedupe
uv run pytest -m live    # hits the API with tests/fixtures/extraction_cases.json (< 1 cent)
```

Pick the extraction model by running the live suite with
`NICHE_MINER_TEST_MODEL=<model>` and keeping the cheapest one that passes.
