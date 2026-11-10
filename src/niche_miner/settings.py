"""Project paths and YAML configuration. Every key has a default, so config files stay short."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class LLMSettings(BaseModel):
    extract_model: str = "claude-haiku-5-5"
    label_model: str = "claude-haiku-5-5"
    effort: str | None = "medium"  # output_config.effort; None for models without effort support
    batch_size: int = Field(20, ge=1)
    max_text_chars: int = 4000
    max_tokens: int = 16000
    # USD per million tokens — used only for cost logging.
    pricing: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {"claude-haiku-5-5": {"input": 0.10, "output": 0.50}}
    )


class FilterSettings(BaseModel):
    min_chars: int = 40
    languages: list[str] = Field(default_factory=lambda: ["es"])


class GooglePlaySettings(BaseModel):
    country: str = "co"
    lang: str = "es"
    max_score: int = Field(2, ge=1, le=5)
    max_reviews_per_app: int = 300
    page_size: int = 100
    min_interval_s: float = 1.0


class ComputrabajoSettings(BaseModel):
    base_url: str = "https://co.computrabajo.com"
    user_agent: str = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36")
    min_interval_s: float = Field(3.0, ge=2.0)  # SPEC: polite delays of at least 2s
    max_pages_per_query: int = Field(2, ge=1)  # search pages hold 20 offers each
    max_offers_per_query: int = Field(30, ge=1)  # offer pages fetched per query per run
    timeout_s: float = 20.0


class YouTubeSettings(BaseModel):
    api_url: str = "https://www.googleapis.com/youtube/v3"
    max_pages_per_video: int = Field(5, ge=1)  # 100 comment threads per page, 1 quota unit each
    order: Literal["time", "relevance"] = "time"
    include_replies: bool = True  # the few replies the API returns inside each thread
    min_interval_s: float = 0.5
    timeout_s: float = 20.0


class ApifyActor(BaseModel):
    """One actor run. Actor ids and inputs live in config/settings.yaml (SPEC), not here."""
    actor: str = ""  # e.g. "compass/crawler-google-places"; empty = the source is skipped
    input: dict[str, Any] = Field(default_factory=dict)  # "{target}"-style placeholders
    max_items: int | None = Field(None, ge=1)  # Apify stops charging per-item past this
    max_charge_usd: float | None = Field(None, gt=0)  # cost cap for one run; None = apify default


class ApifyGmapsSettings(BaseModel):
    run: ApifyActor = Field(default_factory=ApifyActor)
    max_stars: int = Field(3, ge=1, le=5)  # keep reviews with at most this many stars


class ApifyTikTokSettings(BaseModel):
    videos: ApifyActor = Field(default_factory=ApifyActor)  # hashtag -> videos
    comments: ApifyActor = Field(default_factory=ApifyActor)  # video URLs -> comments


class ApifyFacebookSettings(BaseModel):
    run: ApifyActor = Field(default_factory=ApifyActor)
    include_comments: bool = True  # the top comments the actor returns with each post


class ApifySettings(BaseModel):
    max_charge_usd: float = Field(0.25, gt=0)  # default cost cap per actor run
    run_timeout_s: int = Field(900, ge=60)
    gmaps: ApifyGmapsSettings = Field(default_factory=ApifyGmapsSettings)
    tiktok: ApifyTikTokSettings = Field(default_factory=ApifyTikTokSettings)
    facebook: ApifyFacebookSettings = Field(default_factory=ApifyFacebookSettings)


class CollectorSettings(BaseModel):
    retries: int = 3
    backoff_s: float = 2.0
    google_play: GooglePlaySettings = Field(default_factory=GooglePlaySettings)
    computrabajo: ComputrabajoSettings = Field(default_factory=ComputrabajoSettings)
    youtube: YouTubeSettings = Field(default_factory=YouTubeSettings)
    apify: ApifySettings = Field(default_factory=ApifySettings)


class ClusterSettings(BaseModel):
    method: Literal["auto", "hdbscan", "llm"] = "auto"
    llm_max_pains: int = 1500  # auto: llm below this many pains per niche, hdbscan otherwise
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    n_components: int | None = 10  # PCA dimensions before HDBSCAN; None = no reduction
    min_cluster_size: int = Field(5, ge=2)
    min_samples: int | None = 2
    noise_warning: float = 0.6  # warn when a larger share of pains ends up as noise
    label_examples: int = 10
    representatives: int = 3
    llm_themes: int = 15  # themes requested by the llm method
    llm_batch_size: int = Field(50, ge=1)  # pains per theme-assignment request
    llm_theme_effort: str | None = "medium"  # effort for the single theme-proposal call


class ScoreWeights(BaseModel):
    frequency: float = 30
    frequency_saturation: int = Field(50, ge=1)
    spend: float = 30
    b2b: float = 20
    reachability: float = 20
    competition: float = 20
    b2b_min_known_share: float = Field(0.3, ge=0, le=1)  # below this, b2b uses the prior
    no_audience_below: float = 3  # flag clusters whose automated reachability is under this
    unknown_who_weight: float = Field(0.7, ge=0, le=1)  # who_match weight of an unknown `who`
    # Evidence multiplier per pain type; unlisted types weigh 1.
    pain_type_weights: dict[str, float] = Field(
        default_factory=lambda: {"incumbent_quality": 0.3})


class ReportSettings(BaseModel):
    top_n: int = 10
    single_url_flag: float = 0.5  # mark clusters where one document holds more of the pains
    rank_change_flag: int = 2  # mark clusters whose co-only rank moves by more than this


class ReviewSettings(BaseModel):
    """`niche-miner review`: automated competition and reachability per cluster."""
    cache_days: int = Field(30, ge=0)  # data/review/<cluster_id>.json is reused this long
    model: str | None = None  # None = llm.label_model
    query_suffixes: list[str] = Field(default_factory=lambda: ["Colombia", "app", "software"])
    search: ApifyActor = Field(default_factory=ApifyActor)  # Google results; "{queries}"
    results_per_query: int = Field(10, ge=1)
    play_hits: int = Field(10, ge=1, le=30)  # Play Store search results (country co)
    pains_shown: int = Field(8, ge=1)  # representative pains sent with the search results
    # Niche reachability (audience.py): Computrabajo city slugs, gremio search, strong volumes.
    cities: list[str] = Field(default_factory=lambda: ["bogota-dc", "medellin", "cali"])
    gremio_query: str = "{niche} asociación Colombia"
    strong_job_posts: int = 50  # Computrabajo posts in 30 days
    strong_fb_members: int = 20_000  # FB group members combined
    strong_creator_audience: int = 20_000  # creator subscribers combined: +2 instead of +1


class Settings(BaseModel):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    filters: FilterSettings = Field(default_factory=FilterSettings)
    collectors: CollectorSettings = Field(default_factory=CollectorSettings)
    clustering: ClusterSettings = Field(default_factory=ClusterSettings)
    scoring: ScoreWeights = Field(default_factory=ScoreWeights)
    report: ReportSettings = Field(default_factory=ReportSettings)
    review: ReviewSettings = Field(default_factory=ReviewSettings)


def find_root() -> Path:
    """$NICHE_MINER_ROOT, else the nearest directory (cwd or a parent) with config/niches.yaml."""
    if env := os.environ.get("NICHE_MINER_ROOT"):
        return Path(env).expanduser()
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "config" / "niches.yaml").is_file():
            return candidate
    return cwd


class Paths:
    def __init__(self, root: Path | None = None):
        self.root = (root or find_root()).resolve()
        self.config = self.root / "config"
        self.data = self.root / "data"
        self.db = self.data / "signals.db"
        self.reports = self.data / "reports"
        self.review = self.data / "review"
        self.env = self.root / ".env"


def load_yaml(path: Path) -> dict[str, Any]:
    """A YAML mapping, or {} when the file is missing or empty."""
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def load_settings(config_dir: Path) -> Settings:
    return Settings.model_validate(load_yaml(config_dir / "settings.yaml"))


def load_niches(config_dir: Path) -> dict[str, dict[str, Any]]:
    niches: dict[str, dict[str, Any]] = {}
    for entry in load_yaml(config_dir / "niches.yaml").get("niches") or []:
        tag = entry.get("tag") if isinstance(entry, dict) else None
        if not tag:
            raise ValueError("every entry in config/niches.yaml needs a 'tag'")
        niches[tag] = entry
    return niches
