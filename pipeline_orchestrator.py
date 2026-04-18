"""
pipeline_orchestrator.py
────────────────────────
Master script that glues Data Extraction, NLP Sentiment Analysis,
and MySQL loading into a single automated daily run.
"""

import logging
import time
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from youtube_metrics import build_performance_dataset, load_config as load_yt_config 
from youtube_comments import fetch_youtube_comments, CommentFetchConfig
from instagram_metrics import build_ig_performance_dataset, ScraperConfig
from instagram_comments import fetch_ig_comments, IGCommentConfig
from sentiment_engine import AudienceSentimentAnalyzer
from mysql_export import export_to_database, load_db_config
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Centralised tunables for the daily pipeline."""
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    batch_size: int = 50
    dry_run: bool = False          # When True: extract + analyse but skip DB writes


@dataclass
class PipelineResult:
    """Machine-readable summary returned to callers / schedulers."""
    success: bool
    videos_processed: int = 0
    metric_rows_upserted: int = 0
    comments_analysed: int = 0
    relatable_rate: float = 0.0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _with_retry(fn, *, label: str, cfg: PipelineConfig):
    """Call *fn()* up to cfg.max_retries times with exponential back-off."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            wait = cfg.retry_backoff_seconds * (2 ** (attempt - 1))
            log.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                label, attempt, cfg.max_retries, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"{label} failed after {cfg.max_retries} attempts") from last_exc

# ── Pipeline steps ────────────────────────────────────────────────────────────

def _step_extract_metrics(yt_ids: list[str], ig_ids: list[str], yt_config, ig_config, cfg: PipelineConfig):
    """Step 1 – Pull engagement metrics from YouTube AND Instagram."""
    log.info("Step 1/3 │ Extracting metrics across platforms…")
    t0 = time.perf_counter()

    dataframes = []

    # Pull YouTube
    if yt_ids:
        yt_df = _with_retry(
            lambda: build_performance_dataset(yt_ids, config=yt_config),
            label="YouTube metrics extraction", cfg=cfg
        )
        if not yt_df.empty: dataframes.append(yt_df)

    # Pull Instagram
    if ig_ids:
        ig_df = _with_retry(
            lambda: build_ig_performance_dataset(ig_ids, config=ig_config),
            label="Instagram metrics extraction", cfg=cfg
        )
        if not ig_df.empty: dataframes.append(ig_df)

    if not dataframes:
        raise ValueError("No metrics retrieved from any platform — aborting pipeline.")

    # Combine both dataframes vertically
    combined_df = pd.concat(dataframes, ignore_index=True)

    log.info("Step 1/3 │ ✓ Retrieved %d combined rows (%.2fs)", len(combined_df), time.perf_counter() - t0)
    return combined_df


def _step_load_metrics(metrics_df, db_config, cfg: PipelineConfig) -> int:
    """Step 2 – Upsert engagement metrics into MySQL."""
    log.info("Step 2/3 │ Upserting %d metric rows to MySQL…", len(metrics_df))
    t0 = time.perf_counter()

    if cfg.dry_run:
        log.info("Step 2/3 │ [DRY RUN] Skipping database write.")
        return 0

    rows = _with_retry(
        lambda: export_to_database(metrics_df, cfg=db_config),
        label="MySQL upsert",
        cfg=cfg,
    )

    log.info("Step 2/3 │ ✓ Upserted %d records  (%.2fs)", rows, time.perf_counter() - t0)
    return rows


def _step_nlp_sentiment(
    yt_ids: list[str],
    ig_ids: list[str],
    nlp_engine: AudienceSentimentAnalyzer,
    yt_config, 
    db_config,            # <-- NEW: Pass the database config here
    cfg: PipelineConfig,
) -> dict:
    """Step 3 – Fetch real comments from YT & IG, run sentiment analysis, and save to DB."""
    log.info("Step 3/3 │ Running NLP sentiment analysis on omni-channel comments…")
    t0 = time.perf_counter()

    combined_comments = []

    if yt_ids:
        combined_comments.extend(fetch_youtube_comments(yt_ids, api_key=yt_config.api_key))
    if ig_ids:
        combined_comments.extend(fetch_ig_comments(ig_ids))

    if not combined_comments:
        log.warning("Step 3/3 │ No comments retrieved from any platform — skipping NLP.")
        return {}

    primary_id = yt_ids[0] if yt_ids else ig_ids[0]
    sentiment_df = nlp_engine.process_batch(combined_comments, content_id=primary_id)
    
    # ── NEW: Export the sentiment data to MySQL using Pandas ──
    if not cfg.dry_run and not sentiment_df.empty:
        log.info("Step 3/3 │ Appending %d sentiment rows to MySQL...", len(sentiment_df))
        
        # FIX: MySQL cannot store Python lists. Convert 'matched_triggers' to a JSON string.
        if 'matched_triggers' in sentiment_df.columns:
            import json
            # Create a copy to avoid SettingWithCopyWarning
            sentiment_df = sentiment_df.copy()
            sentiment_df['matched_triggers'] = sentiment_df['matched_triggers'].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )
        
        # Create a direct engine using the URL from your config
        from sqlalchemy import create_engine
        engine = create_engine(db_config.url)
        
        # Safely append the AI results to your new table
        sentiment_df.to_sql("comment_sentiment", con=engine, if_exists="append", index=False)
        
        # Clean up the connection
        engine.dispose()
    # ─────────────────────────────────────────────
    stats = nlp_engine.aggregate_stats(sentiment_df)

    log.info(
        "Step 3/3 │ ✓ Analysed & exported %d total comment(s)  Relatable Rate: %.1f%%  (%.2fs)",
        len(combined_comments),
        stats.get("relatable_rate", 0.0),
        time.perf_counter() - t0,
    )

    return {"comments_analysed": len(combined_comments), **stats}


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_daily_pipeline(
    yt_ids: list[str],
    ig_ids: list[str],
    cfg: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """
    Run the full Unlox daily pipeline for both YouTube and Instagram.
    """
    total_media = len(yt_ids) + len(ig_ids)
    if total_media == 0:
        log.error("No media IDs supplied — nothing to do.")
        return PipelineResult(success=False, errors=["No media IDs supplied."])

    cfg = cfg or PipelineConfig()
    result = PipelineResult(success=False, videos_processed=total_media)
    wall_start = time.perf_counter()

    log.info("═" * 60)
    log.info("Unlox Omni-Channel Pipeline — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("YT Videos: %s | IG Posts: %s | dry_run=%s", yt_ids, ig_ids, cfg.dry_run)
    log.info("═" * 60)

    try:
        yt_config = load_yt_config()
        ig_config = ScraperConfig()  # <-- Updated: Using the new ScraperConfig
        db_config = load_db_config()
        nlp_engine = AudienceSentimentAnalyzer()
    except Exception as exc:
        log.exception("Initialisation failed: %s", exc)
        return PipelineResult(success=False, errors=[f"Init error: {exc}"])

    # Step 1 — Extract (Now passing both lists and configs)
    try:
        metrics_df = _step_extract_metrics(yt_ids, ig_ids, yt_config, ig_config, cfg)
    except Exception as exc:
        log.exception("Step 1 failed: %s", exc)
        result.errors.append(f"Extraction: {exc}")
        return result

    # Step 2 — Load
    try:
        result.metric_rows_upserted = _step_load_metrics(metrics_df, db_config, cfg)
    except Exception as exc:
        log.exception("Step 2 failed: %s", exc)
        result.errors.append(f"DB load: {exc}")

    # Step 3 — NLP Sentiment (Running on YouTube only for now)
    # Step 3 — NLP Sentiment (Running on YouTube only for now)
    # Step 3 — NLP Sentiment
    try:
        if yt_ids or ig_ids:
            # Added db_config here!
            nlp_stats = _step_nlp_sentiment(yt_ids, ig_ids, nlp_engine, yt_config, db_config, cfg)
            result.comments_analysed = nlp_stats.get("comments_analysed", 0)
            result.relatable_rate = nlp_stats.get("relatable_rate", 0.0)
    except Exception as exc:
        log.exception("Step 3 failed: %s", exc)
        result.errors.append(f"NLP: {exc}")

    result.success = not result.errors
    result.duration_seconds = round(time.perf_counter() - wall_start, 2)

    status = "✓ COMPLETE" if result.success else "⚠ COMPLETE WITH ERRORS"
    log.info("═" * 60)
    log.info(
        "Unlox Pipeline %s — %ds  │  %d metric rows  │  %d comments  │  %.1f%% relatable",
        status, result.duration_seconds, result.metric_rows_upserted, 
        result.comments_analysed, result.relatable_rate,
    )
    if result.errors:
        for err in result.errors:
            log.warning("  ✗ %s", err)
    log.info("═" * 60)

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

# ── Entry point ───────────────────────────────────────────────────────────────

def extract_youtube_id(url: str) -> str:
    """Helper function to extract the 11-character ID from any YouTube link."""
    # If you just pasted the 11-character ID, return it as-is
    if len(url) == 11 and "youtube" not in url and "youtu.be" not in url:
        return url
        
    # If it's a standard desktop link
    if "v=" in url:
        return url.split("v=")[1][:11]
        
    # If it's a mobile share link
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1][:11]
        
    log.warning(f"Could not extract ID from link: {url}")
    return ""
import re

def extract_ig_shortcode(url: str) -> str:
    """Helper function to extract the shortcode from any Instagram link."""
    # Clean up accidental spaces
    url = url.strip()
    
    # If you just pasted the raw shortcode, return it as-is
    if "instagram.com" not in url:
        return url
        
    # Look for the shortcode right after /p/ or /reel/
    match = re.search(r"/(?:p|reel)/([^/?#]+)", url)
    if match:
        return match.group(1)
        
    log.warning(f"Could not extract IG shortcode from link: {url}")
    return ""


if __name__ == "__main__":
    
    # 👇 PASTE YOUR YOUTUBE LINKS HERE 👇
    YOUTUBE_LINKS = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",           # Standard Link
        "https://youtu.be/jNQXAC9IVRw?si=some_tracking_code",    # Mobile Link
        # "Paste your 3rd link here...",
    ]
    
    # 👇 PASTE YOUR INSTAGRAM SHORTCODES HERE 👇
    TARGET_IG_POSTS = [
        # "DXPfCfQj3Up", 
    ]

    # Automatically convert your messy links into clean IDs
    TARGET_YT_VIDEOS = [
        extract_youtube_id(link) for link in YOUTUBE_LINKS if extract_youtube_id(link)
    ]

    pipeline_cfg = PipelineConfig(
        max_retries=3,
        retry_backoff_seconds=2.0,
        dry_run=False,
    )

    outcome = run_daily_pipeline(TARGET_YT_VIDEOS, TARGET_IG_POSTS, cfg=pipeline_cfg)

    if not outcome.success:
        raise SystemExit(1)