"""
instagram_metrics.py
────────────────────
Scrapes Instagram post metrics using Instaloader.
Assembles them into a structured Pandas DataFrame to match the
Unlox engagement_metrics database schema.

Public metrics only — shares and saves are private Instagram data
and cannot be retrieved via scraping.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import instaloader
import pandas as pd

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class ScraperConfig:
    """
    Centralised knobs for the Instagram scraper.

    Attributes:
        request_delay_seconds:  Sleep between requests to reduce ban risk.
        max_retries:            How many times to retry a failed shortcode.
        retry_delay_seconds:    Sleep between retry attempts.
        username:               Optional IG username for authenticated sessions.
        password:               Optional IG password for authenticated sessions.
    """
    request_delay_seconds: float = 2.0
    max_retries: int = 2
    retry_delay_seconds: float = 5.0
    username: Optional[str] = None
    password: Optional[str] = None


# ── Schema ────────────────────────────────────────────────────────────────────

# Canonical column order that mirrors the engagement_metrics DB schema.
SCHEMA_COLUMNS: list[str] = [
    "content_id",
    "platform",
    "views",
    "likes",
    "comments",
    "shares",
    "saves",
    "engagement_rate",
]

PLATFORM = "Instagram"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_loader(config: ScraperConfig) -> instaloader.Instaloader:
    """Initialise and (optionally) authenticate an Instaloader instance."""
    loader = instaloader.Instaloader(
        download_pictures=False,
        download_video_thumbnails=False,
        download_videos=False,
        download_comments=False,
        save_metadata=False,
        quiet=True,             # suppresses instaloader's own stdout chatter
    )
    if config.username and config.password:
        log.info("Logging into Instagram as '%s'.", config.username)
        loader.login(config.username, config.password)
    return loader


def _compute_engagement_rate(likes: int, comments: int, views: int) -> float:
    """
    Engagement rate = (likes + comments) / views, rounded to 6 d.p.

    Returns 0.0 when views is zero (photos, or Reels with no view data)
    to avoid division-by-zero.
    """
    if views > 0:
        return round((likes + comments) / views, 6)
    return 0.0


def _scrape_one(
    loader: instaloader.Instaloader,
    shortcode: str,
    config: ScraperConfig,
) -> Optional[dict]:
    """
    Fetch metrics for a single shortcode, retrying up to config.max_retries
    times on transient errors.

    Returns a metric dict on success, or None if all attempts fail.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(1, config.max_retries + 2):   # +2 → first try + retries
        try:
            post = instaloader.Post.from_shortcode(loader.context, shortcode)

            likes    = post.likes
            comments = post.comments
            views    = (
                post.video_view_count
                if post.is_video and post.video_view_count is not None
                else 0
            )

            return {
                "content_id":      shortcode,
                "platform":        PLATFORM,
                "views":           views,
                "likes":           likes,
                "comments":        comments,
                "shares":          0,       # private — unavailable via scraping
                "saves":           0,       # private — unavailable via scraping
                "engagement_rate": _compute_engagement_rate(likes, comments, views),
            }

        except instaloader.exceptions.ProfileNotExistsException:
            # Permanent error — no point retrying.
            log.error("Shortcode '%s' does not exist or is private.", shortcode)
            return None

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt <= config.max_retries:
                log.warning(
                    "Attempt %d/%d failed for shortcode '%s': %s — retrying in %.1fs.",
                    attempt,
                    config.max_retries + 1,
                    shortcode,
                    exc,
                    config.retry_delay_seconds,
                )
                time.sleep(config.retry_delay_seconds)

    log.error(
        "All %d attempts failed for shortcode '%s'. Last error: %s",
        config.max_retries + 1,
        shortcode,
        last_exc,
    )
    return None


# ── Extraction ────────────────────────────────────────────────────────────────

def fetch_ig_metrics(
    shortcodes: list[str],
    config: Optional[ScraperConfig] = None,
) -> list[dict]:
    """
    Scrape public metrics for a list of Instagram shortcodes.

    Args:
        shortcodes: Instagram post shortcodes (duplicates are ignored).
        config:     Scraper settings; defaults to ScraperConfig() if omitted.

    Returns:
        List of metric dicts, one per successfully scraped shortcode.
    """
    if not shortcodes:
        log.warning("fetch_ig_metrics called with an empty shortcode list.")
        return []

    cfg = config or ScraperConfig()
    loader = _build_loader(cfg)

    # dict.fromkeys preserves insertion order while deduplicating
    unique_codes = list(dict.fromkeys(shortcodes))
    log.info("Scraping %d unique Instagram shortcode(s).", len(unique_codes))

    results: list[dict] = []

    for shortcode in unique_codes:
        log.info("  → shortcode: %s", shortcode)
        record = _scrape_one(loader, shortcode, cfg)

        if record is not None:
            results.append(record)

        # Throttle every request (success or failure) to stay under radar.
        time.sleep(cfg.request_delay_seconds)

    log.info(
        "Scraping complete. %d/%d shortcodes succeeded.",
        len(results),
        len(unique_codes),
    )
    return results


# ── Dataset Builder ───────────────────────────────────────────────────────────

def build_ig_performance_dataset(
    shortcodes: list[str],
    config: Optional[ScraperConfig] = None,
) -> pd.DataFrame:
    """
    Fetch Instagram metrics and return a DataFrame aligned to the
    engagement_metrics DB schema, ready for mysql_export.py.

    Args:
        shortcodes: Instagram post shortcodes to process.
        config:     Optional scraper configuration.

    Returns:
        DataFrame with columns matching SCHEMA_COLUMNS, or an empty
        DataFrame (with the correct columns) when no data is available.
    """
    empty = pd.DataFrame(columns=SCHEMA_COLUMNS)

    if not shortcodes:
        log.warning("No Instagram shortcodes supplied.")
        return empty

    records = fetch_ig_metrics(shortcodes, config=config)

    if not records:
        log.warning("No metrics were retrieved — returning empty DataFrame.")
        return empty

    df = (
        pd.DataFrame(records)
          .reindex(columns=SCHEMA_COLUMNS)   # enforce column order & completeness
    )

    # Enforce sensible dtypes
    int_cols   = ["views", "likes", "comments", "shares", "saves"]
    float_cols = ["engagement_rate"]
    df[int_cols]   = df[int_cols].fillna(0).astype(int)
    df[float_cols] = df[float_cols].fillna(0.0).astype(float)

    log.info("Instagram dataset built: %d row(s), %d column(s).", *df.shape)
    return df