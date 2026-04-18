"""
youtube_metrics.py
──────────────────
Fetches YouTube video engagement metrics via the YouTube Data API v3,
assembles them into a structured Pandas DataFrame, and provides helpers
for exporting results.

Usage:
    python youtube_metrics.py
    # or import and call build_performance_dataset() directly
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/videos"
MAX_IDS_PER_REQUEST = 50          # YouTube API hard limit per call
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_BASE = 2.0        # seconds; doubles on each retry


# ── Exceptions ────────────────────────────────────────────────────────────────
class YouTubeAPIError(Exception):
    """Raised when the YouTube Data API returns a non-200 response."""


class MissingAPIKeyError(EnvironmentError):
    """Raised when YOUTUBE_API_KEY is absent or empty."""


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    api_key: str
    retries: int = DEFAULT_RETRIES
    backoff_base: float = DEFAULT_BACKOFF_BASE
    timeout: int = 10             # request timeout in seconds


def load_config() -> Config:
    """Load and validate runtime configuration from the environment."""
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        raise MissingAPIKeyError(
            "YOUTUBE_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )
    return Config(api_key=api_key)


# ── API layer ─────────────────────────────────────────────────────────────────
def _request_with_retry(
    url: str,
    params: dict,
    config: Config,
) -> dict:
    """
    GET *url* with *params*, retrying up to config.retries times on
    transient HTTP errors (429, 5xx) using exponential back-off.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(1, config.retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=config.timeout)

            if resp.status_code == 200:
                return resp.json()

            # Retryable server-side or rate-limit errors
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = config.backoff_base ** attempt
                log.warning(
                    "HTTP %s on attempt %d/%d — retrying in %.1fs",
                    resp.status_code, attempt, config.retries, wait,
                )
                time.sleep(wait)
                continue

            # Non-retryable client error
            raise YouTubeAPIError(
                f"YouTube API returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        except requests.RequestException as exc:
            last_exc = exc
            wait = config.backoff_base ** attempt
            log.warning("Request error on attempt %d/%d: %s — retrying in %.1fs",
                        attempt, config.retries, exc, wait)
            time.sleep(wait)

    raise YouTubeAPIError(
        f"All {config.retries} attempts failed."
    ) from last_exc


# ── Metric extraction ─────────────────────────────────────────────────────────
def _parse_video_item(item: dict) -> dict:
    """Convert a single raw API *item* into a flat metrics dictionary."""
    stats = item.get("statistics", {})
    snippet = item.get("snippet", {})

    views    = int(stats.get("viewCount",   0))
    likes    = int(stats.get("likeCount",   0))
    comments = int(stats.get("commentCount", 0))

    # Engagement rate: (likes + comments) / views — guarded against zero-division
    engagement_rate = round((likes + comments) / views, 6) if views else 0.0

    return {
        "content_id":      item["id"],
        "platform":        "YouTube",
        "topic":           snippet.get("title", "Unknown"),
        "channel":         snippet.get("channelTitle", "Unknown"),
        "published_at":    snippet.get("publishedAt", ""),
        "views":           views,
        "likes":           likes,
        "comments":        comments,
        # Shares & saves are not exposed by the public YouTube API
        "shares":          0,
        "saves":           0,
        "engagement_rate": engagement_rate,
    }


def fetch_batch_metrics(video_ids: list[str], config: Config) -> list[dict]:
    """
    Fetch metrics for *video_ids* in batches of up to MAX_IDS_PER_REQUEST.

    Returns a list of metric dictionaries (one per valid video ID).
    Videos not found in the API response are silently skipped.
    """
    results: list[dict] = []

    # Deduplicate while preserving order
    unique_ids = list(dict.fromkeys(video_ids))

    # Chunk into batches the API can handle
    chunks = [
        unique_ids[i : i + MAX_IDS_PER_REQUEST]
        for i in range(0, len(unique_ids), MAX_IDS_PER_REQUEST)
    ]

    for chunk in chunks:
        log.info("Fetching batch of %d video(s): %s", len(chunk), chunk)
        params = {
            "part": "statistics,snippet",
            "id":   ",".join(chunk),
            "key":  config.api_key,
        }

        try:
            data = _request_with_retry(YOUTUBE_API_URL, params, config)
        except YouTubeAPIError as exc:
            log.error("Batch failed — skipping chunk. Reason: %s", exc)
            continue

        items = data.get("items", [])
        if not items:
            log.warning("No items returned for chunk: %s", chunk)
            continue

        for item in items:
            results.append(_parse_video_item(item))

        found_ids = {item["id"] for item in items}
        missing = set(chunk) - found_ids
        if missing:
            log.warning("IDs not found in API response: %s", missing)

    return results


# ── Dataset builder ───────────────────────────────────────────────────────────
def build_performance_dataset(
    video_ids: list[str],
    config: Optional[Config] = None,
) -> pd.DataFrame:
    """
    Build a Pandas DataFrame of YouTube engagement metrics for *video_ids*.

    Parameters
    ----------
    video_ids : list[str]
        One or more YouTube video IDs.
    config : Config, optional
        Runtime config; loaded from the environment when not provided.

    Returns
    -------
    pd.DataFrame
        Columns: content_id, platform, topic, channel, published_at,
                 views, likes, comments, shares, saves, engagement_rate.
        Empty DataFrame if no data could be retrieved.
    """
    if config is None:
        config = load_config()

    if not video_ids:
        log.warning("No video IDs supplied — returning empty DataFrame.")
        return pd.DataFrame()

    metrics = fetch_batch_metrics(video_ids, config)

    if not metrics:
        log.warning("No metrics retrieved.")
        return pd.DataFrame()

    df = pd.DataFrame(metrics)

    # Enforce consistent column types
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    for col in ("views", "likes", "comments", "shares", "saves"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    log.info("Dataset built: %d rows × %d columns", *df.shape)
    return df


# ── Export helpers ────────────────────────────────────────────────────────────
def export_dataset(df: pd.DataFrame, output_dir: str | Path = ".") -> None:
    """Save *df* as both CSV and JSON inside *output_dir*."""
    if df.empty:
        log.warning("DataFrame is empty — nothing exported.")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path  = out / "youtube_metrics.csv"
    json_path = out / "youtube_metrics.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", date_format="iso", indent=2)

    log.info("Exported → %s", csv_path)
    log.info("Exported → %s", json_path)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TEST_VIDEO_IDS = [
        "dQw4w9WgXcQ",   # Rick Astley – Never Gonna Give You Up
        "jNQXAC9IVRw",   # First YouTube video
    ]

    try:
        cfg = load_config()
    except MissingAPIKeyError as e:
        log.critical(e)
        raise SystemExit(1)

    dataset = build_performance_dataset(TEST_VIDEO_IDS, config=cfg)

    print("\n─── Extraction Complete ───")
    print(dataset.to_string(index=False))

    export_dataset(dataset, output_dir="output")

