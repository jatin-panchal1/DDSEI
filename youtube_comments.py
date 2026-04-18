"""
youtube_comments.py
────────────────────
Fetches top-level comments from YouTube videos via the Data API v3.
Returns clean plain-text comments ready for downstream NLP pipelines.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import requests

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class CommentFetchConfig:
    """
    Tuning knobs for the YouTube comment fetcher.

    Attributes:
        max_per_video:      Hard cap on comments collected per video.
        order:              'relevance' (top-voted) or 'time' (chronological).
        request_timeout:    Seconds before a single HTTP call is aborted.
        page_delay_seconds: Politeness sleep between paginated requests.
        min_comment_length: Discard comments shorter than this (filters noise).
    """
    max_per_video:      int   = 100
    order:              str   = "relevance"
    request_timeout:    int   = 10
    page_delay_seconds: float = 0.25
    min_comment_length: int   = 3


# ── YouTube API constants ─────────────────────────────────────────────────────

_COMMENT_THREADS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
_PAGE_SIZE           = 100      # maximum the API allows per page
_HTTP_FORBIDDEN      = 403
_HTTP_NOT_FOUND      = 404


# ── Low-level pagination helper ───────────────────────────────────────────────

def _iter_comment_pages(
    video_id: str,
    api_key:  str,
    cfg:      CommentFetchConfig,
) -> Iterator[list[dict]]:
    """
    Yields raw API 'items' lists, one list per page, until the video's
    comment quota is exhausted or cfg.max_per_video is reached.

    Raises:
        requests.HTTPError: for non-recoverable API errors (re-raised to caller).
    """
    collected = 0
    page_token: Optional[str] = None

    while collected < cfg.max_per_video:
        batch_size = min(_PAGE_SIZE, cfg.max_per_video - collected)

        params: dict = {
            "part":       "snippet",
            "videoId":    video_id,
            "maxResults": batch_size,
            "textFormat": "plainText",
            "order":      cfg.order,
            "key":        api_key,
        }
        if page_token:
            params["pageToken"] = page_token

        response = requests.get(
            _COMMENT_THREADS_URL,
            params=params,
            timeout=cfg.request_timeout,
        )

        # ── Soft failures that should not abort the whole run ──────────────
        if response.status_code in (_HTTP_FORBIDDEN, _HTTP_NOT_FOUND):
            reason = (
                "Comments disabled or access restricted"
                if response.status_code == _HTTP_FORBIDDEN
                else "Video not found"
            )
            log.warning("%s for video '%s' (HTTP %d).", reason, video_id, response.status_code)
            return

        response.raise_for_status()     # surface unexpected errors to the caller
        data = response.json()
        items: list[dict] = data.get("items", [])

        if not items:
            break

        yield items
        collected += len(items)

        page_token = data.get("nextPageToken")
        if not page_token:
            break

        if cfg.page_delay_seconds > 0:
            time.sleep(cfg.page_delay_seconds)


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_text(item: dict) -> Optional[str]:
    """
    Safely drill into the YouTube commentThread item structure and return
    the original comment text, or None if the key path is missing.
    """
    try:
        return (
            item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
        )
    except (KeyError, TypeError):
        log.debug("Unexpected commentThread shape; skipping item.")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_youtube_comments(
    video_ids: list[str],
    api_key:   str,
    config:    Optional[CommentFetchConfig] = None,
) -> list[str]:
    """
    Fetch top-level comments for one or more YouTube videos.

    Uses 'textOriginal' (plain text, no HTML entities) so comments
    are clean for NLP consumption without any pre-processing step.

    Args:
        video_ids: YouTube video IDs to process (duplicates are ignored).
        api_key:   YouTube Data API v3 key.
        config:    Optional fetch settings; falls back to CommentFetchConfig().

    Returns:
        Deduplicated list of comment strings across all requested videos.
    """
    if not video_ids:
        log.warning("fetch_youtube_comments called with an empty video list.")
        return []

    if not api_key:
        raise ValueError("A YouTube Data API key must be provided.")

    cfg = config or CommentFetchConfig()
    unique_ids = list(dict.fromkeys(video_ids))     # preserve order, drop dupes
    seen:         set[str]  = set()
    all_comments: list[str] = []

    log.info(
        "Fetching comments for %d unique video(s) (max %d each, order='%s').",
        len(unique_ids), cfg.max_per_video, cfg.order,
    )

    for video_id in unique_ids:
        video_comments: list[str] = []

        try:
            for page_items in _iter_comment_pages(video_id, api_key, cfg):
                for item in page_items:
                    text = _extract_text(item)
                    if text is None:
                        continue
                    text = text.strip()
                    if len(text) < cfg.min_comment_length:
                        continue
                    if text in seen:        # cross-video deduplication
                        continue
                    seen.add(text)
                    video_comments.append(text)

        except requests.HTTPError as exc:
            log.error(
                "HTTP error fetching comments for video '%s': %s", video_id, exc
            )
            continue
        except requests.RequestException as exc:
            log.error(
                "Network error fetching comments for video '%s': %s", video_id, exc
            )
            continue

        log.info(
            "  → video '%s': collected %d comment(s).", video_id, len(video_comments)
        )
        all_comments.extend(video_comments)

    log.info("Done. Total comments collected: %d.", len(all_comments))
    return all_comments