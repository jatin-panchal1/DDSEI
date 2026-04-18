"""
instagram_comments.py
──────────────────────
Scrapes top-level comments from Instagram posts using Instaloader.
Returns plain-text comments for downstream NLP pipelines.

Limitations (Instagram platform constraints, not code limitations):
  • Requires an authenticated session for reliable comment access.
  • Comment volume is heavily rate-limited — keep max_per_post low.
  • Replies to comments are not collected (top-level only).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import instaloader

log = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class IGCommentConfig:
    """
    Settings for the Instagram comment scraper.

    Attributes:
        max_per_post:           Hard cap on comments collected per post.
                                Keep low (≤ 50) to avoid IP / rate-limit bans.
        request_delay_seconds:  Sleep between posts. Increase if you see blocks.
        min_comment_length:     Discard comments shorter than this (emoji-only
                                spam, single characters, etc.).
        username:               Instagram username for authenticated sessions.
                                Strongly recommended — unauthenticated scraping
                                is aggressively rate-limited by Meta.
        password:               Corresponding Instagram password.
        deduplicate:            When True, identical comment texts are discarded
                                across the entire batch (not just per-post).
    """
    max_per_post:           int   = 40
    request_delay_seconds:  float = 3.0
    min_comment_length:     int   = 3
    username:               Optional[str] = None
    password:               Optional[str] = None
    deduplicate:            bool  = True


# ── Instaloader factory ───────────────────────────────────────────────────────

def _build_loader(cfg: IGCommentConfig) -> instaloader.Instaloader:
    """
    Construct and (optionally) authenticate an Instaloader instance.

    An authenticated session is strongly recommended: unauthenticated
    requests hit Meta's rate limits almost immediately for comment scraping.
    """
    loader = instaloader.Instaloader(
        download_pictures=False,
        download_video_thumbnails=False,
        download_videos=False,
        download_comments=False,    # we iterate comments manually
        save_metadata=False,
        quiet=True,
    )

    if cfg.username and cfg.password:
        log.info("Logging into Instagram as '%s'.", cfg.username)
        try:
            loader.login(cfg.username, cfg.password)
        except instaloader.exceptions.BadCredentialsException:
            log.error("Instagram login failed — bad credentials for '%s'.", cfg.username)
            raise
        except instaloader.exceptions.TwoFactorAuthRequiredException:
            log.error(
                "Instagram account '%s' requires two-factor authentication. "
                "Use a session file instead of username/password.",
                cfg.username,
            )
            raise
    else:
        log.warning(
            "No Instagram credentials provided. Unauthenticated scraping is "
            "heavily rate-limited and may fail immediately."
        )

    return loader


# ── Per-post comment scraping ─────────────────────────────────────────────────

def _scrape_post_comments(
    loader:    instaloader.Instaloader,
    shortcode: str,
    cfg:       IGCommentConfig,
    seen:      set[str],
) -> list[str]:
    """
    Collect up to cfg.max_per_post comments from a single Instagram post.

    Args:
        loader:    Authenticated (or anonymous) Instaloader instance.
        shortcode: Instagram post shortcode.
        cfg:       Scraper configuration.
        seen:      Shared set of already-collected texts for cross-post
                   deduplication. Updated in-place when cfg.deduplicate is True.

    Returns:
        List of clean comment strings for this post (may be empty).
    """
    post = instaloader.Post.from_shortcode(loader.context, shortcode)
    collected: list[str] = []

    for comment in post.get_comments():
        if not comment.text:
            continue

        text = comment.text.strip()

        if len(text) < cfg.min_comment_length:
            continue

        if cfg.deduplicate:
            if text in seen:
                continue
            seen.add(text)

        collected.append(text)

        if len(collected) >= cfg.max_per_post:
            log.debug(
                "Reached max_per_post cap (%d) for shortcode '%s'.",
                cfg.max_per_post, shortcode,
            )
            break

    return collected


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_ig_comments(
    shortcodes: list[str],
    config:     Optional[IGCommentConfig] = None,
) -> list[str]:
    """
    Scrape top-level comments from a list of Instagram posts.

    Args:
        shortcodes: Instagram post shortcodes (duplicates are ignored).
        config:     Optional scraper settings; defaults to IGCommentConfig().

    Returns:
        Flat list of plain-text comment strings across all posts,
        ordered by post then by comment as returned by Instagram.
        Empty list if nothing could be collected.
    """
    if not shortcodes:
        log.warning("fetch_ig_comments called with an empty shortcode list.")
        return []

    cfg          = config or IGCommentConfig()
    unique_codes = list(dict.fromkeys(shortcodes))
    seen:         set[str]  = set()
    all_comments: list[str] = []

    log.info(
        "Fetching comments for %d unique Instagram post(s) "
        "(max %d each, dedup=%s).",
        len(unique_codes), cfg.max_per_post, cfg.deduplicate,
    )

    loader = _build_loader(cfg)

    for shortcode in unique_codes:
        post_comments: list[str] = []

        try:
            post_comments = _scrape_post_comments(loader, shortcode, cfg, seen)

        except instaloader.exceptions.ConnectionException as exc:
            # Meta's rate-limit / IP block — log and keep going
            log.warning(
                "Instagram blocked the request for shortcode '%s' "
                "(rate limit or IP block): %s",
                shortcode, exc,
            )

        except instaloader.exceptions.ProfileNotExistsException:
            log.warning(
                "Shortcode '%s' does not exist or is set to private.", shortcode
            )

        except instaloader.exceptions.LoginRequiredException:
            log.error(
                "Instagram requires a login to read comments for '%s'. "
                "Provide credentials via IGCommentConfig(username=…, password=…).",
                shortcode,
            )

        except Exception as exc:  # noqa: BLE001
            log.error(
                "Unexpected error fetching comments for shortcode '%s': %s",
                shortcode, exc,
            )

        finally:
            # Always log + sleep, even when a post fails, to avoid hammering
            # the API on a burst of errors.
            log.info(
                "  → shortcode '%s': collected %d comment(s).",
                shortcode, len(post_comments),
            )
            all_comments.extend(post_comments)
            time.sleep(cfg.request_delay_seconds)

    log.info("Done. Total Instagram comments collected: %d.", len(all_comments))
    return all_comments