"""
sentiment_engine.py
───────────────────
Processes social media comments to quantify emotional resonance,
classify sentiment, and tag "Relatability" based on weighted
linguistic triggers.

Design decisions
────────────────
- VADER is kept as the scoring backbone; it handles emojis, slang caps,
  and punctuation emphasis out-of-the-box.
- A weighted trigger lexicon replaces the flat list so rare, high-signal
  phrases (e.g. "i thought i was the only one") outrank generic ones
  ("same") without needing a heavy ML model.
- Results are typed dataclasses — no more dict key typos downstream.
- Batch processing returns a DataFrame that slots directly into the
  existing pipeline (youtube_metrics.py → db_export.py).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class SentimentConfig:
    """
    All tunable thresholds in one place.

    Attributes
    ----------
    positive_threshold : float
        Minimum compound score to label a comment Positive.
    negative_threshold : float
        Maximum compound score to label a comment Negative.
    relatable_compound_threshold : float
        Absolute compound score at which a comment is flagged Relatable
        even when no trigger phrase is found (deep emotional charge).
    relatable_trigger_threshold : float
        Minimum weighted trigger score to qualify as Relatable.
    """
    positive_threshold: float           = 0.05
    negative_threshold: float           = -0.05
    relatable_compound_threshold: float = 0.60
    relatable_trigger_threshold: float  = 0.50


# ── Weighted trigger lexicon ───────────────────────────────────────────────────

# Each entry maps a normalised trigger phrase → weight (0.0–1.0).
# Higher weight = stronger signal of felt relatability.
# Organised by category for easy auditing and extension.
DEFAULT_TRIGGER_LEXICON: dict[str, float] = {
    # ── Strong identification ──────────────────────────────────────────────
    "i thought i was the only one":     1.0,
    "i feel heard":                     1.0,
    "are you me":                       0.9,
    "story of my life":                 0.9,
    "you just described my life":       0.9,
    "this is exactly me":               0.85,
    # ── Moderate identification ────────────────────────────────────────────
    "literally me":                     0.80,
    "this is me":                       0.75,
    "felt this":                        0.75,
    "called out":                       0.70,
    "so true":                          0.65,
    "spot on":                          0.65,
    # ── Weak / generic ─────────────────────────────────────────────────────
    "relatable":                        0.55,
    "100%":                             0.50,
    "same":                             0.40,
    "real":                             0.30,
    "facts":                            0.30,
}


# ── Slang / abbreviation normalisation map ─────────────────────────────────────

_SLANG_MAP: dict[str, str] = {
    r"\bfr\b":       "for real",
    r"\bngl\b":      "not gonna lie",
    r"\bfr fr\b":    "for real for real",
    r"\bimo\b":      "in my opinion",
    r"\bthis!!+\b":  "this",
    r"\bsame!+\b":   "same",
    r"\blmao\b":     "laughing",
    r"\blol\b":      "laughing",
    r"\bomg\b":      "oh my god",
}


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class CommentResult:
    """
    Typed result for a single processed comment.

    Attributes
    ----------
    original_text : str
        The raw input text.
    normalised_text : str
        Text after slang expansion and unicode normalisation.
    sentiment_tag : str
        One of: "Positive", "Negative", "Neutral", "Relatable".
    compound_score : float
        VADER compound score in [-1, 1].
    positive_score : float
        VADER pos component.
    neutral_score : float
        VADER neu component.
    negative_score : float
        VADER neg component.
    trigger_score : float
        Weighted relatability trigger score (0.0 if no trigger found).
    matched_triggers : list[str]
        All trigger phrases found in the comment.
    """
    original_text:   str
    normalised_text: str
    sentiment_tag:   str
    compound_score:  float
    positive_score:  float
    neutral_score:   float
    negative_score:  float
    trigger_score:   float          = 0.0
    matched_triggers: list[str]     = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Core analyser ──────────────────────────────────────────────────────────────

class AudienceSentimentAnalyzer:
    """
    Classifies social media comments into four sentiment tags and
    quantifies audience relatability via a weighted trigger lexicon.

    Parameters
    ----------
    config : SentimentConfig, optional
        Scoring thresholds; uses library defaults when not provided.
    trigger_lexicon : dict[str, float], optional
        Override the default weighted trigger phrases.
    """

    def __init__(
        self,
        config: Optional[SentimentConfig] = None,
        trigger_lexicon: Optional[dict[str, float]] = None,
    ) -> None:
        self.config  = config or SentimentConfig()
        self.lexicon = trigger_lexicon or DEFAULT_TRIGGER_LEXICON
        self._vader  = SentimentIntensityAnalyzer()

        # Pre-compile slang regexes once at init for speed
        self._slang_patterns: list[tuple[re.Pattern, str]] = [
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in _SLANG_MAP.items()
        ]

    # ── Text pre-processing ───────────────────────────────────────────────────

    def _normalise(self, text: str) -> str:
        """
        Lightly normalise *text* before scoring.

        Steps:
        1. Unicode NFC normalisation (collapses visually identical chars).
        2. Strip zero-width and control characters.
        3. Expand known slang abbreviations.
        4. Collapse runs of 3+ identical characters → 2
           (e.g. "sooooo" → "soo") so VADER handles them more accurately.
        """
        # NFC unicode normalisation
        text = unicodedata.normalize("NFC", text)

        # Remove zero-width / control characters (keep printable + whitespace)
        text = "".join(
            ch for ch in text
            if unicodedata.category(ch) not in {"Cf", "Cc"} or ch in "\n\t "
        )

        # Expand slang
        for pattern, replacement in self._slang_patterns:
            text = pattern.sub(replacement, text)

        # Collapse character runs: "sooooo" → "soo"
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        return text.strip()

    # ── Trigger matching ──────────────────────────────────────────────────────

    def _score_triggers(self, text_lower: str) -> tuple[float, list[str]]:
        """
        Scan *text_lower* for trigger phrases and return a weighted score.

        The score is the maximum individual trigger weight found
        (rather than a sum) to avoid artificially inflating short comments
        that happen to contain multiple generic phrases like "same" and "facts".

        Returns
        -------
        tuple[float, list[str]]
            (max_weight, list_of_all_matched_triggers)
        """
        matched: list[str] = []
        max_weight = 0.0

        for phrase, weight in self.lexicon.items():
            if phrase in text_lower:
                matched.append(phrase)
                max_weight = max(max_weight, weight)

        return max_weight, matched

    # ── Classification ────────────────────────────────────────────────────────

    def _classify(
        self,
        compound: float,
        trigger_score: float,
        has_triggers: bool,
    ) -> str:
        """
        Map compound score + trigger info to one of four tags.

        Priority order:
          1. Relatable  — strong trigger phrase OR high absolute compound
          2. Positive
          3. Negative
          4. Neutral
        """
        cfg = self.config

        is_relatable = (
            (has_triggers and trigger_score >= cfg.relatable_trigger_threshold)
            or abs(compound) > cfg.relatable_compound_threshold
        )

        if is_relatable:
            return "Relatable"
        if compound >= cfg.positive_threshold:
            return "Positive"
        if compound <= cfg.negative_threshold:
            return "Negative"
        return "Neutral"

    # ── Public interface ──────────────────────────────────────────────────────

    def process_comment(self, comment_text: str) -> CommentResult:
        """
        Analyse a single comment end-to-end.

        Parameters
        ----------
        comment_text : str
            Raw comment string.

        Returns
        -------
        CommentResult
            Fully typed result; call `.to_dict()` for DataFrame-ready output.
        """
        if not comment_text or not comment_text.strip():
            return CommentResult(
                original_text=comment_text or "",
                normalised_text="",
                sentiment_tag="Neutral",
                compound_score=0.0,
                positive_score=0.0,
                neutral_score=1.0,
                negative_score=0.0,
            )

        normalised = self._normalise(comment_text)
        scores     = self._vader.polarity_scores(normalised)
        compound   = scores["compound"]

        trigger_score, matched = self._score_triggers(normalised.lower())
        tag = self._classify(compound, trigger_score, bool(matched))

        return CommentResult(
            original_text=comment_text,
            normalised_text=normalised,
            sentiment_tag=tag,
            compound_score=round(compound, 6),
            positive_score=round(scores["pos"], 6),
            neutral_score=round(scores["neu"], 6),
            negative_score=round(scores["neg"], 6),
            trigger_score=round(trigger_score, 4),
            matched_triggers=matched,
        )

    def process_batch(
        self,
        comments: list[str],
        content_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Analyse a list of comments and return a structured DataFrame.

        Parameters
        ----------
        comments : list[str]
            Raw comment strings.
        content_id : str, optional
            Associates results with a specific piece of content
            (e.g. a YouTube video ID) for downstream joins.

        Returns
        -------
        pd.DataFrame
            One row per comment with all CommentResult fields as columns,
            plus a ``content_id`` column when provided.
        """
        if not comments:
            log.warning("process_batch received an empty comment list.")
            return pd.DataFrame()

        records = []
        for comment in comments:
            result = self.process_comment(comment)
            row = result.to_dict()
            if content_id is not None:
                row["content_id"] = content_id
            records.append(row)

        df = pd.DataFrame(records)

        # Move content_id to the front if present
        if "content_id" in df.columns:
            cols = ["content_id"] + [c for c in df.columns if c != "content_id"]
            df = df[cols]

        log.info(
            "Processed %d comment(s) for content_id=%s | tag distribution: %s",
            len(df),
            content_id or "N/A",
            df["sentiment_tag"].value_counts().to_dict(),
        )
        return df

    def aggregate_stats(self, df: pd.DataFrame) -> dict:
        """
        Summarise a batch DataFrame produced by :meth:`process_batch`.

        Parameters
        ----------
        df : pd.DataFrame
            Output of ``process_batch``.

        Returns
        -------
        dict
            Keys: total_comments, tag_counts, tag_percentages,
            mean_compound, relatable_rate, top_triggers.
        """
        if df.empty:
            return {}

        total = len(df)
        tag_counts = df["sentiment_tag"].value_counts().to_dict()
        tag_pct = {
            tag: round(count / total * 100, 2)
            for tag, count in tag_counts.items()
        }

        # Flatten all matched trigger lists and count occurrences
        all_triggers: list[str] = [
            t for sublist in df["matched_triggers"] for t in sublist
        ]
        trigger_series = pd.Series(all_triggers)
        top_triggers = (
            trigger_series.value_counts().head(5).to_dict()
            if not trigger_series.empty else {}
        )

        return {
            "total_comments":  total,
            "tag_counts":      tag_counts,
            "tag_percentages": tag_pct,
            "mean_compound":   round(df["compound_score"].mean(), 4),
            "relatable_rate":  round(tag_counts.get("Relatable", 0) / total * 100, 2),
            "top_triggers":    top_triggers,
        }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    analyser = AudienceSentimentAnalyzer()

    test_comments = [
        "This is literally me every single morning.",
        "Nice video, good editing.",
        "I thought I was the only one who struggled with this. Thank you.",
        "Algorithm brought me here.",
        "FR FR this hit different, not gonna lie 😭",
        "absolute garbage, waste of my time",
        "sooooo relatable omg!!!",
        "",                           # edge case: empty string
    ]

    VIDEO_ID = "dQw4w9WgXcQ"
    results_df = analyser.process_batch(test_comments, content_id=VIDEO_ID)
    stats      = analyser.aggregate_stats(results_df)

    print("\n─── Sentiment Analysis Results ───")
    display_cols = [
        "original_text", "sentiment_tag",
        "compound_score", "trigger_score", "matched_triggers",
    ]
    print(results_df[display_cols].to_string(index=False))

    print("\n─── Aggregate Stats ───")
    for key, val in stats.items():
        print(f"  {key:<20} {val}")