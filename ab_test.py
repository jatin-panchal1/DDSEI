"""
ab_test.py
──────────
Two-proportion Z-test for comparing hook performance (share rate,
click-through rate, etc.) between a control and a variant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from statsmodels.stats.proportion import proportions_ztest, proportion_confint

log = logging.getLogger(__name__)


# ── Types ─────────────────────────────────────────────────────────────────────

class Alternative(str, Enum):
    """
    Direction of the hypothesis test.

    TWO_SIDED  – detect any difference (use unless you have a strong prior).
    LARGER     – test whether Hook B's rate is *higher* than Hook A's.
    SMALLER    – test whether Hook B's rate is *lower*  than Hook A's.
    """
    TWO_SIDED = "two-sided"
    LARGER    = "larger"
    SMALLER   = "smaller"


@dataclass(frozen=True)
class HookData:
    """Raw observations for one hook variant."""
    label:   str
    views:   int
    actions: int

    def __post_init__(self) -> None:
        if self.views <= 0:
            raise ValueError(f"[{self.label}] views must be > 0, got {self.views}.")
        if self.actions < 0:
            raise ValueError(f"[{self.label}] actions cannot be negative, got {self.actions}.")
        if self.actions > self.views:
            raise ValueError(
                f"[{self.label}] actions ({self.actions}) cannot exceed "
                f"views ({self.views})."
            )

    @property
    def rate(self) -> float:
        return self.actions / self.views


@dataclass(frozen=True)
class ABTestResult:
    """
    Structured result from a single A/B test.

    Raw numeric fields are kept alongside formatted strings so callers
    can aggregate or plot results without re-parsing formatted text.
    """
    # Raw metrics
    rate_a:      float
    rate_b:      float
    uplift_pct:  float          # (rate_b - rate_a) / rate_a * 100
    z_stat:      float
    p_value:     float
    alpha:       float

    # Confidence intervals on each rate  (lower, upper)
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]

    # Test configuration
    alternative: Alternative
    label_a:     str
    label_b:     str

    # Interpretation helpers
    @property
    def is_significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def confidence_level_pct(self) -> float:
        return (1 - self.alpha) * 100

    @property
    def verdict(self) -> str:
        """Plain-English summary of the result."""
        if not self.is_significant:
            return (
                f"No significant difference detected "
                f"(p={self.p_value:.4f} ≥ α={self.alpha}). "
                "Continue collecting data or accept the null hypothesis."
            )
        winner = self.label_b if self.rate_b >= self.rate_a else self.label_a
        direction = "outperforms" if self.rate_b >= self.rate_a else "underperforms"
        return (
            f"{self.label_b} {direction} {self.label_a} "
            f"(uplift {self.uplift_pct:+.2f}%, p={self.p_value:.4f}). "
            f"Result is significant at the {self.confidence_level_pct:.0f}% confidence level. "
            f"Recommend: ship {winner}."
        )

    def summary(self) -> str:
        """Multi-line formatted report."""
        ci_a_pct = f"[{self.ci_a[0]:.2%}, {self.ci_a[1]:.2%}]"
        ci_b_pct = f"[{self.ci_b[0]:.2%}, {self.ci_b[1]:.2%}]"
        return (
            f"\n{'─'*48}\n"
            f"  A/B Test: {self.label_a}  vs  {self.label_b}\n"
            f"{'─'*48}\n"
            f"  {self.label_a:<20} rate : {self.rate_a:.4%}  95% CI {ci_a_pct}\n"
            f"  {self.label_b:<20} rate : {self.rate_b:.4%}  95% CI {ci_b_pct}\n"
            f"  Uplift                    : {self.uplift_pct:+.2f}%\n"
            f"  Z-statistic               : {self.z_stat:.4f}\n"
            f"  P-value                   : {self.p_value:.4f}\n"
            f"  Alpha                     : {self.alpha}\n"
            f"  Hypothesis (alternative)  : {self.alternative.value}\n"
            f"  Significant               : {'✓ YES' if self.is_significant else '✗ NO'}\n"
            f"{'─'*48}\n"
            f"  Verdict: {self.verdict}\n"
            f"{'─'*48}\n"
        )


# ── Core test function ────────────────────────────────────────────────────────

def run_ab_test(
    hook_a: HookData,
    hook_b: HookData,
    alpha: float = 0.05,
    alternative: Alternative = Alternative.TWO_SIDED,
) -> ABTestResult:
    """
    Run a two-proportion Z-test comparing hook_a (control) to hook_b (variant).

    Args:
        hook_a:      Control group observations.
        hook_b:      Variant group observations.
        alpha:       Significance threshold (default 0.05 → 95% confidence).
        alternative: Direction of the hypothesis test (default: two-sided).

    Returns:
        ABTestResult with raw metrics, confidence intervals, and interpretation.
    """
    # statsmodels convention: count array and nobs array must be aligned.
    # We explicitly name both to prevent the array-order bug.
    counts = [hook_a.actions, hook_b.actions]
    nobs   = [hook_a.views,   hook_b.views]

    z_stat, p_value = proportions_ztest(
        count=counts,
        nobs=nobs,
        alternative=alternative.value,
    )

    # Per-variant Wilson confidence intervals (more accurate than normal approx
    # at low counts or extreme rates).
    ci_a = proportion_confint(hook_a.actions, hook_a.views, alpha=alpha, method="wilson")
    ci_b = proportion_confint(hook_b.actions, hook_b.views, alpha=alpha, method="wilson")

    uplift_pct = (hook_b.rate - hook_a.rate) / hook_a.rate * 100

    result = ABTestResult(
        rate_a=hook_a.rate,
        rate_b=hook_b.rate,
        uplift_pct=uplift_pct,
        z_stat=z_stat,
        p_value=p_value,
        alpha=alpha,
        ci_a=ci_a,
        ci_b=ci_b,
        alternative=alternative,
        label_a=hook_a.label,
        label_b=hook_b.label,
    )

    log.info("A/B test complete: %s", result.verdict)
    return result


# ── Batch helper ──────────────────────────────────────────────────────────────

def run_batch_tests(
    experiments: list[tuple[HookData, HookData]],
    alpha: float = 0.05,
    alternative: Alternative = Alternative.TWO_SIDED,
) -> list[ABTestResult]:
    """
    Run multiple A/B tests and return all results.

    Note on multiple comparisons: running N tests at α=0.05 inflates the
    family-wise error rate. Consider applying a Bonferroni correction by
    passing alpha=0.05/N when testing many hooks simultaneously.
    """
    if not experiments:
        raise ValueError("experiments list is empty.")

    n = len(experiments)
    if n > 1:
        log.warning(
            "Running %d simultaneous tests at α=%.3f. "
            "Consider Bonferroni correction (α=%.4f per test) to control FWER.",
            n, alpha, alpha / n,
        )

    return [run_ab_test(a, b, alpha=alpha, alternative=alternative) for a, b in experiments]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    # Example: "Visual Hook" (control) vs "Text Hook" (variant)
    control = HookData(label="Visual Hook", views=15_000, actions=450)
    variant = HookData(label="Text Hook",   views=16_500, actions=610)

    result = run_ab_test(control, variant)
    print(result.summary())

    # Batch example with Bonferroni correction
    experiments = [
        (HookData("Control", 10_000, 300), HookData("Variant A", 10_200, 340)),
        (HookData("Control", 10_000, 300), HookData("Variant B",  9_800, 280)),
    ]
    batch_results = run_batch_tests(experiments, alpha=0.05 / len(experiments))
    for r in batch_results:
        print(r.summary())