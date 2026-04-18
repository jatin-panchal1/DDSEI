"""
virality_model.py
─────────────────
Trains a Random Forest regressor to predict the Viral Coefficient
from YouTube engagement metrics stored in MySQL.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

# ── Constants ─────────────────────────────────────────────────────────────────

# Virality weight coefficients — adjust to reflect platform priorities.
W_SHARE, W_SAVE, W_COMMENT, W_LIKE = 5.0, 4.0, 2.0, 1.0

CATEGORICAL_FEATURES = ["content_type"]
NUMERIC_FEATURES     = ["views", "likes", "comments", "shares", "saves", "duration_seconds"]

# Minimum views to include a video — keeps the VC calculation stable.
MIN_VIEWS_THRESHOLD = 100

# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    n_estimators: int  = 100
    max_depth: int     = 10
    random_state: int  = 42
    test_size: float   = 0.20
    cv_folds: int      = 2
    model_output_path: Path = field(default_factory=lambda: Path("models/virality_model.joblib"))


# ── Data loading ──────────────────────────────────────────────────────────────

_QUERY = text("""
    SELECT
        em.content_id,
        em.views,
        em.likes,
        em.comments,
        em.shares,
        em.saves,
        cm.duration_seconds,
        cm.content_type
    FROM engagement_metrics em
    JOIN content_metadata cm ON em.content_id = cm.content_id
    WHERE em.views > :min_views
""")


def _build_engine() -> Engine:
    load_dotenv()
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise EnvironmentError(
            "DATABASE_URL is not set. "
            "Export it before running: export DATABASE_URL='mysql+pymysql://...'"
        )
    return create_engine(url, future=True)


def load_data(engine: Engine, min_views: int = MIN_VIEWS_THRESHOLD) -> pd.DataFrame:
    """Fetch engagement + metadata rows from MySQL."""
    log.info("Loading data from database (min_views=%d)…", min_views)
    with engine.connect() as conn:
        df = pd.read_sql(
            _QUERY,
            conn,
            params={"min_views": min_views},
        )

    if df.empty:
        raise ValueError(
            "Query returned 0 rows. "
            "Check that the pipeline has run and engagement_metrics is populated."
        )

    log.info("Loaded %d rows, %d columns.", *df.shape)
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def compute_viral_coefficient(df: pd.DataFrame) -> pd.Series:
    """
    Weighted share of interactions per view, expressed as a percentage.

    Formula:
        VC = ((shares×W_SHARE) + (saves×W_SAVE) +
               (comments×W_COMMENT) + (likes×W_LIKE)) / views × 100

    Views is validated upstream (> MIN_VIEWS_THRESHOLD), but we guard
    against any zero that slips through to avoid inf values.
    """
    safe_views = df["views"].replace(0, np.nan)
    vc = (
        df["shares"]   * W_SHARE   +
        df["saves"]    * W_SAVE    +
        df["comments"] * W_COMMENT +
        df["likes"]    * W_LIKE
    ) / safe_views * 100
    n_invalid = vc.isna().sum()
    if n_invalid:
        log.warning("%d row(s) have views=0 after filtering; dropping them.", n_invalid)
    return vc


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns and one-hot encode categoricals.

    Note: engagement_rate (likes+comments / views) is intentionally excluded
    from the model features because it is a linear combination of columns that
    are already in NUMERIC_FEATURES — including it would introduce multicollinearity
    and leak target-adjacent signal without adding independent information.
    """
    df = df.copy()
    df["viral_coefficient"] = compute_viral_coefficient(df)
    df = df.dropna(subset=["viral_coefficient"])

    # One-hot encode without dropping the first level so every category is explicit.
    # This avoids the dummy-variable trap for tree-based models (trees don't need it).
    encoded = pd.get_dummies(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], drop_first=False)

    return encoded, df["viral_coefficient"]


# ── Model training & evaluation ───────────────────────────────────────────────

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: ModelConfig,
) -> RandomForestRegressor:
    log.info(
        "Training RandomForestRegressor  (n_estimators=%d, max_depth=%d)…",
        cfg.n_estimators, cfg.max_depth,
    )
    model = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        n_jobs=-1,           # use all available cores
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: ModelConfig,
) -> dict:
    """Return a dict of evaluation metrics (hold-out + cross-validated)."""
    predictions = model.predict(X_test)

    hold_out_r2   = r2_score(y_test, predictions)
    hold_out_rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Cross-validated R² gives a more robust estimate than a single split.
    cv_r2_scores = cross_val_score(
        model, X, y,
        cv=cfg.cv_folds,
        scoring="r2",
        n_jobs=-1,
    )

    return {
        "hold_out_r2":   hold_out_r2,
        "hold_out_rmse": hold_out_rmse,
        "cv_r2_mean":    cv_r2_scores.mean(),
        "cv_r2_std":     cv_r2_scores.std(),
    }


def feature_importance_report(
    model: RandomForestRegressor,
    feature_names: pd.Index,
    top_n: int = 10,
) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model: RandomForestRegressor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info("Model saved → %s", path)


def load_model(path: Path) -> RandomForestRegressor:
    if not path.exists():
        raise FileNotFoundError(f"No saved model found at '{path}'.")
    return joblib.load(path)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_virality_model(cfg: ModelConfig | None = None) -> dict:
    """
    End-to-end pipeline: load → engineer → train → evaluate → save.

    Returns:
        A dict with model metrics and the trained model instance.
    """
    cfg    = cfg or ModelConfig()
    engine = _build_engine()

    # 1. Load
    raw_df = load_data(engine)

    # 2. Feature engineering
    X, y = engineer_features(raw_df)
    log.info("Feature matrix: %d rows × %d columns.", *X.shape)

    # 3. Train / test split — stratification not applicable for regression,
    #    but shuffle=True (default) ensures randomness.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )

    # 4. Train
    model = train_model(X_train, y_train, cfg)

    # 5. Evaluate
    metrics = evaluate_model(model, X, y, X_test, y_test, cfg)

    log.info("--- Model Evaluation ---")
    log.info("Hold-Out R2:    %.4f", metrics["hold_out_r2"])
    log.info("CV Mean R2:     %.4f (±%.4f)", metrics["cv_r2_mean"], metrics["cv_r2_std"])
    log.info("Hold-Out RMSE:  %.4f", metrics["hold_out_rmse"])

    # 6. Extract Insights
    importances = feature_importance_report(model, X.columns)
    log.info("\n--- Strategy Report: Top Drivers of Virality ---")
    for _, row in importances.iterrows():
        log.info("%-20s : %.4f", row["feature"], row["importance"])

    # 7. Save Model
    save_model(model, cfg.model_output_path)

    return {"model": model, "metrics": metrics, "importances": importances}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run_virality_model()
    except Exception as e:
        log.error("Pipeline failed: %s", e)
    

