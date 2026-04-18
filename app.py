"""
app.py - DDSEI Control Center Frontend

A production-grade Streamlit application for the DDSEI Platform,
providing data visualization, A/B testing analysis, virality prediction,
and live Omni-Channel data extraction.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional
import pandas as pd

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Local module imports
from ab_test import HookData, run_ab_test
from virality_model import load_model 
from pipeline_orchestrator import run_daily_pipeline, PipelineConfig, extract_youtube_id
from pipeline_orchestrator import run_daily_pipeline, PipelineConfig, extract_youtube_id, extract_ig_shortcode

# ----------------------------------------------------------------------
# Environment & Configuration
# ----------------------------------------------------------------------
load_dotenv()  # Load environment variables from .env file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppConfig:
    """Centralized application configuration loaded from environment."""

    POWERBI_EMBED_URL: str = os.getenv(
        "POWERBI_EMBED_URL",
        "https://app.powerbi.com/reportEmbed?reportId=YOUR_REPORT_ID",
    )
    VIRALITY_MODEL_PATH: str = os.getenv("VIRALITY_MODEL_PATH", "models/virality_model.joblib")
    MAX_INPUT_VALUE: int = 10_000_000  # Prevent extreme values


config = AppConfig()

# ----------------------------------------------------------------------
# Page Setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Unlox Engagement Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📱 Unlox: Data-Driven Social Engagement")
st.markdown("---")

# Create navigation tabs (Added the Content Analyzer tab!)
tab_dashboard, tab_extractor, tab_ab_test, tab_predictor = st.tabs(
    ["📊 PowerBI Dashboard", "📥 Content Analyzer", "🧪 A/B Testing Lab", "🤖 Virality Predictor"]
)

# ----------------------------------------------------------------------
# Helper Functions with Caching
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
@st.cache_resource(show_spinner=False)
def get_virality_model():
    import pathlib
    """
    Load the virality prediction model with caching.
    """
    try:
        logger.info(f"Loading virality model from {config.VIRALITY_MODEL_PATH}")
        
        # 1. Convert the string into a Path object
        model_path = pathlib.Path(config.VIRALITY_MODEL_PATH) 
        
        # 2. Pass the Path object into your load function
        model = load_model(model_path)
        
        return model
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        st.error("Virality model could not be loaded. Please contact support.")
        return None
    except Exception as e:
        logger.exception("Unexpected error loading virality model")
        st.error("An unexpected error occurred while loading the prediction model.")
        return None


def validate_ab_inputs(views_a: int, actions_a: int, views_b: int, actions_b: int) -> Optional[str]:
    """Validate A/B test input values."""
    if actions_a > views_a:
        return "Actions (A) cannot exceed Views (A)."
    if actions_b > views_b:
        return "Actions (B) cannot exceed Views (B)."
    if views_a <= 0 or views_b <= 0:
        return "Views must be positive integers."
    if actions_a < 0 or actions_b < 0:
        return "Actions cannot be negative."
    if max(views_a, views_b, actions_a, actions_b) > config.MAX_INPUT_VALUE:
        return f"Values cannot exceed {config.MAX_INPUT_VALUE:,}."
    return None

def predict_virality(model, features: dict) -> float:
    """Bridges the Streamlit UI inputs to the Scikit-learn Random Forest model."""
    
    # We must provide exactly the 8 columns the model was trained on!
    model_input = {
        "views": features.get("views_24h", 5000),
        "likes": features.get("likes", 250),
        "comments": 50,  
        "shares": features.get("shares", 80),
        "saves": 30,     
        "duration_seconds": features.get("duration", 15),
        "content_type_Short": 1, # Defaulting to Short for the prediction
        "content_type_Video": 0  # <--- Added the missing column!
    }
    
    # Convert to dataframe
    input_df = pd.DataFrame([model_input])
    
    # Make the prediction
    prediction = model.predict(input_df)[0]
    
    return float(prediction)

# ----------------------------------------------------------------------
# Tab 1: PowerBI Dashboard
# ----------------------------------------------------------------------
with tab_dashboard:
    st.header("Growth & Sentiment Visualization")
    st.markdown("Live reporting from the MySQL database. *Note: Ensure you have appropriate PowerBI access if prompted.*")

    try:
        with st.container():
            components.iframe(
                src=config.POWERBI_EMBED_URL,
                width=None,
                height=800,
                scrolling=True,
            )
    except Exception as e:
        logger.error(f"PowerBI embed failed: {e}")
        st.error("Unable to load the PowerBI dashboard. Please verify the embed URL and your network connection.")

# ----------------------------------------------------------------------
# Tab 2: Content Analyzer (NEW - Pipeline Orchestrator)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Tab 2: Content Analyzer (NEW - Pipeline Orchestrator)
# ----------------------------------------------------------------------
with tab_extractor:
    st.header("📥 Omni-Channel Data Extractor")
    st.markdown("Paste a YouTube link or Instagram link below to extract live metrics, pull comments, and analyze audience sentiment into MySQL.")

    # Input Form
    with st.form("pipeline_form"):
        col1, col2 = st.columns(2)
        with col1:
            yt_url = st.text_input("YouTube Video Link", placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        with col2:
            # RESTORED: The text input box is back! We named the variable 'ig_url'
            ig_url = st.text_input("Instagram Link (Optional)", placeholder="https://www.instagram.com/p/DXPfCfQj3Up/")
        
        submitted = st.form_submit_button("🚀 Run AI Analysis", type="primary", use_container_width=True)

    # Execution Block
    if submitted:
        if not yt_url and not ig_url:
            st.warning("⚠️ Please enter at least one link to analyze.")
        else:
            # We pass 'ig_url' into the function 'extract_ig_shortcode'
            yt_ids = [extract_youtube_id(yt_url)] if yt_url else []
            ig_ids = [extract_ig_shortcode(ig_url)] if ig_url else []

            with st.spinner("🤖 Fetching APIs, scraping comments, and running NLP Engine..."):
                cfg = PipelineConfig(dry_run=False)
                result = run_daily_pipeline(yt_ids, ig_ids, cfg=cfg)

                if result.success:
                    st.success("✅ Analysis Complete! Data saved to MySQL.")
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Comments Analyzed", result.comments_analysed)
                    with metric_col2:
                        st.metric("Relatable Rate", f"{result.relatable_rate}%")
                    with metric_col3:
                        st.metric("DB Rows Saved", result.metric_rows_upserted)
                else:
                    st.error("❌ Pipeline finished with errors.")
                    for err in result.errors:
                        st.write(f"- {err}")

# ----------------------------------------------------------------------
# Tab 3: A/B Testing Lab
# ----------------------------------------------------------------------
with tab_ab_test:
    st.header("Hook Performance Testing")
    st.markdown("Compare two hook variants using a two-proportion z-test.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🔵 Control Hook (A)")
        views_a = st.number_input("Views (A)", min_value=1, value=10000, step=100, format="%d", key="views_a")
        actions_a = st.number_input("Actions (A)", min_value=0, value=500, step=10, format="%d", key="actions_a")

    with col_b:
        st.subheader("🟢 Variant Hook (B)")
        views_b = st.number_input("Views (B)", min_value=1, value=10000, step=100, format="%d", key="views_b")
        actions_b = st.number_input("Actions (B)", min_value=0, value=650, step=10, format="%d", key="actions_b")

    if st.button("🚀 Run Statistical Analysis", key="ab_test_btn", type="primary", use_container_width=True):
        error_msg = validate_ab_inputs(views_a, actions_a, views_b, actions_b)
        if error_msg:
            st.error(f"⚠️ Input Error: {error_msg}")
        else:
            with st.spinner("Calculating statistical significance..."):
                try:
                    hook_a = HookData("Control", views_a, actions_a)
                    hook_b = HookData("Variant", views_b, actions_b)
                    result = run_ab_test(hook_a, hook_b)

                    if result.is_significant:
                        winner = "Variant B" if result.rate_b > result.rate_a else "Control A"
                        st.success(f"✅ Statistically Significant! Winner: **{winner}** (p-value = {result.p_value:.4f})")
                        st.balloons()
                    else:
                        st.warning(f"⚠️ Result is inconclusive. The difference is not statistically significant (p-value = {result.p_value:.4f}).")

                    with st.expander("📋 View Statistical Details"):
                        st.code(result.summary(), language="text")
                except Exception as e:
                    logger.exception("A/B test analysis failed")
                    st.error("An internal error occurred during analysis.")

# ----------------------------------------------------------------------
# Tab 4: Virality Predictor
# ----------------------------------------------------------------------
with tab_predictor:
    st.header("📈 Pre-Publish Virality Predictor")
    st.markdown("Estimate the viral coefficient of your content based on expected early metrics.")

    model = get_virality_model()
    if model is None:
        st.stop()

    with st.form("virality_form"):
        col1, col2 = st.columns(2)
        with col1:
            expected_views = st.slider("Expected Views (first 24h)", min_value=0, max_value=1_000_000, value=5000, step=100, format="%d")
            expected_likes = st.slider("Expected Likes", min_value=0, max_value=100_000, value=250, step=10)
        with col2:
            expected_shares = st.slider("Expected Shares", min_value=0, max_value=50_000, value=80, step=5)
            video_duration = st.slider("Video Duration (seconds)", min_value=1, max_value=300, value=15)

        submit = st.form_submit_button("🔮 Predict Virality", use_container_width=True)

    if submit:
        with st.spinner("Running prediction model..."):
            try:
                features = {"views_24h": expected_views, "likes": expected_likes, "shares": expected_shares, "duration": video_duration}
                viral_coefficient = predict_virality(model, features)
                st.metric(label="Viral Coefficient (k)", value=f"{viral_coefficient:.3f}")

                if viral_coefficient >= 1.2:
                    st.success("🔥 High virality potential! This content is likely to spread rapidly.")
                elif viral_coefficient >= 0.8:
                    st.info("📊 Moderate virality. With a little boost, it could catch on.")
                else:
                    st.warning("🐢 Low virality. Consider refining the hook or targeting a different audience.")
            except Exception as e:
                logger.exception("Virality prediction failed")
                st.error("Prediction failed. Please check the model and input values.")

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown("---")
st.caption("© 2026 Unlox • DDSEI Platform • v1.0.0")