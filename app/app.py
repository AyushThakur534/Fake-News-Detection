"""Fake News Detection + evidence retrieval Streamlit app."""
import json
import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocess import clean_text
from src.verify_claims import verify_claim

MODEL_PATH = ROOT / "models" / "fake_news_pipeline.joblib"
RESULTS_PATH = ROOT / "models" / "model_comparison.csv"
SUMMARY_PATH = ROOT / "models" / "training_summary.json"

st.set_page_config(page_title="Fake News Detection & Claim Review", page_icon="📰", layout="wide")
st.title("Fake News Detection & Claim Verification")
st.caption("5th Semester Data Mining Project")
st.warning(
    "The classifier learns text patterns; it does not prove truth. Evidence search "
    "retrieves sources for review and does not automatically declare a claim true or false."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = None
if MODEL_PATH.exists():
    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
else:
    st.info("Train the model with: python -m src.train --data data/news.csv")

tab1, tab2, tab3 = st.tabs(["Text Classifier", "Evidence Verification", "Model Evaluation"])

with tab1:
    st.subheader("Classify a news article")
    article = st.text_area("Headline and/or article text", height=230, key="article_input")
    if st.button("Classify Text", type="primary"):
        if model is None:
            st.error("Train the model before classification.")
        else:
            cleaned = clean_text(article)
            if len(cleaned) < 20:
                st.warning("Enter a longer article (at least 20 characters).")
            else:
                try:
                    prediction = model.predict([cleaned])[0]
                    if prediction == "FAKE":
                        st.error("Text-pattern prediction: FAKE")
                    else:
                        st.success("Text-pattern prediction: REAL")
                    st.write(
                        "This is a learned classification, not a fact-check. Verify specific "
                        "names, dates, statistics, and quotations separately."
                    )
                    classifier = model.named_steps["classifier"]
                    if hasattr(model, "predict_proba"):
                        scores = model.predict_proba([cleaned])[0]
                        st.write("Model class scores (not proof of truth):")
                        st.dataframe(pd.DataFrame({
                            "Label": classifier.classes_,
                            "Score": [round(float(v), 4) for v in scores],
                        }), hide_index=True, use_container_width=True)
                    elif hasattr(model, "decision_function"):
                        st.caption(
                            "This model returns decision scores, not calibrated probabilities. "
                            f"Raw score: {model.decision_function([cleaned]).tolist()}"
                        )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

with tab2:
    st.subheader("Search evidence for a specific claim")
    st.write(
        "Enter a focused claim, including relevant names, numbers, and dates. "
        "The tool searches for existing fact-check reviews and related web sources."
    )
    claim = st.text_area("Claim to verify", height=130, key="claim_input")
    if st.button("Search for Evidence", type="primary"):
        if len(claim.strip()) < 12:
            st.warning("Enter a specific claim of at least 12 characters.")
        else:
            with st.spinner("Searching for fact-checks and sources..."):
                try:
                    result = verify_claim(claim)
                    st.info(result["status"])
                    st.write(result["notice"])
                    if result["fact_checks"]:
                        st.markdown("#### Existing fact-check reviews")
                        for item in result["fact_checks"]:
                            st.markdown(f"**{item['title']}**")
                            st.write(item["snippet"])
                            if item["url"]:
                                st.markdown(f"[Open fact-check review]({item['url']})")
                            st.divider()
                    if result["web_results"]:
                        st.markdown("#### Web search results")
                        for item in result["web_results"]:
                            st.markdown(f"**{item['title']}**")
                            if item["source"]:
                                st.caption(f"Source: {item['source']}")
                            st.write(item["snippet"] or "No snippet supplied.")
                            if item["url"]:
                                st.markdown(f"[Open source]({item['url']})")
                            st.divider()
                    if not result["fact_checks"] and not result["web_results"]:
                        st.warning(
                            "No usable results were returned. This does not mean the claim is true."
                        )
                except Exception as exc:
                    st.error(
                        "Evidence search failed. Check internet connectivity and try again. "
                        f"Details: {exc}"
                    )

with tab3:
    st.subheader("Model evaluation")
    if RESULTS_PATH.exists():
        st.dataframe(pd.read_csv(RESULTS_PATH), hide_index=True, use_container_width=True)
        st.caption(
            "Chronological holdout tests later articles when dates are usable, but does not "
            "prove generalization to unseen publishers."
        )
    else:
        st.info("Run the training script to generate model evaluation results.")
    if SUMMARY_PATH.exists():
        try:
            st.markdown("#### Training summary")
            st.json(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
        except Exception as exc:
            st.error(f"Could not read summary: {exc}")

st.divider()
st.caption(
    "A missing record is not proof of zero activity. Compare exact claims with dated, "
    "primary or otherwise reliable sources."
)
