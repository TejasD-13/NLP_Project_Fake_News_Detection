"""
TruthLens – Fake News Detection System
Streamlit Frontend Application
Trained on WELFake Dataset using NLP & Machine Learning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import load_artifacts, predict_news, preprocess_text

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Light Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TruthLens – Fake News Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Human-Made Bright Theme CSS with Navy & Blue Accents
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
    }

    /* Main background */
    .stApp {
        background-color: #f8fafc;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3.5rem;
        max-width: 1200px;
    }

    /* Brand Header Box */
    .brand-banner {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 16px;
    }

    .brand-title {
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .brand-title span {
        color: #2563eb;
    }

    .brand-badge {
        background-color: #eff6ff;
        color: #1d4ed8;
        font-size: 13px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
        border: 1px solid #bfdbfe;
        display: inline-block;
    }

    .brand-subtitle {
        color: #475569;
        font-size: 14px;
        margin-top: 4px;
        line-height: 1.5;
    }

    /* Professional Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07);
    }

    .metric-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
    }

    .metric-value.highlight-blue {
        color: #1d4ed8;
    }

    .metric-value.highlight-emerald {
        color: #059669;
    }

    .metric-sub {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 6px;
    }

    /* Content Cards */
    .content-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
    }

    .content-card-title {
        font-size: 18px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* NLP Pipeline Flowchart */
    .workflow-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        flex-wrap: wrap;
        padding: 16px;
        background: #f8fafc;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }

    .workflow-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        flex: 1;
        min-width: 120px;
        padding: 8px 12px;
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
    }

    .workflow-num {
        width: 22px;
        height: 22px;
        background: #2563eb;
        color: #ffffff;
        border-radius: 50%;
        font-size: 11px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 6px;
    }

    .workflow-title {
        font-size: 13px;
        font-weight: 600;
        color: #1e293b;
    }

    .workflow-desc {
        font-size: 11px;
        color: #64748b;
        margin-top: 2px;
    }

    .workflow-arrow {
        color: #94a3b8;
        font-size: 18px;
        font-weight: 700;
    }

    /* Result Badges */
    .result-alert-fake {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 22px 24px;
        margin: 20px 0;
    }

    .result-alert-real {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 22px 24px;
        margin: 20px 0;
    }

    .result-header {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .result-icon {
        font-size: 32px;
    }

    .result-title-fake {
        color: #b91c1c;
        font-size: 24px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.01em;
    }

    .result-title-real {
        color: #047857;
        font-size: 24px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.01em;
    }

    .result-desc {
        font-size: 14px;
        color: #334155;
        margin-top: 8px;
        line-height: 1.5;
    }

    /* Disclaimer Box */
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 13px;
        color: #92400e;
        line-height: 1.5;
        margin-top: 18px;
    }

    /* Preprocessed token view */
    .token-box {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        background: #f1f5f9;
        padding: 12px 14px;
        border-radius: 8px;
        border: 1px solid #cbd5e1;
        color: #1e293b;
        max-height: 140px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* Buttons styling */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 600;
        font-size: 14px;
        padding: 8px 20px;
        transition: all 0.15s ease;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    .sidebar-header {
        padding: 12px 0 20px 0;
        border-bottom: 1px solid #f1f5f9;
        margin-bottom: 16px;
    }

    .sidebar-title {
        font-size: 20px;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }

    .sidebar-info-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px;
        margin-top: 20px;
        font-size: 12px;
        color: #475569;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Resource Loading & Verification
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_system_artifacts():
    try:
        models, vectorizer, eval_df = load_artifacts()
        return models, vectorizer, eval_df, None
    except Exception as e:
        return {}, None, None, str(e)


models_dict, tfidf_vectorizer, eval_df, load_err = get_system_artifacts()

if load_err:
    st.error(f"Error loading system artifacts: {load_err}")
    st.stop()


# -----------------------------------------------------------------------------
# 3. Sidebar Navigation & Metadata
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header">
            <h2 class="sidebar-title">🛡️ Truth<span>Lens</span></h2>
            <div style="font-size:12px; color:#64748b; font-weight:500;">
                Fake News Detection System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page_selection = st.radio(
        "Navigation",
        options=["Home", "Fake News Detection", "Model Performance"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Project metadata in sidebar
    st.markdown(
        """
        <div class="sidebar-info-card">
            <div style="font-weight:700; color:#0f172a; margin-bottom:6px;">📌 Project Details</div>
            <div><b>Dataset:</b> WELFake Benchmark</div>
            <div><b>Cleaned Records:</b> 63,289</div>
            <div><b>Default Model:</b> Linear SVM</div>
            <div><b>Best Macro F1:</b> 96.92%</div>
            <div><b>Best Accuracy:</b> 96.94%</div>
            <div style="margin-top:8px; padding-top:8px; border-top:1px solid #e2e8f0; font-size:11px; color:#64748b;">
                Built with Python 3.11 • Scikit-learn • Streamlit
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 4. PAGE 1: HOME
# -----------------------------------------------------------------------------
if page_selection == "Home":
    st.markdown(
        """
        <div class="brand-banner">
            <div>
                <h1 class="brand-title">🛡️ Truth<span>Lens</span> Fake News Detection</h1>
                <p class="brand-subtitle">
                    An advanced Natural Language Processing system trained on the <b>WELFake dataset</b> to accurately classify news articles as authentic or deceptive.
                </p>
            </div>
            <div class="brand-badge">
                Linear SVM • 96.94% Accuracy
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key Project Metrics
    st.markdown(
        """
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Original Dataset</div>
                <div class="metric-value">72,134</div>
                <div class="metric-sub">Raw news articles collected</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Cleaned Records</div>
                <div class="metric-value highlight-blue">63,289</div>
                <div class="metric-sub">Deduplicated & verified tokens</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Fake News Articles</div>
                <div class="metric-value" style="color:#dc2626;">34,519</div>
                <div class="metric-sub">54.54% of dataset (Label 0)</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Real News Articles</div>
                <div class="metric-value highlight-emerald">28,770</div>
                <div class="metric-sub">45.46% of dataset (Label 1)</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">ML Models Trained</div>
                <div class="metric-value">4</div>
                <div class="metric-sub">SVM, LR, RF, Naive Bayes</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Best Model Accuracy</div>
                <div class="metric-value highlight-blue">96.94%</div>
                <div class="metric-sub">Linear SVM (Macro F1: 96.92%)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Visual Distribution & Model Overview Section
    col_dist, col_info = st.columns([1, 1], gap="medium")

    with col_dist:
        st.markdown(
            """
            <div class="content-card">
                <div class="content-card-title">📊 Dataset Class Distribution</div>
            """,
            unsafe_allow_html=True,
        )

        # Plotly Donut Chart
        labels = ["Fake News", "Real News"]
        values = [34519, 28770]
        colors = ["#ef4444", "#10b981"]

        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.6,
                    marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                    textinfo="label+percent",
                    textfont=dict(size=13, family="Plus Jakarta Sans"),
                    hoverinfo="label+value+percent",
                )
            ]
        )
        fig_donut.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=260,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=12),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_info:
        st.markdown(
            """
            <div class="content-card">
                <div class="content-card-title">⚡ Machine Learning Models Overview</div>
                <p style="font-size:13.5px; color:#475569; line-height:1.6; margin-bottom:12px;">
                    Four diverse classification architectures were trained and benchmarked using <b>stratified 80/20 train-test splits</b> with <b>macro-averaged</b> evaluation:
                </p>
                <div style="font-size:13px; line-height:1.8; color:#1e293b;">
                    <div>• <b>Linear SVM:</b> Best model with <b>96.94% accuracy</b> and <b>96.92% macro F1</b>. Maximizes hyper-plane margins for high-dimensional sparse TF-IDF text.</div>
                    <div>• <b>Logistic Regression:</b> 96.11% accuracy, providing reliable probabilistic margins.</div>
                    <div>• <b>Random Forest:</b> 93.62% accuracy, ensemble of 100 balanced decision trees.</div>
                    <div>• <b>Multinomial Naive Bayes:</b> 88.36% accuracy, fast probabilistic baseline.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # End-to-End NLP Workflow
    st.markdown(
        """
        <div class="content-card">
            <div class="content-card-title">🔄 End-to-End NLP Processing Pipeline</div>
            <div class="workflow-container">
                <div class="workflow-step">
                    <div class="workflow-num">1</div>
                    <div class="workflow-title">Dataset</div>
                    <div class="workflow-desc">72,134 WELFake records</div>
                </div>
                <div class="workflow-arrow">→</div>
                <div class="workflow-step">
                    <div class="workflow-num">2</div>
                    <div class="workflow-title">Data Cleaning</div>
                    <div class="workflow-desc">Deduplication & null filters</div>
                </div>
                <div class="workflow-arrow">→</div>
                <div class="workflow-step">
                    <div class="workflow-num">3</div>
                    <div class="workflow-title">Text Preprocessing</div>
                    <div class="workflow-desc">NLTK stop & lemmatization</div>
                </div>
                <div class="workflow-arrow">→</div>
                <div class="workflow-step">
                    <div class="workflow-num">4</div>
                    <div class="workflow-title">TF-IDF Vectorizer</div>
                    <div class="workflow-desc">30k n-grams (1, 2)</div>
                </div>
                <div class="workflow-arrow">→</div>
                <div class="workflow-step">
                    <div class="workflow-num">5</div>
                    <div class="workflow-title">ML Models</div>
                    <div class="workflow-desc">Linear SVM default</div>
                </div>
                <div class="workflow-arrow">→</div>
                <div class="workflow-step" style="border: 2px solid #2563eb; background:#eff6ff;">
                    <div class="workflow-num">6</div>
                    <div class="workflow-title" style="color:#1d4ed8;">Prediction</div>
                    <div class="workflow-desc">Fake (0) or Real (1)</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 5. PAGE 2: FAKE NEWS DETECTION
# -----------------------------------------------------------------------------
elif page_selection == "Fake News Detection":
    st.markdown(
        """
        <div class="brand-banner">
            <div>
                <h1 class="brand-title">🔍 Fake News <span>Detection</span></h1>
                <p class="brand-subtitle">
                    Enter a news headline and full article text to analyze its credibility using trained machine learning models.
                </p>
            </div>
            <div class="brand-badge">
                Interactive Classifier
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state for input fields
    if "headline_input" not in st.session_state:
        st.session_state["headline_input"] = ""
    if "article_input" not in st.session_state:
        st.session_state["article_input"] = ""
    if "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = None

    # Sample articles for quick testing
    sample_options = {
        "— Select a sample to load —": ("", ""),
        "Real News Sample (Scientific Discovery)": (
            "NASA Mars Rover Discovers Ancient Stream Sediments",
            "Scientists analyzing data from NASA's Mars rover mission have identified geological sediment layers that confirm the long-term presence of liquid water streams billions of years ago. The research findings were published in a peer-reviewed scientific journal following extensive validation."
        ),
        "Real News Sample (Economic Report)": (
            "Central Bank Releases Official Annual Economic Report",
            "The central bank published its comprehensive annual monetary report today, detailing inflation benchmarks, interest rate projections, and economic growth statistics across multiple industrial sectors."
        ),
        "Fake News Sample (Sensational Conspiracy)": (
            "BREAKING: Secret Alien Technology Found Operating in Government Basement",
            "Anonymous social media posts claim that high-ranking officials have been operating extraterrestrial energy devices secretly since 1952. No verified sources or scientific evidence have been presented to substantiate these sensational claims."
        ),
        "Fake News Sample (Fabricated Political Claim)": (
            "Shocking Revelation: Secret Executive Order To Ban All Elections Tomorrow",
            "A viral chain message claims that an immediate executive directive has canceled all regional voting procedures nationwide starting tomorrow morning. No official register or legislative confirmation exists."
        )
    }

    st.markdown("<div class='content-card'>", unsafe_allow_html=True)

    col_sample, col_model = st.columns([1, 1], gap="medium")

    with col_sample:
        selected_sample = st.selectbox(
            "💡 Load Quick Test Sample (Optional):",
            options=list(sample_options.keys()),
            index=0,
            help="Select a benchmark sample to automatically populate the input fields."
        )
        if selected_sample != "— Select a sample to load —":
            samp_head, samp_art = sample_options[selected_sample]
            st.session_state["headline_input"] = samp_head
            st.session_state["article_input"] = samp_art

    with col_model:
        model_options = [
            "Linear SVM",
            "Logistic Regression",
            "Random Forest",
            "Naive Bayes"
        ]
        # Linear SVM is default
        selected_model_name = st.selectbox(
            "⚙️ Select Machine Learning Model:",
            options=model_options,
            index=0,
            help="Linear SVM is the default best-performing model (96.94% accuracy)."
        )

    # Input Fields
    headline_val = st.text_input(
        "📰 News Headline",
        value=st.session_state.get("headline_input", ""),
        placeholder="e.g., Government announces new economic growth plan...",
        key="headline_field"
    )

    article_val = st.text_area(
        "📝 News Article Content",
        value=st.session_state.get("article_input", ""),
        height=190,
        placeholder="Paste the full article text or paragraphs here...",
        key="article_field"
    )

    col_btn_pred, col_btn_clear, col_space = st.columns([1.2, 1, 3.8])

    with col_btn_pred:
        predict_clicked = st.button("🚀 Analyze Article", type="primary", use_container_width=True)

    with col_btn_clear:
        clear_clicked = st.button("🗑️ Clear Inputs", use_container_width=True)

    if clear_clicked:
        st.session_state["headline_input"] = ""
        st.session_state["article_input"] = ""
        st.session_state["prediction_result"] = None
        st.rerun()

    # Perform Prediction
    if predict_clicked:
        if not headline_val.strip() and not article_val.strip():
            st.warning("⚠️ Please provide at least a news headline or article body before analyzing.")
        else:
            with st.spinner("Applying NLP preprocessing and generating model prediction..."):
                res = predict_news(
                    headline=headline_val,
                    article=article_val,
                    model_name=selected_model_name,
                    models=models_dict,
                    vectorizer=tfidf_vectorizer
                )
                st.session_state["prediction_result"] = res

    # Display Prediction Results
    if st.session_state.get("prediction_result"):
        res = st.session_state["prediction_result"]

        if not res.get("success", False):
            st.error(f"❌ Analysis Error: {res.get('error', 'Unable to process text.')}")
        else:
            is_fake = res["is_fake"]

            if is_fake:
                st.markdown(
                    f"""
                    <div class="result-alert-fake">
                        <div class="result-header">
                            <span class="result-icon">🚨</span>
                            <div>
                                <h3 class="result-title-fake">FAKE NEWS DETECTED</h3>
                                <div class="result-desc">
                                    The textual patterns and vocabulary distribution in this article strongly correlate with <b>deceptive, misleading, or fabricated content</b> identified in the WELFake benchmark dataset.
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-alert-real">
                        <div class="result-header">
                            <span class="result-icon">✅</span>
                            <div>
                                <h3 class="result-title-real">REAL NEWS DETECTED</h3>
                                <div class="result-desc">
                                    The textual structure, vocabulary, and semantic attributes of this article match patterns characteristic of <b>authentic, verified journalistic news</b> in the WELFake benchmark dataset.
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Metadata Row
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Model Evaluator", res["model_name"])
            with col_m2:
                st.metric("Input Words", f"{res['input_word_count']:,}")
            with col_m3:
                st.metric("Cleaned Tokens", f"{res['cleaned_word_count']:,}")
            with col_m4:
                st.metric("TF-IDF Dimensions", "30,000 max features")

            # Expandable Preprocessed Text View
            with st.expander("🔍 View Preprocessed & Lemmatized Tokens"):
                st.markdown(
                    "<p style='font-size:12px; color:#64748b; margin-bottom:6px;'>Text after lowercase conversion, URL/HTML removal, punctuation stripping, stopword filtering, and WordNet lemmatization:</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='token-box'>{res['cleaned_text']}</div>",
                    unsafe_allow_html=True,
                )

            # Mandatory Disclaimer Box
            st.markdown(
                """
                <div class="disclaimer-box">
                    <b>⚠️ Notice & Disclaimer:</b> This system identifies patterns learned from the WELFake dataset. It does not independently verify the factual accuracy of news articles.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. PAGE 3: MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif page_selection == "Model Performance":
    st.markdown(
        """
        <div class="brand-banner">
            <div>
                <h1 class="brand-title">📊 Model <span>Performance Benchmark</span></h1>
                <p class="brand-subtitle">
                    Comprehensive comparison of all four trained machine learning algorithms evaluated on 12,658 test articles from the WELFake dataset.
                </p>
            </div>
            <div class="brand-badge">
                Macro-Averaged Metrics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Actual Verified Evaluation Data (Macro Averaged)
    perf_data = [
        {
            "Model": "Linear SVM",
            "Accuracy": 96.94,
            "Precision": 96.92,
            "Recall": 96.91,
            "F1-Score": 96.92,
            "ROC-AUC": 99.58,
            "Training Time (s)": 2.02,
            "Rank": "1st (Best Model)"
        },
        {
            "Model": "Logistic Regression",
            "Accuracy": 96.11,
            "Precision": 96.08,
            "Recall": 96.08,
            "F1-Score": 96.08,
            "ROC-AUC": 99.35,
            "Training Time (s)": 1.24,
            "Rank": "2nd"
        },
        {
            "Model": "Random Forest",
            "Accuracy": 93.62,
            "Precision": 93.55,
            "Recall": 93.58,
            "F1-Score": 93.57,
            "ROC-AUC": 98.49,
            "Training Time (s)": 14.11,
            "Rank": "3rd"
        },
        {
            "Model": "Naive Bayes",
            "Accuracy": 88.36,
            "Precision": 88.22,
            "Recall": 88.45,
            "F1-Score": 88.30,
            "ROC-AUC": 95.28,
            "Training Time (s)": 0.27,
            "Rank": "4th"
        }
    ]
    df_perf = pd.DataFrame(perf_data)

    # Performance Table Card
    st.markdown(
        """
        <div class="content-card">
            <div class="content-card-title">🏆 Model Comparison Table</div>
        """,
        unsafe_allow_html=True,
    )

    # Display styled table
    display_df = df_perf.copy()
    display_df["Accuracy"] = display_df["Accuracy"].map("{:.2f}%".format)
    display_df["Precision"] = display_df["Precision"].map("{:.2f}%".format)
    display_df["Recall"] = display_df["Recall"].map("{:.2f}%".format)
    display_df["F1-Score"] = display_df["F1-Score"].map("{:.2f}%".format)
    display_df["ROC-AUC"] = display_df["ROC-AUC"].map("{:.2f}%".format)
    display_df["Training Time (s)"] = display_df["Training Time (s)"].map("{:.2f}s".format)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Model": st.column_config.TextColumn("Algorithm Name", width="medium"),
            "Rank": st.column_config.TextColumn("Benchmark Standing", width="small"),
            "Accuracy": st.column_config.TextColumn("Accuracy", width="small"),
            "Precision": st.column_config.TextColumn("Macro Precision", width="small"),
            "Recall": st.column_config.TextColumn("Macro Recall", width="small"),
            "F1-Score": st.column_config.TextColumn("Macro F1", width="small"),
            "ROC-AUC": st.column_config.TextColumn("ROC-AUC", width="small"),
            "Training Time (s)": st.column_config.TextColumn("Train Time", width="small"),
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Interactive Charts
    col_c1, col_c2 = st.columns([1, 1], gap="medium")

    with col_c1:
        st.markdown(
            """
            <div class="content-card">
                <div class="content-card-title">📈 Model Accuracy Comparison</div>
            """,
            unsafe_allow_html=True,
        )
        bar_colors = ["#2563eb", "#60a5fa", "#94a3b8", "#cbd5e1"]
        fig_acc = px.bar(
            df_perf,
            x="Model",
            y="Accuracy",
            text="Accuracy",
            color="Model",
            color_discrete_sequence=bar_colors,
        )
        fig_acc.update_traces(
            texttemplate="%{y:.2f}%",
            textposition="outside",
            marker_line_color="#1e293b",
            marker_line_width=1,
        )
        fig_acc.update_layout(
            yaxis=dict(range=[80, 100], title="Accuracy (%)", gridcolor="#f1f5f9"),
            xaxis=dict(title=""),
            height=320,
            margin=dict(t=20, b=20, l=10, r=10),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_acc, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_c2:
        st.markdown(
            """
            <div class="content-card">
                <div class="content-card-title">🎯 ROC-AUC Comparison</div>
            """,
            unsafe_allow_html=True,
        )
        fig_auc = px.bar(
            df_perf,
            x="Model",
            y="ROC-AUC",
            text="ROC-AUC",
            color="Model",
            color_discrete_sequence=["#10b981", "#34d399", "#94a3b8", "#cbd5e1"],
        )
        fig_auc.update_traces(
            texttemplate="%{y:.2f}%",
            textposition="outside",
            marker_line_color="#064e3b",
            marker_line_width=1,
        )
        fig_auc.update_layout(
            yaxis=dict(range=[90, 102], title="ROC-AUC (%)", gridcolor="#f1f5f9"),
            xaxis=dict(title=""),
            height=320,
            margin=dict(t=20, b=20, l=10, r=10),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_auc, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Grouped Precision, Recall & F1 Chart
    st.markdown(
        """
        <div class="content-card">
            <div class="content-card-title">📊 Multi-Metric Comparison (Precision, Recall, F1-Score)</div>
        """,
        unsafe_allow_html=True,
    )

    metrics_melted = pd.melt(
        df_perf,
        id_vars=["Model"],
        value_vars=["Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score"
    )

    fig_grouped = px.bar(
        metrics_melted,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        color_discrete_sequence=["#2563eb", "#10b981", "#f59e0b"],
    )
    fig_grouped.update_layout(
        yaxis=dict(range=[80, 100], title="Score (%)", gridcolor="#f1f5f9"),
        xaxis=dict(title=""),
        height=340,
        margin=dict(t=20, b=20, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_grouped, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Actual Saved Test Confusion Matrices
    st.markdown(
        """
        <div class="content-card">
            <div class="content-card-title">🧩 Test Confusion Matrices (12,658 Test Articles)</div>
            <p style="font-size:13px; color:#64748b; margin-bottom:16px;">
                Confusion matrices calculated directly from the test split (6,904 Fake News articles and 5,754 Real News articles):
            </p>
        """,
        unsafe_allow_html=True,
    )

    # Confusion matrix data:
    # rows: Actual [Fake, Real], cols: Predicted [Fake, Real]
    cm_dict = {
        "Linear SVM (Best Model - 96.94%)": np.array([[6697, 207], [178, 5576]]),
        "Logistic Regression (96.11%)": np.array([[6630, 274], [218, 5536]]),
        "Random Forest (93.62%)": np.array([[6495, 409], [399, 5355]]),
        "Naive Bayes (88.36%)": np.array([[6009, 895], [579, 5175]]),
    }

    cols_cm = st.columns(4)
    for idx, (m_title, matrix) in enumerate(cm_dict.items()):
        with cols_cm[idx]:
            st.markdown(f"<div style='font-size:13px; font-weight:700; color:#1e293b; margin-bottom:6px; text-align:center;'>{m_title.split('(')[0]}</div>", unsafe_allow_html=True)
            fig_cm = px.imshow(
                matrix,
                text_auto=True,
                x=["Pred Fake", "Pred Real"],
                y=["True Fake", "True Real"],
                color_continuous_scale="Blues",
            )
            fig_cm.update_layout(
                height=220,
                margin=dict(t=10, b=10, l=10, r=10),
                coloraxis_showscale=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

    st.markdown("</div>", unsafe_allow_html=True)
