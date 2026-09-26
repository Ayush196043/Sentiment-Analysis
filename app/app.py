import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Ensure src directory is in Python path for imports
ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
RESULTS_DIR = ROOT_DIR / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Import prediction pipeline from src/predict.py
try:
    from predict import predict_review
except Exception as e:
    st.error(f"Failed to load prediction pipeline from src/predict.py: {e}")

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIG & CUSTOM STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Customer Feedback Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for SaaS aesthetics
st.markdown("""
<style>
    /* Metric Card Customization */
    [data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 700 !important;
    }
    
    /* Sentiment Badges */
    .badge-positive {
        background-color: #2ecc71;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 16px;
    }
    .badge-negative {
        background-color: #e74c3c;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 16px;
    }
    .badge-neutral {
        background-color: #f1c40f;
        color: black;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 16px;
    }
    
    /* Result Cards */
    .result-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Cache dataset loading for speed
@st.cache_data
def load_aspect_summary():
    p = RESULTS_DIR / "aspect_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_data
def load_model_comparison():
    p = RESULTS_DIR / "model_comparison.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.title("🤖 Feedback Intelligence")
st.sidebar.caption("AI-Powered Customer Review Analytics")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🔍 Analyze Review", "📊 Analytics", "🧩 Aspect Insights", "ℹ️ About"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Specifications")
st.sidebar.info("""
**Deployment Model**: Logistic Regression  
**Features**: TF-IDF (20,000 N-Grams)  
**Macro F1-Score**: 67.04%  
**Accuracy**: 88.12%
""")

st.sidebar.caption("⚠️ Predictions are model-generated proxy estimates.")

# =========================================================
# PAGE 1 — OVERVIEW
# =========================================================
if page == "🏠 Overview":
    st.title("Customer Feedback Intelligence System")
    st.markdown("#### *Automated sentiment classification & aspect-level customer insights at scale.*")
    st.markdown("---")
    
    # KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews Analyzed", "393,579", help="Unique deduplicated Amazon Fine Food corpus")
    with col2:
        st.metric("Positive Sentiment", "77.94%", delta="306,758 reviews", delta_color="normal")
    with col3:
        st.metric("Neutral Sentiment", "7.56%", delta="29,754 reviews", delta_color="off")
    with col4:
        st.metric("Negative Sentiment", "14.50%", delta="57,067 reviews", delta_color="inverse")
        
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📊 Sentiment Distribution")
        sent_df = pd.DataFrame({
            'Sentiment': ['Positive', 'Negative', 'Neutral'],
            'Count': [306758, 57067, 29754]
        })
        fig_pie = px.pie(
            sent_df, values='Count', names='Sentiment',
            color='Sentiment',
            color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#f1c40f'},
            hole=0.4
        )
        fig_pie.update_layout(margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        st.subheader("⭐ Star Rating (Score) Distribution")
        score_df = pd.DataFrame({
            'Star Rating': ['1 Star', '2 Star', '3 Star', '4 Star', '5 Star'],
            'Count': [36275, 20792, 29754, 56042, 250716]
        })
        fig_bar = px.bar(
            score_df, x='Star Rating', y='Count',
            color='Star Rating',
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig_bar.update_layout(showlegend=False, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")
    st.subheader("💡 Key Executive Insights")
    st.success("🟢 **Dominant Product Strength**: Over 77.9% of customer reviews express high satisfaction, driven primarily by product **Taste** (26,474 positive mentions).")
    st.warning("⚠️ **Top Value Friction Point**: **Price / Value** exhibits the highest negative mention ratio (**14.13%** of mentions are negative), driven by shipping costs and portion sizes.")
    st.error("🔴 **Operational Risk**: **Packaging** represents the 2nd largest complaint source (**2,208 negative complaints**) due to damaged boxes and unsealed bags in transit.")

# =========================================================
# PAGE 2 — ANALYZE REVIEW (HERO FEATURE)
# =========================================================
elif page == "🔍 Analyze Review":
    st.title("🔍 Analyze Customer Review")
    st.markdown("Enter a customer review below to extract **Overall Sentiment**, **Probability Confidence**, and **Aspect-Level Insights**.")
    
    st.markdown("##### ⚡ Quick Test Examples (Click to populate):")
    example_cols = st.columns(4)
    
    selected_example = ""
    if example_cols[0].button("🟢 Positive Sample"):
        selected_example = "The dog food quality is amazing and my Labrador loves it! Great price and fast delivery."
    if example_cols[1].button("🔴 Negative Sample"):
        selected_example = "Product arrived damaged, box was unsealed, and the peanuts were stale and rancid. Terrible experience."
    if example_cols[2].button("🔀 Mixed Sentiment"):
        selected_example = "The taste is excellent and delicious, but the delivery was late and packaging was crushed."
    if example_cols[3].button("🚫 Negation Sample"):
        selected_example = "This coffee is not worth the price and definitely not fresh."
        
    review_input = st.text_area(
        "Paste Customer Review Text:",
        value=selected_example if selected_example else "",
        height=140,
        placeholder="Paste a customer review here..."
    )
    
    if st.button("🔍 Analyze Review", type="primary"):
        if not review_input or len(review_input.strip()) == 0:
            st.warning("Please enter or paste a valid review text to analyze.")
        else:
            with st.spinner("Analyzing text with pre-trained TF-IDF + Logistic Regression & ABSA pipeline..."):
                result = predict_review(review_input)
                
            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown("---")
                st.subheader("📋 Prediction Results")
                
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    sent = result['sentiment']
                    conf = result['confidence'] * 100
                    
                    if sent == "Positive":
                        st.markdown(f"### Overall Sentiment: <span class='badge-positive'>🟢 POSITIVE</span>", unsafe_allow_html=True)
                    elif sent == "Negative":
                        st.markdown(f"### Overall Sentiment: <span class='badge-negative'>🔴 NEGATIVE</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"### Overall Sentiment: <span class='badge-neutral'>🟡 NEUTRAL</span>", unsafe_allow_html=True)
                        
                    st.markdown(f"**Model Confidence**: **{conf:.2f}%**")
                    st.progress(result['confidence'])
                    st.caption("Model probability derived from Logistic Regression `predict_proba()`.")
                    
                with res_col2:
                    st.markdown("##### Class Probability Distribution")
                    probs_df = pd.DataFrame(list(result['probabilities'].items()), columns=['Class', 'Probability'])
                    fig_prob = px.bar(
                        probs_df, x='Probability', y='Class', orientation='h',
                        color='Class',
                        color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#f1c40f'},
                        text_auto='.2%'
                    )
                    fig_prob.update_layout(height=180, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                # Aspect Insights Section
                st.markdown("---")
                st.subheader("🧩 Aspect-Level Feedback Insights")
                
                aspects = result['aspects']
                if not aspects:
                    st.info("No explicit domain aspect keywords detected in this review.")
                else:
                    asp_data = []
                    for asp in aspects:
                        sent_tag = "🟢 Positive" if asp['sentiment'] == 'Positive' else ("🔴 Negative" if asp['sentiment'] == 'Negative' else "🟡 Neutral")
                        asp_data.append({
                            "Aspect Category": asp['aspect'],
                            "Aspect Sentiment": sent_tag,
                            "Relevant Sentence Clause": asp['matching_clause']
                        })
                    st.table(pd.DataFrame(asp_data))
                    
                with st.expander("🔍 View Preprocessing Details"):
                    st.write("**Original Text**:", result['original_text'])
                    st.write("**Cleaned Text** (Passed to TF-IDF):", result['clean_text'])

# =========================================================
# PAGE 3 — ANALYTICS
# =========================================================
elif page == "📊 Analytics":
    st.title("📊 Business Analytics Dashboard")
    st.markdown("Historical analytics derived from **393,579 unique customer reviews** and model benchmarks.")
    st.markdown("---")
    
    summary_df = load_aspect_summary()
    comp_df = load_model_comparison()
    
    if summary_df is not None:
        st.subheader("🧩 Aspect Mention Frequencies across Corpus")
        fig_aspect_freq = px.bar(
            summary_df, x='Total Mentions', y='Aspect',
            orientation='h', color='Aspect',
            color_discrete_sequence=px.colors.qualitative.Plotly,
            text_auto=',d'
        )
        fig_aspect_freq.update_layout(showlegend=False, margin=dict(t=20, b=0, l=0, r=0))
        st.plotly_chart(fig_aspect_freq, use_container_width=True)
        
        st.markdown("---")
        st.subheader("⚖️ Aspect Sentiment Breakdown (Positive vs Neutral vs Negative)")
        
        aspect_melted = summary_df.melt(
            id_vars=['Aspect'],
            value_vars=['Positive Mentions', 'Neutral Mentions', 'Negative Mentions'],
            var_name='Sentiment Type', value_name='Count'
        )
        aspect_melted['Sentiment Type'] = aspect_melted['Sentiment Type'].str.replace(' Mentions', '')
        
        fig_aspect_sent = px.bar(
            aspect_melted, x='Count', y='Aspect', color='Sentiment Type',
            orientation='h',
            color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#f1c40f'},
            barmode='stack'
        )
        st.plotly_chart(fig_aspect_sent, use_container_width=True)
        
    if comp_df is not None:
        st.markdown("---")
        st.subheader("🤖 Model Benchmarks Comparison (Phase 8 Results)")
        st.dataframe(comp_df, use_container_width=True)

# =========================================================
# PAGE 4 — ASPECT INSIGHTS
# =========================================================
elif page == "🧩 Aspect Insights":
    st.title("🧩 Deep Dive Aspect Insights")
    st.markdown("Filter and inspect specific feedback categories to identify root causes.")
    st.markdown("---")
    
    summary_df = load_aspect_summary()
    
    if summary_df is not None:
        selected_aspect = st.selectbox("Select Feedback Aspect Category:", summary_df['Aspect'].unique())
        
        aspect_row = summary_df[summary_df['Aspect'] == selected_aspect].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Aspect Mentions", f"{aspect_row['Total Mentions']:,}")
        col2.metric("Positive Mentions", f"{aspect_row['Positive Mentions']:,}")
        col3.metric("Negative Mentions", f"{aspect_row['Negative Mentions']:,}")
        col4.metric("Negative Ratio (%)", f"{aspect_row['Negative Mention Ratio (%)']}%")
        
        st.markdown("---")
        st.markdown(f"#### Business Assessment for **{selected_aspect}**:")
        neg_ratio = aspect_row['Negative Mention Ratio (%)']
        if neg_ratio > 12.0:
            st.error(f"🔴 **HIGH PRIORITY ACTION ITEM**: {selected_aspect} has a high negative complaint ratio ({neg_ratio}%). Customer feedback requires immediate operational investigation.")
        elif neg_ratio > 7.0:
            st.warning(f"⚠️ **MODERATE PRIORITY**: {selected_aspect} shows moderate negative feedback ({neg_ratio}%). Monitor fulfillment and supplier quality.")
        else:
            st.success(f"🟢 **HEALTHY DIMENSION**: {selected_aspect} demonstrates strong positive customer satisfaction ({neg_ratio}% negative ratio).")

# =========================================================
# PAGE 5 — ABOUT
# =========================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Project Overview
    The **AI-Powered Customer Feedback Intelligence System** is an end-to-end Machine Learning and Natural Language Processing (NLP) pipeline designed to automatically analyze large volumes of customer review text, classify overall sentiment, and extract aspect-level feedback insights.

    ---

    ### 🛠️ Technology Stack
    * **Core Language**: Python 3.11
    * **Data Wrangling**: Pandas, NumPy
    * **NLP Feature Extraction**: Scikit-Learn `TfidfVectorizer` (20,000 Unigrams + Bigrams)
    * **Machine Learning Model**: Logistic Regression (`max_iter=1000`)
    * **Visualization & UI**: Streamlit, Plotly Express
    * **Persistence**: Joblib

    ---

    ### 📊 Dataset & Target Label Creation
    * **Data Source**: Amazon Fine Food Reviews Dataset (1999–2012 historical customer review corpus).
    * **Deduplicated Samples**: **393,579 unique customer reviews**.
    * **Target Label Mapping**:
      * `Score 1–2` $\to$ **Negative** (14.50%)
      * `Score 3` $\to$ **Neutral** (7.56%)
      * `Score 4–5` $\to$ **Positive** (77.94%)

    > ⚠️ **Documentation Note**: Sentiment labels are **project-defined proxy labels** derived from 1–5 star ratings. They are not human-annotated sentiment labels.
    """)
    
    st.markdown("---")
    st.caption("Developed as a Portfolio Project | AI-Powered Customer Feedback Intelligence System")
