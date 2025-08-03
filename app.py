import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# ====== PAGE CONFIGURATION ======
st.set_page_config(
    page_title="Customer Segmentation & Recommendation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== CUSTOM CSS ======
st.markdown("""
<style>
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
[data-testid="stSidebar"] .stRadio > div > label {
    background-color: transparent;
    padding: 0.5rem 1rem;
    margin: 0.25rem 0;
    border-radius: 0.5rem;
    border: none;
    color: black;
    font-weight: 500;
    transition: all 0.2s ease;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background-color: #fee2e2;
    color: #ef4444;
}
[data-testid="stSidebar"] .stRadio > div > label[data-selected="true"] {
    background-color: #ef4444;
    color: white !important;
}
.main-header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 1rem 0;
}
.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.5rem;
}
.form-container {
    background: #ffffff;
    border-radius: 1rem;
    padding: 2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    border: 1px solid #e2e8f0;
    margin: 1rem 0;
}
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background-color: #f8fafc !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 0.5rem !important;
    padding: 0.75rem !important;
    font-size: 1rem !important;
    color: #374151 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 0.5rem !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(239, 68, 68, 0.2) !important;
    width: 100% !important;
    margin-top: 1rem !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
    transform: translateY(-1px) !important;
}
.recommendation-container {
    background: #ffffff;
    border-radius: 1rem;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}
.recommendation-title {
    color: #1e293b;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    border-bottom: 2px solid #ef4444;
    padding-bottom: 0.5rem;
}
.recommendation-item {
    background: #f8fafc;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 0.5rem;
    border-left: 4px solid #ef4444;
    font-weight: 500;
    color: #374151;
}
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ====== UTILITY FUNCTIONS ======
def clean_name(name):
    return re.sub(r'\W+', '', name.strip().lower())

def get_similar_products(product_name, product_names_dict, similarity_df, top_n=5):
    target_cleaned = clean_name(product_name)
    stock_code = None
    for code, name in product_names_dict.items():
        if clean_name(name) == target_cleaned:
            stock_code = code
            break
    if stock_code is None:
        return ["❌ Product not found in catalog"]
    sim_scores = similarity_df[stock_code].sort_values(ascending=False)
    top_similar = sim_scores.iloc[1:top_n+1].index
    return [product_names_dict.get(code, code) for code in top_similar]

# ====== LOAD MODELS & DATA ======
@st.cache_resource
def load_all():
    try:
        kmeans_model = joblib.load("kmeans_model.pkl")
        scaler = joblib.load("scaler.pkl")
        product_similarity = joblib.load("product_similarity.pkl")
        product_names = joblib.load("product_names.pkl")
        df = pd.read_csv("cleaned_data.csv")
        return kmeans_model, scaler, product_similarity, product_names, df
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None, None, None, None

kmeans_model, scaler, product_similarity, product_names, df = load_all()

# ====== CLUSTERING FUNCTION ======
def predict_customer_segment(recency, frequency, monetary, model, scaler):
    rfm_input = pd.DataFrame([[recency, frequency, monetary]],
                             columns=["Recency", "Frequency", "Monetary"])
    rfm_log = np.log1p(rfm_input)
    rfm_log.columns = ["Recency_log", "Frequency_log", "Monetary_log"]
    rfm_scaled = scaler.transform(rfm_log)
    cluster = model.predict(rfm_scaled)[0]
    cluster_map = {
        0: 'Regular',
        1: 'High-Value',
        2: 'Occasional',
        3: 'At-Risk'
    }
    return cluster_map.get(cluster, f"Cluster {cluster}"), cluster

# ====== SIDEBAR NAVIGATION ======
st.sidebar.markdown("### 🏠 Navigation")
section = st.sidebar.radio(
    "",
    ["🏠 Home", "📊 Clustering", "🛍️ Recommendation"],
    key="navigation"
)

# ====== HOME TAB ======
if section == "🏠 Home":
    st.title("Customer Intelligence Platform")
    st.write("Advanced Customer Segmentation & Product Recommendation System")

# ====== CLUSTERING TAB ======
elif section == "📊 Clustering":
    st.header("Customer Segmentation")
    st.write("Enter customer purchase data to determine their segment")

    if kmeans_model is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            recency = st.number_input("Recency (days since last purchase)", min_value=0, value=325)
        with col2:
            frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=1)
        with col3:
            monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=765322.00, format="%.2f")

        if st.button("🔍 Predict Segment"):
            segment, cluster_number = predict_customer_segment(recency, frequency, monetary, kmeans_model, scaler)
            st.success(f"This customer belongs to: **{segment} Shopper** (Cluster {cluster_number})")
    else:
        st.error("❌ Model files could not be loaded. Please ensure all required files are present.")

# ====== RECOMMENDATION TAB ======
elif section == "🛍️ Recommendation":
    st.header("Product Recommender")
    st.write("Get personalized product recommendations based on similarity")

    if product_similarity is not None and product_names is not None:
        unique_product_names = sorted(list(set(product_names.values())))
        product_input = st.selectbox("🔍 Select a Product", unique_product_names)

        if st.button("🛍️ Get Recommendations"):
            if product_input.strip():
                similar_products = get_similar_products(product_input, product_names, product_similarity)
                if similar_products:
                    st.markdown("""
                        <div class="recommendation-container">
                        <div class="recommendation-title">📦 Recommended Products</div>
                    """, unsafe_allow_html=True)
                    for i, name in enumerate(similar_products, 1):
                        st.markdown(f"<div class='recommendation-item'>{i}. {name}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error("❌ Product not found. Please check the name and try again.")
            else:
                st.warning("⚠️ Please select a product name.")
    else:
        st.error("❌ Recommendation system could not be loaded. Please ensure all required files are present.")
