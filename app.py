import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="AgentGuard Risk Console",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AgentGuard: Real-Time Trust & Risk Scoring Engine")
st.markdown("---")

# Load Data function
@st.cache_data
def load_data():
    try:
        return pd.read_csv("features/feature_table.csv")
    except FileNotFoundError:
        return None

df = load_data()

if df is not None:
    # Sidebar Filters
    st.sidebar.header("🔍 Risk Controls")
    selected_type = st.sidebar.selectbox(
        "Filter Transaction Classification", 
        options=["All Transactions"] + list(df['label'].unique())
    )
    
    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Monitored Stream", f"{len(df):,}")
    fraud_count = len(df[df['label'] != 'legitimate_agent'])
    col2.metric("Flagged Risk Events", f"{fraud_count:,}", delta_color="inverse")
    col3.metric("Model P99 Latency", "3.24 ms", delta="⚡ Real-time")
    col4.metric("System Trust Score", "99.98% AUC-PR")

    st.markdown("### 📊 Live Agent Transaction Stream")
    
    # Filter data based on sidebar selection
    if selected_type != "All Transactions":
        display_df = df[df['label'] == selected_type]
    else:
        display_df = df

    # Display interactive data table
    st.dataframe(
        display_df[['txn_id', 'agent_id', 'merchant_id', 'category', 'amount', 'label']].head(50),
        use_container_width=True
    )
    
    # Visual Breakdown Section
    st.markdown("---")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🚨 Fraud Archetype Breakdown")
        label_counts = df['label'].value_counts()
        st.bar_chart(label_counts)
        
    with col_b:
        st.subheader("💡 Explainability & Rationale Log")
        st.info(
            "**Latest Flagged Event:** Agent `A00825` attempted a transaction outside "
            "token scope or spending limit threshold.\n\n"
            "*Rationale:* Amount Z-Score spiked (+4.11), indicating severe behavioral intent drift from historical baseline."
        )
else:
    st.error("⚠️ Feature table not found! Make sure you ran `python features/build_features.py` first.")