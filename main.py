import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Import page modules
from pages import dashboard, visuals, chatbot, knowledge_base, settings

# Load environment
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Enhanced page config with better styling
st.set_page_config(
    page_title="🏦 Singaji Bank AI Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏦"
)

# Enhanced CSS for modern, attractive UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Navigation Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 3px solid #667eea;
    }
    
    .nav-item {
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        font-weight: 500;
    }
    
    .nav-item:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .nav-item.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #4c63d2;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Search Box */
    .search-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #dee2e6;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* Chat Container */
    .chat-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background-color: #28a745;
        box-shadow: 0 0 10px rgba(40, 167, 69, 0.5);
    }
    
    .status-inactive {
        background-color: #dc3545;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background: linear-gradient(135deg, #f0f2f6 0%, #e9ecef 100%);
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* File Uploader */
    .stFileUploader > div {
        border: 2px dashed #667eea;
        border-radius: 12px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Plotly Charts */
    .js-plotly-plot {
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
    }
    
    /* Loading Spinner */
    .stSpinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 2s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header
st.markdown("""
<div class="main-header">
    <h1>🏦 Singaji Bank AI Dashboard</h1>
    <p>🤖 AI-Powered Financial Intelligence & Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "transactions" not in st.session_state:
    st.session_state.transactions = pd.DataFrame()
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"

# Enhanced Sidebar Navigation
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    st.markdown("---")
    
    # Navigation items with enhanced styling
    nav_items = [
        ("📊 Dashboard", "Dashboard"),
        ("📈 Visualizations", "Visuals"),
        ("🤖 AI Chatbot", "Chatbot"),
        ("🧠 Knowledge Base", "Knowledge Base"),
        ("⚙️ Settings", "Settings")
    ]
    
    # Create navigation buttons
    for icon_text, page_key in nav_items:
        is_active = st.session_state.current_page == page_key
        button_style = "active" if is_active else ""
        
        if st.button(
            icon_text, 
            key=f"nav_{page_key}",
            use_container_width=True,
            type="primary" if is_active else "secondary"
        ):
            st.session_state.current_page = page_key
            st.rerun()
    
    st.markdown("---")
    
    # Knowledge Base Status with enhanced styling
    st.markdown("### 🧠 Knowledge Base")
    if st.session_state.rag_ready:
        st.markdown("""
        <div class="status-indicator status-active"></div>
        <span style="color: #28a745; font-weight: 500;">✅ Active</span>
        """, unsafe_allow_html=True)
        
        if not st.session_state.transactions.empty:
            st.markdown(f"📊 **{len(st.session_state.transactions)}** transactions loaded")
    else:
        st.markdown("""
        <div class="status-indicator status-inactive"></div>
        <span style="color: #dc3545; font-weight: 500;">⏳ Not initialized</span>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats with enhanced styling
    if not st.session_state.transactions.empty:
        st.markdown("### 📈 Quick Stats")
        
        txns = st.session_state.transactions
        total_spent = txns[txns["Type"] == "debit"]["Amount"].sum()
        total_received = txns[txns["Type"] == "credit"]["Amount"].sum()
        
        st.metric("💰 Total Spent", f"₹{total_spent:,.2f}")
        st.metric("💵 Total Received", f"₹{total_received:,.2f}")
        st.metric("📝 Transactions", len(txns))
        
        # Balance calculation
        balance = total_received - total_spent
        balance_color = "normal" if balance >= 0 else "inverse"
        st.metric("💳 Net Balance", f"₹{balance:,.2f}", delta=None)

# Main Content Area with enhanced routing
if st.session_state.current_page == "Dashboard":
    dashboard.render_dashboard()
elif st.session_state.current_page == "Visuals":
    visuals.render_visuals()
elif st.session_state.current_page == "Chatbot":
    chatbot.render_chatbot()
elif st.session_state.current_page == "Knowledge Base":
    knowledge_base.render_knowledge_base()
elif st.session_state.current_page == "Settings":
    settings.render_settings()

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 12px; margin-top: 2rem;">
    <h4 style="margin: 0; color: #667eea;">🏦 Singaji Bank AI Dashboard</h4>
    <p style="margin: 0.5rem 0; font-size: 0.9rem;">
        🤖 Powered by Gemini AI | 💡 Built with Streamlit | 🔒 Secure & Private
    </p>
    <p style="margin: 0; font-size: 0.8rem; opacity: 0.7;">
        <em>All data is processed securely and stored locally on your device.</em>
    </p>
</div>
""", unsafe_allow_html=True)
