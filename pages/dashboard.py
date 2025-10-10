import streamlit as st
import pandas as pd
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from utils.pdf_parser import parse_pdf
import plotly.express as px
from utils.rag_engine import RAGEngine

# Load environment
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

def render_dashboard():
    """Render the enhanced dashboard page with integrated search"""
    st.header("📊 Dashboard & Search")
    
    # Check if knowledge base exists
    if not st.session_state.rag_ready or st.session_state.transactions.empty:
        # Show message to upload to knowledge base first
        st.warning("📋 **No Data Available**")
        st.info("""
        **Please upload bank statements to the Knowledge Base first!**
        
        To get started:
        1. Go to **🧠 Knowledge Base** page in the sidebar
        2. Upload your bank statement PDFs there
        3. Come back to Dashboard to view your data
        
        The Knowledge Base will automatically process and extract your transaction data.
        """)
        
        # Quick action button to go to Knowledge Base
        if st.button("🚀 Go to Knowledge Base", type="primary"):
            st.session_state.current_page = "Knowledge Base"
            st.rerun()
        
        return

    # Dashboard Summary - Only show if knowledge base exists
    st.subheader("📊 Quick Overview")
    txns = st.session_state.transactions.copy()
    
    # Enhanced metrics with better styling
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total Spent", f"₹{txns[txns['Type'] == 'debit']['Amount'].sum():,.2f}")
    with col2:
        st.metric("💵 Total Received", f"₹{txns[txns['Type'] == 'credit']['Amount'].sum():,.2f}")
    with col3:
        st.metric("📝 Transactions", len(txns))
    with col4:
        st.metric("📅 Date Range", f"{txns['Date'].min().strftime('%d %b')} - {txns['Date'].max().strftime('%d %b')}")
    
    # Quick visualizations
    col1, col2 = st.columns(2)
    with col1:
        # Spending by category
        category_totals = txns.groupby("Category")["Amount"].sum()
        if len(category_totals) > 1:
            fig = px.pie(values=category_totals.values, names=category_totals.index, 
                       title="Spending by Category")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly trend
        txns_copy = txns.copy()
        txns_copy['month'] = txns_copy['Date'].dt.to_period('M')
        monthly_totals = txns_copy.groupby('month')['Amount'].sum()
        if len(monthly_totals) > 1:
            fig = px.line(x=[str(m) for m in monthly_totals.index], 
                        y=monthly_totals.values,
                        title="Monthly Transaction Trend")
            fig.update_layout(xaxis_title="Month", yaxis_title="Amount")
            st.plotly_chart(fig, use_container_width=True)

    # Integrated Advanced Search Section
    if not st.session_state.transactions.empty:
        st.markdown("---")
        st.subheader("🔍 Advanced Search & Filter")
        
        # NLP Search Box
        st.markdown("""
        <div class="search-box">
        <strong>Try queries like:</strong><br>
        • "Show me all debit transactions from 1 Oct to 30 Nov"<br>
        • "Credit history from Dec 09 to Dec 20"<br>
        • "Transactions above ₹5000"<br>
        • "All food expenses this month"
        </div>
        """, unsafe_allow_html=True)
        
        nlp_query = st.text_input("🔍 Natural Language Query:", placeholder="Type your query here...")
        
        if nlp_query:
            filtered_txns = _process_nlp_query(st.session_state.transactions, nlp_query)
            if not filtered_txns.empty:
                st.success(f"Found {len(filtered_txns)} transactions matching your query")
                st.dataframe(filtered_txns, use_container_width=True)
            else:
                st.warning("No transactions found matching your query")
        
        # Advanced Filters
        st.subheader("🎛️ Advanced Filters")
        txns = st.session_state.transactions.copy()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Date range filter
            if not txns.empty:
                min_date = txns['Date'].min().date()
                max_date = txns['Date'].max().date()
                date_range = st.date_input("📅 Date Range", value=(min_date, max_date), 
                                         min_value=min_date, max_value=max_date)
                if len(date_range) == 2:
                    txns = txns[txns['Date'].dt.date.between(date_range[0], date_range[1])]
        
        with col2:
            # Amount range
            if not txns.empty:
                min_amount = txns["Amount"].min()
                max_amount = txns["Amount"].max()
                amount_range = st.slider("💰 Amount Range (₹)", min_amount, max_amount, (min_amount, max_amount))
                txns = txns[txns["Amount"].between(amount_range[0], amount_range[1])]
        
        with col3:
            # Category filter
            categories = ["All"] + sorted(txns["Category"].dropna().unique().tolist())
            selected_category = st.selectbox("📂 Category", categories)
            if selected_category != "All":
                txns = txns[txns["Category"] == selected_category]
        
        # Transaction type filter
        col1, col2 = st.columns(2)
        with col1:
            transaction_types = st.multiselect("💳 Transaction Types", 
                                             ["debit", "credit"], 
                                             default=["debit", "credit"])
            if transaction_types:
                txns = txns[txns["Type"].isin(transaction_types)]
        
        with col2:
            # Search in description
            search_desc = st.text_input("🔍 Search in Description", placeholder="Enter keywords...")
            if search_desc:
                txns = txns[txns["Description"].str.contains(search_desc, case=False, na=False)]
        
        # Display filtered results
        if not txns.empty:
            st.subheader(f"📋 Filtered Results ({len(txns)} transactions)")
            st.dataframe(txns.sort_values("Date", ascending=False), use_container_width=True)
        else:
            st.info("No transactions match the current filters")

def init_rag_automatically(df):
    """Auto-initialize RAG system with transaction data"""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.documents import Document
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=API_KEY
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True
    )
    
    # Convert rows to Documents
    docs = [Document(page_content=str(row)) for _, row in df.iterrows()]
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="./chroma_db")
    
    # Prompt with chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a sharp financial assistant! Answer using this transaction context."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    qa_chain = db.as_retriever(search_kwargs={"k":5}), prompt, llm
    st.session_state.vectorstore = db
    st.session_state.qa_chain = qa_chain
    st.session_state.rag_ready = True

def _process_nlp_query(df, query):
    """Process natural language queries to filter transactions using both rule-based and AI-powered search."""
    query_lower = query.lower()
    filtered_df = df.copy()
    
    # First try AI-powered similarity search if RAG is available
    if hasattr(st.session_state, 'rag_ready') and st.session_state.rag_ready:
        try:
            # Use AI to understand the query intent
            ai_query = f"Find transactions related to: {query}"
            similar_docs = st.session_state.vectorstore.similarity_search(ai_query, k=15)
            
            # Extract relevant transaction indices
            similar_indices = set()
            for doc in similar_docs:
                # Parse the document content to find matching transactions
                doc_content = doc.page_content.lower()
                
                # Match by description keywords
                for idx, row in df.iterrows():
                    desc_words = str(row['Description']).lower().split()
                    doc_words = doc_content.split()
                    
                    # Check for common words between description and AI result
                    common_words = set(desc_words) & set(doc_words)
                    if len(common_words) >= 2:  # At least 2 common words
                        similar_indices.add(idx)
            
            # If we found similar transactions through AI, use them
            if similar_indices:
                ai_filtered = df.iloc[list(similar_indices)]
                filtered_df = ai_filtered  # Start with AI results
                st.info(f"🤖 AI found {len(ai_filtered)} potentially relevant transactions")
        except Exception as e:
            st.warning(f"AI search encountered an issue: {str(e)}. Using rule-based filtering only.")
            filtered_df = df.copy()
    
    # Rule-based filtering (original logic)
    
    # Date range parsing
    date_patterns = [
        r'from (\w+ \d+) to (\w+ \d+)',
        r'(\w+ \d+) to (\w+ \d+)',
        r'between (\w+ \d+) and (\w+ \d+)',
        r'from (\d+ \w+) to (\d+ \w+)',
        r'(\d+ \w+) to (\d+ \w+)'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, query_lower)
        if match:
            try:
                start_str, end_str = match.groups()
                # Simple date parsing (you can make this more sophisticated)
                if 'oct' in start_str and 'nov' in end_str:
                    filtered_df = filtered_df[filtered_df['Date'].dt.month.isin([10, 11])]
                elif 'dec' in start_str and 'dec' in end_str:
                    filtered_df = filtered_df[filtered_df['Date'].dt.month == 12]
                break
            except Exception:
                continue
    
    # Transaction type parsing
    if 'debit' in query_lower:
        filtered_df = filtered_df[filtered_df['Type'] == 'debit']
    elif 'credit' in query_lower:
        filtered_df = filtered_df[filtered_df['Type'] == 'credit']
    
    # Amount parsing - Fixed regex patterns
    amount_patterns = [
        r'above\s+(\d+)',
        r'over\s+(\d+)',
        r'more\s+than\s+(\d+)',
        r'greater\s+than\s+(\d+)',
        r'(\d+)\s*\+',  # For patterns like "5000+"
        r'₹?(\d+)\s*above',
        r'₹?(\d+)\s*over'
    ]
    
    for pattern in amount_patterns:
        amount_match = re.search(pattern, query_lower)
        if amount_match:
            try:
                amount = float(amount_match.group(1))
                filtered_df = filtered_df[filtered_df['Amount'] > amount]
                break
            except (ValueError, AttributeError):
                continue
    
    # Below amount patterns
    below_patterns = [
        r'below\s+(\d+)',
        r'under\s+(\d+)',
        r'less\s+than\s+(\d+)',
        r'(\d+)\s*-',  # For patterns like "5000-"
        r'₹?(\d+)\s*below',
        r'₹?(\d+)\s*under'
    ]
    
    for pattern in below_patterns:
        amount_match = re.search(pattern, query_lower)
        if amount_match:
            try:
                amount = float(amount_match.group(1))
                filtered_df = filtered_df[filtered_df['Amount'] < amount]
                break
            except (ValueError, AttributeError):
                continue
    
    # Enhanced category parsing
    category_keywords = {
        'food': ['food', 'restaurant', 'dining', 'grocery', 'eat', 'meal', 'cafe', 'kitchen'],
        'transport': ['transport', 'travel', 'taxi', 'uber', 'bus', 'train', 'flight', 'petrol', 'fuel'],
        'shopping': ['shopping', 'mall', 'store', 'amazon', 'flipkart', 'purchase', 'buy'],
        'entertainment': ['entertainment', 'movie', 'cinema', 'game', 'netflix', 'spotify', 'music'],
        'utilities': ['utilities', 'electricity', 'water', 'gas', 'internet', 'phone', 'bill'],
        'healthcare': ['healthcare', 'hospital', 'doctor', 'medical', 'pharmacy', 'medicine'],
        'salary': ['salary', 'income', 'credit', 'deposit', 'bonus', 'payroll'],
        'transfer': ['transfer', 'sent', 'received', 'payment', 'loan', 'emi']
    }
    
    for category, keywords in category_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            filtered_df = filtered_df[filtered_df['Category'] == category]
            break
    
    return filtered_df
