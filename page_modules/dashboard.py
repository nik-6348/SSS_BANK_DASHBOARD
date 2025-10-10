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
    # st.header("📊 Dashboard & Search")
    
    # Check if we have transaction data
    if st.session_state.transactions.empty:
        # Show message to upload to knowledge base first
        st.warning("📋 **No Transaction Data Available**")
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
    from langchain_chroma import Chroma
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
    """Process natural language queries using Gemini AI for intelligent filtering."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    if not API_KEY:
        st.error("API Key not configured for AI-powered search.")
        return pd.DataFrame()
    
    try:
        # Use Gemini AI to understand the query and generate filtering criteria
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=API_KEY,
            temperature=0.1
        )
        
        # Create a detailed prompt for the AI to understand the query
        ai_prompt = f"""
        You are a financial data analyst. I have a DataFrame with columns: Date, Type (debit/credit), Category, Amount, Description.
        
        User Query: "{query}"
        
        Based on this query, I need you to return a JSON response with filtering criteria. The response should be in this exact format:
        
        {{
            "filters": {{
                "date_range": {{"start": "YYYY-MM-DD or null", "end": "YYYY-MM-DD or null"}},
                "amount_range": {{"min": "number or null", "max": "number or null"}},
                "transaction_types": ["debit", "credit"] or ["debit"] or ["credit"] or [],
                "categories": ["category1", "category2"] or [],
                "description_keywords": ["keyword1", "keyword2"] or [],
                "amount_conditions": {{"above": "number or null", "below": "number or null", "exactly": "number or null"}}
            }},
            "explanation": "Brief explanation of what the query is looking for"
        }}
        
        Examples:
        - "transactions above 5000" -> {{"amount_conditions": {{"above": 5000}}}}
        - "food expenses in december" -> {{"categories": ["food"], "date_range": {{"start": "2024-12-01", "end": "2024-12-31"}}}}
        - "all debit transactions" -> {{"transaction_types": ["debit"]}}
        
        Only return the JSON, no other text.
        """
        
        with st.spinner("🤖 AI is analyzing your query..."):
            ai_response = llm.invoke(ai_prompt)
        
        # Parse AI response
        import json
        try:
            # Extract JSON from response
            response_text = ai_response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            criteria = json.loads(response_text)
            filters = criteria.get("filters", {})
            explanation = criteria.get("explanation", "AI-powered search")
            
            st.info(f"🤖 AI Analysis: {explanation}")
            
        except json.JSONDecodeError:
            st.warning("AI response could not be parsed. Using fallback search.")
            return _fallback_search(df, query)
        
        # Apply AI-generated filters
        filtered_df = df.copy()
        
        # Date range filter
        date_range = filters.get("date_range", {})
        if date_range.get("start") and date_range.get("end"):
            try:
                start_date = pd.to_datetime(date_range["start"])
                end_date = pd.to_datetime(date_range["end"])
                filtered_df = filtered_df[
                    (filtered_df['Date'] >= start_date) & 
                    (filtered_df['Date'] <= end_date)
                ]
            except Exception:
                pass
        
        # Amount range filter
        amount_range = filters.get("amount_range", {})
        if amount_range.get("min") is not None:
            filtered_df = filtered_df[filtered_df['Amount'] >= amount_range["min"]]
        if amount_range.get("max") is not None:
            filtered_df = filtered_df[filtered_df['Amount'] <= amount_range["max"]]
        
        # Transaction type filter
        transaction_types = filters.get("transaction_types", [])
        if transaction_types:
            filtered_df = filtered_df[filtered_df['Type'].isin(transaction_types)]
        
        # Category filter
        categories = filters.get("categories", [])
        if categories:
            filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
        
        # Description keywords filter
        keywords = filters.get("description_keywords", [])
        if keywords:
            keyword_mask = pd.Series([False] * len(filtered_df))
            for keyword in keywords:
                keyword_mask |= filtered_df['Description'].str.contains(keyword, case=False, na=False)
            filtered_df = filtered_df[keyword_mask]
        
        # Amount conditions
        amount_conditions = filters.get("amount_conditions", {})
        if amount_conditions.get("above") is not None:
            filtered_df = filtered_df[filtered_df['Amount'] > amount_conditions["above"]]
        if amount_conditions.get("below") is not None:
            filtered_df = filtered_df[filtered_df['Amount'] < amount_conditions["below"]]
        if amount_conditions.get("exactly") is not None:
            filtered_df = filtered_df[filtered_df['Amount'] == amount_conditions["exactly"]]
        
        return filtered_df
        
    except Exception as e:
        st.warning(f"AI search encountered an issue: {str(e)}. Using fallback search.")
        return _fallback_search(df, query)

def _fallback_search(df, query):
    """Fallback rule-based search when AI fails"""
    query_lower = query.lower()
    filtered_df = df.copy()
    
    # Simple keyword matching
    if 'debit' in query_lower:
        filtered_df = filtered_df[filtered_df['Type'] == 'debit']
    elif 'credit' in query_lower:
        filtered_df = filtered_df[filtered_df['Type'] == 'credit']
    
    # Amount patterns
    import re
    amount_patterns = [
        r'above\s+(\d+)', r'over\s+(\d+)', r'more\s+than\s+(\d+)',
        r'below\s+(\d+)', r'under\s+(\d+)', r'less\s+than\s+(\d+)'
    ]
    
    for pattern in amount_patterns:
        match = re.search(pattern, query_lower)
        if match:
            try:
                amount = float(match.group(1))
                if 'above' in pattern or 'over' in pattern or 'more' in pattern:
                    filtered_df = filtered_df[filtered_df['Amount'] > amount]
                else:
                    filtered_df = filtered_df[filtered_df['Amount'] < amount]
                break
            except (ValueError, AttributeError):
                continue
    
    return filtered_df
