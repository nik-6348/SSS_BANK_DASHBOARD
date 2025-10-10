import streamlit as st
import pandas as pd
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from utils.pdf_parser import parse_pdf
import plotly.express as px
# import plotly.graph_objects as go  # Not currently used
# from plotly.subplots import make_subplots  # Not currently used

# LangChain + Gemini imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Modern page config
st.set_page_config(
    page_title="🏦 Singaji Bank AI Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏦"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .search-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🏦 Singaji Bank AI Dashboard</h1>
    <p>🤖 AI-Powered Financial Intelligence & Analytics</p>
</div>
""", unsafe_allow_html=True)

# Session state setup
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
    st.session_state.current_page = "📊 Dashboard"
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

# RAG Functions
def init_rag_chain(df):
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
    from langchain_core.documents import Document
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

# Helper Functions
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

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Choose a page:",
    ["📊 Dashboard", "🔍 Advanced Search", "📈 Visualizations", "🤖 AI Chatbot", "⚙️ Settings"],
    index=["📊 Dashboard", "🔍 Advanced Search", "📈 Visualizations", "🤖 AI Chatbot", "⚙️ Settings"].index(st.session_state.current_page)
)

st.session_state.current_page = page

# Sidebar Knowledge Base Status
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Knowledge Base")
if st.session_state.rag_ready:
    st.sidebar.success("✅ Ready")
    if not st.session_state.transactions.empty:
        st.sidebar.info(f"📊 {len(st.session_state.transactions)} transactions loaded")
else:
    st.sidebar.warning("⏳ Not initialized")

# Sidebar Quick Stats
if not st.session_state.transactions.empty:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Quick Stats")
    total_spent = st.session_state.transactions[st.session_state.transactions["Type"] == "debit"]["Amount"].sum()
    total_received = st.session_state.transactions[st.session_state.transactions["Type"] == "credit"]["Amount"].sum()
    st.sidebar.metric("💰 Total Spent", f"₹{total_spent:,.2f}")
    st.sidebar.metric("💵 Total Received", f"₹{total_received:,.2f}")
    st.sidebar.metric("📝 Transactions", len(st.session_state.transactions))

# Main Content Area based on selected page
if page == "📊 Dashboard":
    st.header("📊 Dashboard Overview")
    
    # File Upload Section
    st.subheader("📁 Upload Bank Statements")
    st.info("🤖 **AI-Powered Extraction**: Our system uses Gemini AI to intelligently extract transactions from any bank statement format!")
    
    uploaded_files = st.file_uploader(
        "Upload Bank Statement PDFs", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload one or more bank statement PDFs for analysis"
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        total_df = pd.DataFrame()
        successful_uploads = 0
        
        with st.spinner("🤖 AI is processing your bank statements..."):
            for pdf in uploaded_files:
                try:
                    df = parse_pdf(pdf, API_KEY)  # Pass API key to AI-powered parser
                    if not df.empty:
                        total_df = pd.concat([total_df, df], ignore_index=True)
                        successful_uploads += 1
                    else:
                        st.warning(f"Could not extract structured data from {pdf.name}. The file will be available for RAG queries.")
                except Exception as e:
                    st.error(f"Error processing {pdf.name}: {str(e)}")
        
        st.session_state.transactions = total_df
        
        if not total_df.empty:
            st.success(f"🤖 AI extracted {len(total_df)} transactions from {successful_uploads} file(s)!")
            st.info("💡 Using AI-powered extraction - much smarter than traditional parsing!")
            
            # Show sample of extracted data
            st.subheader("📋 Sample Extracted Data")
            st.dataframe(total_df.head(10), use_container_width=True)
        else:
            st.info(f"Processed {len(uploaded_files)} file(s) but no structured transactions found. You can still use the RAG chatbot below to query the documents.")
    else:
        st.info("Upload bank statement PDFs to analyze transactions or use the RAG chatbot if you have existing knowledge.")

    # Dashboard Summary
    if not st.session_state.transactions.empty:
        st.subheader("📊 Quick Overview")
        txns = st.session_state.transactions.copy()
        
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

elif page == "🔍 Advanced Search":
    st.header("🔍 Advanced Search & Filter")
    
    if st.session_state.transactions.empty:
        st.warning("Please upload bank statements first to use advanced search.")
    else:
        # NLP Search Box
        st.subheader("🧠 Natural Language Search")
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

elif page == "📈 Visualizations":
    st.header("📈 Advanced Visualizations")
    
    if st.session_state.transactions.empty:
        st.warning("Please upload bank statements first to view visualizations.")
    else:
        txns = st.session_state.transactions.copy()
        
        # Visualization tabs
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["💰 Spending Analysis", "📊 Category Breakdown", "📅 Time Series", "🔍 Custom Analysis"])
        
        with viz_tab1:
            st.subheader("💰 Spending Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Debit vs Credit comparison
                type_totals = txns.groupby('Type')['Amount'].sum()
                fig = px.bar(x=type_totals.index, y=type_totals.values, 
                           title="Total Amount by Transaction Type",
                           color=type_totals.index,
                           color_discrete_map={'debit': '#ff6b6b', 'credit': '#4ecdc4'})
                fig.update_layout(xaxis_title="Transaction Type", yaxis_title="Amount (₹)")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Top transactions
                top_transactions = txns.nlargest(10, 'Amount')
                fig = px.bar(top_transactions, x='Amount', y='Description', 
                           orientation='h', title="Top 10 Transactions by Amount")
                fig.update_layout(yaxis_title="Description", xaxis_title="Amount (₹)")
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab2:
            st.subheader("📊 Category Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Category pie chart
                category_totals = txns.groupby('Category')['Amount'].sum()
                fig = px.pie(values=category_totals.values, names=category_totals.index,
                           title="Spending Distribution by Category")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Category bar chart
                fig = px.bar(x=category_totals.index, y=category_totals.values,
                           title="Amount by Category")
                fig.update_layout(
                    xaxis_title="Category", 
                    yaxis_title="Amount (₹)",
                    xaxis=dict(tickangle=45)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab3:
            st.subheader("📅 Time Series Analysis")
            
            # Daily spending trend
            daily_totals = txns.groupby(txns['Date'].dt.date)['Amount'].sum().reset_index()
            fig = px.line(daily_totals, x='Date', y='Amount', 
                         title="Daily Transaction Amount Trend")
            fig.update_layout(xaxis_title="Date", yaxis_title="Amount (₹)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly breakdown
            txns_copy = txns.copy()
            txns_copy['month'] = txns_copy['Date'].dt.to_period('M')
            monthly_totals = txns_copy.groupby('month')['Amount'].sum()
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(x=[str(m) for m in monthly_totals.index], y=monthly_totals.values,
                           title="Monthly Transaction Totals")
                fig.update_layout(xaxis_title="Month", yaxis_title="Amount (₹)")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(x=[str(m) for m in monthly_totals.index], y=monthly_totals.values,
                           title="Monthly Transaction Trend")
                fig.update_layout(xaxis_title="Month", yaxis_title="Amount (₹)")
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab4:
            st.subheader("🔍 Custom Analysis")
            
            # Interactive filters for custom analysis
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_categories = st.multiselect("Select Categories", 
                                                   txns['Category'].unique().tolist(),
                                                   default=txns['Category'].unique().tolist())
            
            with col2:
                selected_types = st.multiselect("Select Transaction Types",
                                              txns['Type'].unique().tolist(),
                                              default=txns['Type'].unique().tolist())
            
            with col3:
                min_amount = st.number_input("Minimum Amount", value=0.0)
            
            # Apply filters
            custom_txns = txns[
                (txns['Category'].isin(selected_categories)) &
                (txns['Type'].isin(selected_types)) &
                (txns['Amount'] >= min_amount)
            ]
            
            if not custom_txns.empty:
                # Custom visualization
                fig = px.scatter(custom_txns, x='Date', y='Amount', 
                               color='Category', size='Amount',
                               title="Custom Filtered Transactions",
                               hover_data=['Description', 'Type'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.subheader("📊 Custom Analysis Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Amount", f"₹{custom_txns['Amount'].sum():,.2f}")
                with col2:
                    st.metric("Average Amount", f"₹{custom_txns['Amount'].mean():,.2f}")
                with col3:
                    st.metric("Max Amount", f"₹{custom_txns['Amount'].max():,.2f}")
                with col4:
                    st.metric("Transaction Count", len(custom_txns))
            else:
                st.info("No transactions match the selected criteria")

elif page == "🤖 AI Chatbot":
    st.header("🤖 AI Financial Assistant")
    
    if not st.session_state.rag_ready:
        st.warning("Please upload bank statements and initialize the knowledge base first.")
    else:
        st.success("✅ AI Assistant is ready! Ask me anything about your financial data.")
        
        # Chat interface
        user_input = st.chat_input("Ask about your bank statement, e.g. 'Show all travel transactions over ₹5000'")
        
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Process with RAG
            try:
                retriever, prompt, llm = st.session_state.qa_chain
                
                # Prepare chat history
                chat_history = []
                for msg in st.session_state.messages[-6:]:  # Last 6 messages for context
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    else:
                        chat_history.append(AIMessage(content=msg["content"]))
                
                # Get relevant documents
                docs = retriever.get_relevant_documents(user_input)
                context = "\n".join([doc.page_content for doc in docs])
                
                # Format the prompt
                formatted_prompt = prompt.format_messages(
                    chat_history=chat_history,
                    input=user_input
                )
                
                # Get AI response
                response = llm.invoke(formatted_prompt)
                
                # Add assistant response to chat
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error processing your request."})
        
        # Display chat history
        for message in st.session_state.messages[-10:]:  # Show last 10 messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

elif page == "⚙️ Settings":
    st.header("⚙️ Settings & Configuration")
    
    st.subheader("🧠 Knowledge Base Management")
    
    # Check if we have uploaded files from the Dashboard page
    if st.session_state.uploaded_files and st.button("Initialize RAG Knowledge Base", type="primary"):
        if not st.session_state.transactions.empty:
            init_rag_chain(st.session_state.transactions)
            st.success("✅ Knowledge base initialized with transaction data!")
        else:
            # Initialize RAG with document content from PDFs
            try:
                from utils.pdf_processor import PDFProcessor
                processor = PDFProcessor()
                documents = []
                
                for pdf in st.session_state.uploaded_files:
                    docs = processor.process_pdf(pdf)
                    documents.extend(docs)
                
                if documents:
                    # Create vectorstore with documents
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
                    
                    from langchain_community.vectorstores import Chroma
                    db = Chroma.from_documents(
                        documents=documents, 
                        embedding=embeddings, 
                        persist_directory="./chroma_db",
                        collection_name="bank_statements"
                    )
                    
                    # Setup QA chain
                    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful assistant for analyzing bank statements. Use the provided context to answer questions about financial transactions, balances, and account activity."),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}")
                    ])
                    
                    st.session_state.vectorstore = db
                    st.session_state.qa_chain = (db.as_retriever(search_kwargs={"k":5}), prompt, llm)
                    st.session_state.rag_ready = True
                    st.success(f"✅ Loaded {len(documents)} document chunks into knowledge base!")
                else:
                    st.error("No document content could be extracted from the uploaded files.")
            except Exception as e:
                st.error(f"Error initializing RAG knowledge base: {str(e)}")
    
    st.subheader("🔧 Application Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Current Configuration:**")
        st.write(f"• API Key: {'✅ Configured' if API_KEY else '❌ Missing'}")
        st.write(f"• Knowledge Base: {'✅ Ready' if st.session_state.rag_ready else '⏳ Not initialized'}")
        st.write(f"• Transactions Loaded: {len(st.session_state.transactions) if not st.session_state.transactions.empty else 0}")
    
    with col2:
        st.info("**System Status:**")
        st.write(f"• PDF Processing: ✅ AI-Powered")
        st.write(f"• Visualizations: ✅ Interactive")
        st.write(f"• Search: ✅ NLP-Enabled")
    
    # Clear data option
    st.subheader("🗑️ Data Management")
    
    if st.button("Clear All Data", type="secondary"):
        st.session_state.transactions = pd.DataFrame()
        st.session_state.vectorstore = None
        st.session_state.qa_chain = None
        st.session_state.rag_ready = False
        st.session_state.messages = []
        st.session_state.uploaded_files = None
        st.success("✅ All data cleared successfully!")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🏦 <strong>Singaji Bank AI Dashboard</strong> | 🤖 Powered by Gemini AI | 💡 Built with Streamlit</p>
    <p><em>All data is processed securely and stored locally.</em></p>
</div>
""", unsafe_allow_html=True)
