import streamlit as st
from dotenv import load_dotenv
import os
from utils.pdf_processor import PDFProcessor
from utils.rag_engine import RAGEngine
from datetime import datetime

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Bank Statement RAG Chatbot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #EEF2FF;
        border-left: 4px solid #4F46E5;
    }
    .assistant-message {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
    }
    .source-doc {
        background-color: #F8FAFC;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.875rem;
        border: 1px solid #E2E8F0;
    }
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        font-weight: 600;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .sidebar-info {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "pdf_processor" not in st.session_state:
    st.session_state.pdf_processor = PDFProcessor()
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


def initialize_rag_engine():
    """Initialize RAG engine with API key."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ Google API Key not found. Please set it in .env file")
        st.stop()
    
    if st.session_state.rag_engine is None:
        st.session_state.rag_engine = RAGEngine(api_key)


def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files."""
    with st.spinner("📄 Processing PDF files..."):
        all_documents = []
        
        for uploaded_file in uploaded_files:
            try:
                # Process PDF
                documents = st.session_state.pdf_processor.process_pdf(uploaded_file)
                all_documents.extend(documents)
                
                if uploaded_file.name not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(uploaded_file.name)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        if all_documents:
            # Create vectorstore
            st.session_state.rag_engine.create_vectorstore(all_documents)
            st.session_state.documents_loaded = True
            st.success(f"✅ Successfully processed {len(uploaded_files)} file(s) with {len(all_documents)} chunks!")
        else:
            st.error("❌ No documents could be processed.")


def display_chat_message(role: str, content: str, sources=None):
    """Display a chat message with styling."""
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <div style="font-weight: 600; margin-bottom: 0.5rem;">{icon} {role.title()}</div>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display sources if available
    if sources and role == "assistant":
        with st.expander("📚 View Sources", expanded=False):
            for i, doc in enumerate(sources[:3], 1):
                st.markdown(f"""
                <div class="source-doc">
                    <strong>Source {i}:</strong> {doc.metadata.get('source', 'Unknown')} 
                    (Page {doc.metadata.get('page', 'N/A')}, Chunk {doc.metadata.get('chunk', 'N/A')})
                    <br><em>{doc.page_content[:200]}...</em>
                </div>
                """, unsafe_allow_html=True)


# Main UI
st.markdown('<div class="main-header">🏦 Bank Statement RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload your bank statements and ask questions using AI-powered analysis</div>', unsafe_allow_html=True)

# Initialize RAG engine
initialize_rag_engine()

# Sidebar
with st.sidebar:
    st.markdown("### 📋 Document Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Bank Statement PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more bank statement PDF files"
    )
    
    if uploaded_files:
        if st.button("🚀 Process Documents", width='stretch'):
            process_uploaded_files(uploaded_files)
    
    # Display loaded files
    if st.session_state.uploaded_files:
        st.markdown("#### ✅ Loaded Documents:")
        for file_name in st.session_state.uploaded_files:
            st.markdown(f"- {file_name}")
    
    st.markdown("---")
    
    # Clear conversation
    if st.button("🗑️ Clear Conversation", width='stretch'):
        st.session_state.messages = []
        if st.session_state.rag_engine:
            st.session_state.rag_engine.clear_memory()
        st.rerun()
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-info">
        <strong>💡 How to use:</strong><br>
        1. Upload bank statement PDF(s)<br>
        2. Click "Process Documents"<br>
        3. Ask questions about transactions, balances, dates, etc.<br>
        4. Get AI-powered answers with sources
    </div>
    """, unsafe_allow_html=True)
    
    # Example questions
    st.markdown("#### 🔍 Example Questions:")
    example_questions = [
        "What is my total balance?",
        "Show me all transactions above $500",
        "What was my largest expense?",
        "List all transactions in December",
        "What are my recurring payments?"
    ]
    
    for question in example_questions:
        if st.button(question, key=f"example_{question}", width='stretch'):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

# Main chat area
if not st.session_state.documents_loaded:
    st.info("👈 Please upload and process bank statement PDFs to start chatting!")
else:
    # Display chat messages
    for message in st.session_state.messages:
        display_chat_message(
            message["role"],
            message["content"],
            message.get("sources")
        )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your bank statement..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        display_chat_message("user", prompt)
        
        # Get AI response
        with st.spinner("🤔 Analyzing..."):
            try:
                answer, sources = st.session_state.rag_engine.query(prompt)
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
                # Display assistant message
                display_chat_message("assistant", answer, sources)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.875rem;">
    Powered by LangChain, Google Gemini 2.0, ChromaDB & Streamlit | Built with ❤️ for secure bank statement analysis
</div>
""", unsafe_allow_html=True)
