import streamlit as st

# Custom CSS for modern UI
st.markdown(
    """
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
    .chat-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        max-height: 70vh;
        overflow-y: auto;
    }
    
    /* Enhanced loading states */
    .loading-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #4F46E5;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin-right: 1rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Fix for stable text in loading states */
    .stSpinner {
        animation: none !important;
    }
    
    .stSpinner > div {
        animation: none !important;
    }
    
    /* Ensure text doesn't rotate */
    .loading-text {
        animation: none !important;
        transform: none !important;
    }
    
    /* Responsive quick question buttons */
    .quick-question-btn {
        width: 100%;
        margin: 0.25rem 0;
        padding: 0.75rem;
        border-radius: 8px;
        border: 2px solid #e9ecef;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        transition: all 0.3s ease;
        font-size: 0.9rem;
        text-align: center;
    }
    
    .quick-question-btn:hover {
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
    }
    
    /* Mobile responsive design */
    @media (max-width: 768px) {
        .chat-message {
            padding: 1rem;
            margin-bottom: 0.75rem;
        }
        
        .chat-container {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .quick-question-btn {
            font-size: 0.8rem;
            padding: 0.5rem;
        }
        
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1rem;
        }
    }
    
    /* Tablet responsive design */
    @media (max-width: 1024px) and (min-width: 769px) {
        .chat-container {
            padding: 1.25rem;
        }
        
        .quick-question-btn {
            font-size: 0.85rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


def display_chat_message(role: str, content: str, sources=None):
    """Display a chat message with styling."""
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🤖"

    st.markdown(
        f"""
    <div class="chat-message {css_class}">
        <div style="font-weight: 600; margin-bottom: 0.5rem;">{icon} {role.title()}</div>
        <div>{content}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Display sources if available
    if sources and role == "assistant":
        with st.expander("📚 View Sources", expanded=False):
            for i, doc in enumerate(sources[:3], 1):
                st.markdown(
                    f"""
                <div class="source-doc">
                    <strong>Source {i}:</strong> {doc.metadata.get("source", "Unknown")} 
                    (Page {doc.metadata.get("page", "N/A")}, Chunk {doc.metadata.get("chunk", "N/A")})
                    <br><em>{doc.page_content[:200]}...</em>
                </div>
                """,
                    unsafe_allow_html=True,
                )


def render_chatbot():
    """Render the enhanced AI chatbot page"""
    # Modern header like chatbot_ui.py
    st.markdown('<div class="main-header">🏦 AI Financial Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about your bank statements and financial data using AI-powered analysis</div>', unsafe_allow_html=True)

    # Check if we have data to work with
    if not st.session_state.rag_ready and st.session_state.transactions.empty:
        st.warning("⚠️ **No Data Available**")
        st.info("""
        **Please set up your knowledge base first!**
        
        To get started:
        1. Go to **🧠 Knowledge Base** page in the sidebar
        2. Add your PDF documents there
        3. Come back here to chat with your data
        
        The AI assistant needs your documents to answer questions about your financial data.
        """)

        # Quick action button to go to Knowledge Base
        if st.button("🚀 Go to Knowledge Base", type="primary"):
            st.session_state.current_page = "Knowledge Base"
            st.rerun()

        return

    # Quick question buttons with improved layout
    st.markdown("### 🚀 Quick Questions")
    
    # Responsive quick questions layout
    quick_questions = [
        "What are my biggest expenses?",
        "Show me food transactions", 
        "How much did I spend this month?",
        "Find transactions above ₹5000",
        "Summarize my financial activity",
    ]
    
    # Use responsive columns based on screen size
    if len(quick_questions) <= 3:
        cols = st.columns(len(quick_questions))
    else:
        # For mobile, show 2 columns, for desktop show 3
        cols = st.columns(3)
    
    for i, question in enumerate(quick_questions):
        col_idx = i % len(cols)
        with cols[col_idx]:
            if st.button(
                question, 
                key=f"quick_{i}",
                use_container_width=True
            ):
                # Directly process the quick question
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()

    # Chat input with enhanced styling and better placeholder
    placeholder_text = "Ask about your bank statement, e.g. 'Show all travel transactions over ₹5000'"
    
    # Create a container for the chat input with better styling
    st.markdown("### 💬 Ask a Question")
    user_input = st.chat_input(
        placeholder_text, 
        key="chat_input"
    )

    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Process with RAG if available, otherwise use transaction data
        try:
            if st.session_state.rag_ready and hasattr(st.session_state, 'rag_engine') and st.session_state.rag_engine:
                # Use RAG system with proper integration
                with st.spinner("🤔 AI is analyzing your bank documents and preparing a response..."):
                    try:
                        # Prepare chat history for RAG engine
                        chat_history = []
                        for msg in st.session_state.messages[-6:]:  # Last 6 messages for context
                            chat_history.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })
                        
                        # Query the RAG engine
                        answer, sources = st.session_state.rag_engine.query(user_input, chat_history)
                        
                        # Add assistant response to chat with sources
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error generating response: {str(e)}")
                        st.session_state.messages.append(
                            {
                                "role": "assistant", 
                                "content": "Sorry, I encountered an error while processing your request. Please try again or rephrase your question."
                            }
                        )

            elif not st.session_state.transactions.empty:
                # Use transaction data directly with AI
                from langchain_google_genai import ChatGoogleGenerativeAI
                from dotenv import load_dotenv
                import os

                load_dotenv()
                api_key = os.getenv("GOOGLE_API_KEY")

                if api_key:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash-exp",
                        google_api_key=api_key,
                        temperature=0.3,
                    )

                    # Create context from transaction data
                    txns_summary = st.session_state.transactions.describe()
                    sample_txns = st.session_state.transactions.head(10).to_string()

                    transaction_prompt = f"""
                    You are a financial assistant. Based on the following transaction data, answer the user's question.
                    
                    Transaction Summary:
                    {txns_summary}
                    
                    Sample Transactions:
                    {sample_txns}
                    
                    User Question: {user_input}
                    
                    Provide a helpful response based on the transaction data. If you need more specific information, suggest what the user should ask.
                    """

                    with st.spinner("🤔 AI is analyzing your transaction data and generating insights..."):
                        try:
                            response = llm.invoke(transaction_prompt)
                            
                            st.session_state.messages.append(
                                {"role": "assistant", "content": response.content}
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error analyzing transaction data: {str(e)}")
                            st.session_state.messages.append(
                                {
                                    "role": "assistant", 
                                    "content": "Sorry, I encountered an error while analyzing your transaction data. Please try again."
                                }
                            )
                else:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "I can see your transaction data, but I need an API key to provide AI-powered responses. Please configure your GOOGLE_API_KEY.",
                        }
                    )
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "I don't have any data to work with. Please upload your bank statements to the Knowledge Base first.",
                    }
                )

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Sorry, I encountered an error processing your request. Please try again.",
                }
            )

    # Display chat history with enhanced styling and responsive design
    if st.session_state.messages:
        st.markdown("---")
        st.markdown("### 💬 Chat History")
        
        # Add clear chat button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Clear Chat", help="Clear all chat messages"):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            st.info(f"📊 Showing {len(st.session_state.messages)} messages")

        # Chat container with scrolling and responsive design
        chat_container = st.container()
        with chat_container:
            # Show messages in chronological order (oldest first) like chatbot_ui.py
            for message in st.session_state.messages[-10:]:  # Show last 10 messages
                display_chat_message(
                    message["role"], message["content"], message.get("sources")
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748B; font-size: 0.875rem;">
        Powered by LangChain, Google Gemini 2.0, ChromaDB & Streamlit | Built with ❤️ for secure bank statement analysis
    </div>
    """, unsafe_allow_html=True)
