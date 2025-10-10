import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

def render_chatbot():
    """Render the enhanced AI chatbot page"""
    st.header("🤖 AI Financial Assistant")
    
    if not st.session_state.rag_ready:
        st.warning("Please upload bank statements and initialize the knowledge base first.")
        return
    
    st.success("✅ AI Assistant is ready! Ask me anything about your financial data.")
    
    # Enhanced chat interface
    st.markdown("""
    <div class="chat-container">
        <h4>💬 Chat with your financial data</h4>
        <p>Ask questions like:</p>
        <ul>
            <li>"What are my biggest expenses this month?"</li>
            <li>"Show me all food-related transactions"</li>
            <li>"How much did I spend on transport?"</li>
            <li>"What's my spending pattern for entertainment?"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat input with enhanced styling
    user_input = st.chat_input(
        "Ask about your bank statement, e.g. 'Show all travel transactions over ₹5000'",
        key="chat_input"
    )
    
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
    
    # Display chat history with enhanced styling
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages[-10:]:  # Show last 10 messages
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Chat controls
    if st.session_state.messages:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat History", type="secondary"):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("📊 Export Chat", type="secondary"):
                # Export chat history
                chat_text = "\n\n".join([
                    f"{msg['role'].upper()}: {msg['content']}" 
                    for msg in st.session_state.messages
                ])
                st.download_button(
                    label="Download Chat",
                    data=chat_text,
                    file_name="financial_chat_export.txt",
                    mime="text/plain"
                )
