import streamlit as st
import os
import shutil
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils.pdf_processor import PDFProcessor
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")


def save_transactions_backup():
    """Save transactions to CSV backup"""
    if not st.session_state.transactions.empty:
        st.session_state.transactions.to_csv("transactions_backup.csv", index=False)


def init_rag_automatically(documents):
    """Auto-initialize RAG system with documents"""
    if not API_KEY:
        st.error(
            "API Key not configured. Please set GOOGLE_API_KEY in your environment."
        )
        return False

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=API_KEY
        )
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=API_KEY,
            temperature=0.3,
            convert_system_message_to_human=True,
        )

        # Create or update vectorstore
        if hasattr(st.session_state, "vectorstore") and st.session_state.vectorstore:
            # Add to existing vectorstore
            st.session_state.vectorstore.add_documents(documents)
        else:
            # Create new vectorstore
            db = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory="./chroma_db",
                collection_name="bank_statements",
            )

            # Setup QA chain
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant for analyzing bank statements. Use the provided context to answer questions about financial transactions, balances, and account activity.",
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ]
            )

            st.session_state.vectorstore = db
            st.session_state.qa_chain = (
                db.as_retriever(search_kwargs={"k": 5}),
                prompt,
                llm,
            )

        st.session_state.rag_ready = True
        return True

    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return False


def render_knowledge_base():
    """Render the enhanced knowledge base management page"""
    st.header("🧠 Knowledge Base Management")

    # Show current transaction data
    if not st.session_state.transactions.empty:
        st.subheader("📊 Current Transaction Data")
        st.write(f"**Total Transactions:** {len(st.session_state.transactions)}")

        # Show sample transactions
        with st.expander("👁️ View Sample Transactions", expanded=False):
            sample_txns = st.session_state.transactions.head(10)
            st.dataframe(sample_txns, width="stretch")

            if len(st.session_state.transactions) > 10:
                st.write(
                    f"... and {len(st.session_state.transactions) - 10} more transactions"
                )

        st.markdown("---")

    # Main CRUD Operations
    st.subheader("📁 Document Management")

    # Add New Documents Section
    with st.expander("➕ Add New Documents", expanded=True):
        st.write("Upload PDF documents to add to your knowledge base:")

        new_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            key="kb_file_upload",
            help="Select one or more PDF files to add to the knowledge base",
        )

        if new_files:
            st.write(f"**Selected {len(new_files)} file(s):**")
            for file in new_files:
                st.write(f"📄 {file.name} ({file.size:,} bytes)")

            if st.button("🚀 Add to Knowledge Base", type="primary"):
                if not API_KEY:
                    st.error(
                        "❌ API Key not configured. Please set GOOGLE_API_KEY in your environment."
                    )
                else:
                    with st.spinner("Processing documents..."):
                        try:
                            # Process new documents for both RAG and transaction extraction
                            processor = PDFProcessor()
                            new_documents = []
                            all_transactions = pd.DataFrame()

                            for pdf in new_files:
                                # Extract documents for RAG
                                docs = processor.process_pdf(pdf)
                                new_documents.extend(docs)

                                # Extract transactions using AI
                                from utils.pdf_parser import parse_pdf

                                transactions_df = parse_pdf(pdf, API_KEY)
                                if not transactions_df.empty:
                                    all_transactions = pd.concat(
                                        [all_transactions, transactions_df],
                                        ignore_index=True,
                                    )

                            # Store transaction data
                            if not all_transactions.empty:
                                if not st.session_state.transactions.empty:
                                    # Merge with existing transactions
                                    st.session_state.transactions = pd.concat(
                                        [
                                            st.session_state.transactions,
                                            all_transactions,
                                        ],
                                        ignore_index=True,
                                    )
                                else:
                                    # Set as new transactions
                                    st.session_state.transactions = all_transactions

                                # Save transactions backup
                                save_transactions_backup()
                                st.success(
                                    f"✅ Extracted {len(all_transactions)} transactions from {len(new_files)} file(s)!"
                                )

                            # Initialize RAG system
                            if new_documents:
                                success = init_rag_automatically(new_documents)
                                if success:
                                    st.success(
                                        f"✅ Successfully added {len(new_documents)} document chunks to knowledge base!"
                                    )
                                    st.rerun()
                            else:
                                st.error(
                                    "No document content could be extracted from the uploaded files."
                                )

                        except Exception as e:
                            st.error(
                                f"Error adding documents to knowledge base: {str(e)}"
                            )

    # View Documents Section
    with st.expander("👁️ View Documents"):
        if hasattr(st.session_state, "vectorstore") and st.session_state.vectorstore:
            try:
                collection = st.session_state.vectorstore._collection
                results = collection.get()

                if results["ids"]:
                    st.write(
                        f"**📂 Knowledge Base Contents ({len(results['ids'])} documents):**"
                    )

                    # Display documents in a more organized way
                    for i, doc_id in enumerate(results["ids"]):
                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 1])

                            with col1:
                                st.write(f"**Document {i + 1}:**")
                                content_preview = (
                                    results["documents"][i][:200] + "..."
                                    if len(results["documents"][i]) > 200
                                    else results["documents"][i]
                                )
                                st.write(content_preview)

                            with col2:
                                if st.button("🗑️ Delete", key=f"del_{doc_id}"):
                                    try:
                                        collection.delete([doc_id])
                                        st.success("✅ Document deleted!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error deleting document: {e}")

                            with col3:
                                st.write(f"ID: {doc_id[:8]}...")

                            st.markdown("---")
                else:
                    st.info("No documents found in the knowledge base.")
            except Exception as e:
                st.error(f"Error accessing knowledge base: {e}")
        else:
            st.info("Knowledge base not initialized. Add documents to get started.")

    # Delete All Documents Section
    with st.expander("🗑️ Delete All Documents", expanded=False):
        st.warning("⚠️ This action cannot be undone!")
        st.write("This will permanently delete all documents from the knowledge base.")

        if st.button("🗑️ Clear All Documents", type="secondary"):
            if (
                hasattr(st.session_state, "vectorstore")
                and st.session_state.vectorstore
            ):
                try:
                    collection = st.session_state.vectorstore._collection
                    collection.delete()

                    # Reset session state
                    st.session_state.vectorstore = None
                    st.session_state.qa_chain = None
                    st.session_state.rag_ready = False

                    # Remove ChromaDB directory
                    if os.path.exists("./chroma_db"):
                        shutil.rmtree("./chroma_db")

                    st.success("✅ All documents cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing documents: {e}")
            else:
                st.info("No documents to clear.")

    st.markdown("---")

    # System Information
    st.subheader("📊 System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Configuration:**")
        st.write(f"• API Key: {'✅ Configured' if API_KEY else '❌ Missing'}")
        st.write(
            f"• Knowledge Base: {'✅ Ready' if st.session_state.rag_ready else '⏳ Not initialized'}"
        )
        st.write("• Embedding Model: models/embedding-001")
        st.write("• LLM Model: gemini-2.0-flash-exp")
        st.write("• Vector Store: ChromaDB")

    with col2:
        st.write("**Status:**")
        st.write("• PDF Processing: ✅ AI-Powered")
        st.write("• Document Storage: ✅ Persistent")
        st.write("• Search Capability: ✅ Semantic")
        st.write("• Chat History: ✅ Enabled")

    # Auto-save transactions when they change
    if not st.session_state.transactions.empty:
        save_transactions_backup()
