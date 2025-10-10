import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from utils.pdf_parser import parse_pdf

# LangChain + Gemini imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Bank Statement Dashboard", layout="wide")
st.title("🏦 Bank Statement AI Dashboard")

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

# -- Step 1: Upload PDF & Extract
st.header("🔼 Upload & Extract Transactions")
st.info("🤖 **AI-Powered Extraction**: Our system uses Gemini AI to intelligently extract transactions from any bank statement format - much smarter than traditional rule-based parsing!")
uploaded_files = st.file_uploader(
    "Upload Bank Statement PDFs", type=["pdf"], accept_multiple_files=True
)
if uploaded_files:
    total_df = pd.DataFrame()
    successful_uploads = 0
    
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
    else:
        st.info(f"Processed {len(uploaded_files)} file(s) but no structured transactions found. You can still use the RAG chatbot below to query the documents.")
else:
    st.info("Upload bank statement PDFs to analyze transactions or use the RAG chatbot if you have existing knowledge.")

# -- Step 2: Show, Search, Filter
if not st.session_state.transactions.empty:
    txns = st.session_state.transactions.copy()
    st.subheader("🔍 Transaction Search & Filter")

    # Simple filters
    col1, col2 = st.columns(2)
    with col1:
        # Handle case where all amounts are the same (including all zeros)
        min_amount = int(txns["Amount"].min())
        max_amount = int(txns["Amount"].max())
        
        if min_amount == max_amount:
            # If all amounts are the same, create a range around that value
            if min_amount == 0:
                # All amounts are zero, show a simple message
                st.info("All transactions have zero amounts")
                min_amt, max_amt = 0, 0
            else:
                # All amounts are the same non-zero value, create a small range
                min_amt, max_amt = st.slider("Amount Range", 
                    min_amount - 1, max_amount + 1, (min_amount, max_amount))
        else:
            # Normal case with different amounts
            min_amt, max_amt = st.slider("Amount Range", 
                min_amount, max_amount, (min_amount, max_amount))
    with col2:
        categories = ["All"] + sorted(txns["Category"].dropna().unique())
        cat = st.selectbox("Category", categories)
        if cat != "All":
            txns = txns[txns["Category"] == cat]
    
    # Apply amount filter only if we have a valid range
    if min_amt != max_amt or min_amt != 0:
        txns = txns[txns["Amount"].between(min_amt, max_amt)]

    # Only show analysis if we have meaningful transaction data
    if not txns.empty and txns["Amount"].sum() > 0:
        st.dataframe(txns.sort_values("Date", ascending=False), use_container_width=True)

        # -- Step 3: Aggregate Panels
        st.subheader("📊 Aggregates")
        st.metric("Total Spent", txns[txns["Type"] == "debit"]["Amount"].sum())
        st.metric("Total Received", txns[txns["Type"] == "credit"]["Amount"].sum())
        st.metric("Transactions", len(txns))

        # -- Step 4: Visualizations
        st.subheader("📈 Visualizations")
        
        # Timeline
        if len(txns) > 1:
            st.line_chart(txns.groupby("Date")["Amount"].sum())
        
        # Pie by category (only if we have multiple categories)
        category_totals = txns.groupby("Category")["Amount"].sum()
        if len(category_totals) > 1:
            st.write("Spending by Category")
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 6))
                category_totals.plot.pie(autopct='%1.1f%%', ax=ax)
                ax.set_ylabel('')  # Remove the default ylabel
                st.pyplot(fig)
                plt.close(fig)  # Clean up memory
            except ImportError:
                st.error("matplotlib not available for pie charts")
                st.write("Category breakdown:", category_totals.to_dict())
        else:
            st.write("Single category spending:", category_totals.index[0] if len(category_totals) > 0 else "N/A")
        
        # Bar by month (only if we have multiple months)
        if len(txns) > 1:
            txns_copy = txns.copy()
            txns_copy['month'] = txns_copy['Date'].dt.to_period('M')
            monthly_totals = txns_copy.groupby('month')['Amount'].sum()
            if len(monthly_totals) > 1:
                st.bar_chart(monthly_totals)
            else:
                st.write("Single month data available")
    else:
        st.info("No meaningful transaction data found. The uploaded files will be available for RAG queries below.")

# -- Step 5: RAG Chain Setup (after PDF or knowledgebase present)
st.header("🤖 RAG Chatbot & Knowledgebase")

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

# Allow initializing knowledgebase (by upload or load)
if uploaded_files and st.button("Initialize RAG Knowledgebase"):
    if not st.session_state.transactions.empty:
        init_rag_chain(st.session_state.transactions)
        st.success("Bank transactions loaded into knowledgebase!")
    else:
        # Initialize RAG with document content from PDFs
        try:
            from utils.pdf_processor import PDFProcessor
            processor = PDFProcessor()
            documents = []
            
            for pdf in uploaded_files:
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
                st.success(f"Loaded {len(documents)} document chunks into knowledgebase!")
            else:
                st.error("No document content could be extracted from the uploaded files.")
        except Exception as e:
            st.error(f"Error initializing RAG knowledgebase: {str(e)}")

elif st.session_state.vectorstore or st.session_state.rag_ready:
    st.session_state.rag_ready = True

if st.session_state.rag_ready:
    st.success("Knowledgebase ready! Ask finance questions below.")
    user_input = st.chat_input("Ask about your bank statement, e.g. 'Show all travel transactions over 5000'")
    if user_input:
        # Prepare chat history
        chat_history = []
        for msg in st.session_state.messages:
            if msg["role"]=="user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))
        # Run QA (simplified, you may want more structure!)
        retriever, prompt, llm = st.session_state.qa_chain
        try:
            # Get relevant documents
            docs = retriever.get_relevant_documents(user_input)
            # Format docs to string
            context = "\n".join([doc.page_content for doc in docs])
            
            # Create messages with chat history
            messages = [HumanMessage(content=user_input)]
            if chat_history:
                messages = chat_history + messages
            
            # Format the prompt
            formatted_prompt = prompt.format_messages(
                chat_history=chat_history,
                input=user_input
            )
            
            # Get response from LLM
            response = llm.invoke(formatted_prompt)
            answer = response
        except Exception as e:
            st.error(f"Error querying knowledgebase: {str(e)}")
            answer = type('obj', (object,), {'content': f"Sorry, I encountered an error: {str(e)}"})
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": answer.content})
        st.markdown(f"**Assistant:** {answer.content}")

    # Show chat history
    for msg in st.session_state.messages[-10:]:
        if msg["role"]=="user":
            st.markdown(f"🧑‍💼 **You:** {msg['content']}")
        else:
            st.markdown(f"🤖 **Assistant:** {msg['content']}")

else:
    st.info("Please upload a PDF and initialize, or load an existing knowledgebase.")

# Footer
st.caption("All data stored in-memory only. Add SQL DB in the next step!")
