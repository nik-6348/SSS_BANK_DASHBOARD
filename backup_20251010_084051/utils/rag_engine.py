from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Tuple, Dict
from langchain.schema import Document


class RAGEngine:
    """Handles RAG pipeline with ChromaDB and Google GenAI."""
    
    def __init__(self, api_key: str, persist_directory: str = "./chroma_db"):
        self.api_key = api_key
        self.persist_directory = persist_directory
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # Initialize LLM with latest Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        
        self.vectorstore = None
        self.qa_chain = None
    
    def create_vectorstore(self, documents: List[Document]):
        """
        Create or update ChromaDB vectorstore with documents.
        
        Args:
            documents: List of Document objects to index
        """
        if self.vectorstore is None:
            # Create new vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name="bank_statements"
            )
        else:
            # Add to existing vectorstore
            self.vectorstore.add_documents(documents)
        
        # Setup QA chain
        self._setup_qa_chain()
    
    def _setup_qa_chain(self):
        """Setup the retrieval QA chain with chat history support."""
        
        # Create prompt with chat history support
        system_prompt = """You are an AI assistant specialized in analyzing bank statements. 
Use the following pieces of context from the bank statement to answer the question. 
If you cannot find the answer in the context, say so clearly.

Always reference specific numbers, dates, and transaction details when available.

Context from bank statement:
{context}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Create document chain
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Create retrieval chain
        self.qa_chain = create_retrieval_chain(
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            combine_docs_chain=question_answer_chain
        )
    
    def query(self, question: str, chat_history: List[Dict[str, str]]) -> Tuple[str, List[Document]]:
        """
        Query the RAG system with a question and chat history.
        
        Args:
            question: User's question about the bank statement
            chat_history: List of previous messages in format [{"role": "user/assistant", "content": "..."}]
            
        Returns:
            Tuple of (answer, source_documents)
        """
        if self.qa_chain is None:
            raise ValueError("Vectorstore not initialized. Please upload a PDF first.")
        
        # Convert chat history to LangChain message format
        langchain_chat_history = []
        for message in chat_history:
            if message["role"] == "user":
                langchain_chat_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                langchain_chat_history.append(AIMessage(content=message["content"]))
        
        # Invoke the chain with chat history
        result = self.qa_chain.invoke({
            "input": question,
            "chat_history": langchain_chat_history
        })
        
        return result["answer"], result.get("context", [])
    
    def get_similar_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Get similar documents from vectorstore.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            return []
        
        return self.vectorstore.similarity_search(query, k=k)
