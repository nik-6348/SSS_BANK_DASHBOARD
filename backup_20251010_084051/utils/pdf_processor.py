from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
from typing import List
from langchain.schema import Document


class PDFProcessor:
    """Handles PDF loading and text extraction for bank statements."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_pdf(self, uploaded_file) -> List[Document]:
        """
        Process uploaded PDF file and extract text chunks.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            List of Document objects containing text chunks
        """
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "source": uploaded_file.name,
                    "chunk": i,
                    "total_chunks": len(chunks)
                })
            
            return chunks
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    def extract_bank_statement_info(self, text: str) -> dict:
        """
        Extract key information from bank statement text.
        
        Args:
            text: Raw text from bank statement
            
        Returns:
            Dictionary with extracted information
        """
        # This is a simple extraction - can be enhanced with regex patterns
        info = {
            "has_transactions": "transaction" in text.lower() or "debit" in text.lower() or "credit" in text.lower(),
            "has_balance": "balance" in text.lower(),
            "has_dates": any(month in text.lower() for month in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"])
        }
        return info
