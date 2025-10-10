import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
import os
import json
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_pdf(uploaded_pdf, api_key: str = None):
    """
    Parse bank statement PDF using AI-powered extraction.
    Much more intelligent than rule-based parsing!
    """
    if not api_key:
        logger.warning("No API key provided, falling back to document mode")
        return pd.DataFrame(columns=['Date', 'Type', 'Category', 'Amount', 'Description'])
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Extract raw text from PDF
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        
        # Combine all text from all pages
        full_text = "\n".join([page.page_content for page in pages])
        
        # Use AI to extract transactions
        transactions = _extract_transactions_with_ai(full_text, api_key)
        
        if not transactions:
            logger.info("No transactions found by AI, returning empty DataFrame")
            return pd.DataFrame(columns=['Date', 'Type', 'Category', 'Amount', 'Description'])
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Clean and validate the DataFrame
        if not df.empty:
            # Ensure required columns exist
            required_columns = ['Date', 'Type', 'Category', 'Amount', 'Description']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Convert Date column to datetime
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Convert Amount to numeric
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            
            # Fill missing values
            df['Type'] = df['Type'].fillna('unknown')
            df['Category'] = df['Category'].fillna('uncategorized')
            df['Description'] = df['Description'].fillna('')
            
            # Remove rows where both Date and Amount are invalid
            df = df.dropna(subset=['Date', 'Amount'], how='all')
        
        logger.info(f"AI extracted {len(df)} transactions successfully!")
        return df
        
    except Exception as e:
        logger.error(f"Error parsing PDF with AI: {e}")
        # Return empty DataFrame with proper structure
        return pd.DataFrame(columns=['Date', 'Type', 'Category', 'Amount', 'Description'])
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def _extract_transactions_with_ai(text: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Use Gemini AI to intelligently extract transaction data from bank statement text.
    Much smarter than regex patterns!
    """
    try:
        # Initialize Gemini
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0.1,  # Low temperature for consistent extraction
            convert_system_message_to_human=True
        )
        
        # Create a detailed prompt for transaction extraction
        extraction_prompt = f"""
You are a financial data extraction expert. Extract ALL transactions from this bank statement text.

INSTRUCTIONS:
1. Find ALL transactions (debits/credits) in the text
2. Extract: Date, Amount, Description, Type (debit/credit), and Category
3. For categories, use: food, transport, shopping, entertainment, utilities, healthcare, salary, transfer, or other
4. Return ONLY a valid JSON array, no other text
5. If no transactions found, return empty array []

REQUIRED JSON FORMAT:
[
  {{
    "Date": "YYYY-MM-DD",
    "Amount": number,
    "Description": "transaction description",
    "Type": "debit" or "credit", 
    "Category": "category name"
  }}
]

BANK STATEMENT TEXT:
{text[:8000]}  # Limit text to avoid token limits

RESPOND WITH ONLY THE JSON ARRAY:
"""

        # Get AI response
        response = llm.invoke(extraction_prompt)
        
        # Parse JSON response
        try:
            # Clean the response - remove any markdown formatting
            response_text = response.content.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            transactions = json.loads(response_text)
            
            # Validate and clean the transactions
            cleaned_transactions = []
            for tx in transactions:
                if isinstance(tx, dict) and all(key in tx for key in ['Date', 'Amount', 'Description', 'Type', 'Category']):
                    # Ensure amount is numeric
                    try:
                        tx['Amount'] = float(tx['Amount'])
                        cleaned_transactions.append(tx)
                    except (ValueError, TypeError):
                        continue
            
            logger.info(f"AI successfully extracted {len(cleaned_transactions)} transactions")
            return cleaned_transactions
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.error(f"AI Response: {response.content}")
            return []
            
    except Exception as e:
        logger.error(f"Error in AI extraction: {e}")
        return []
