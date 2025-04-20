"""
Document cleaning and normalization for the RAG system.
"""
from typing import Dict, List, Optional, Any
import re
import unicodedata
from langchain.schema import Document

class DocumentCleaner:
    """Cleaning and normalization for documents."""
    
    def __init__(self, 
                 remove_extra_whitespace: bool = True,
                 normalize_unicode: bool = True,
                 remove_urls: bool = False,
                 remove_email: bool = False,
                 lowercase: bool = False):
        """
        Initialize the document cleaner.
        
        Args:
            remove_extra_whitespace: Whether to remove extra whitespace
            normalize_unicode: Whether to normalize Unicode characters
            remove_urls: Whether to remove URLs
            remove_email: Whether to remove email addresses
            lowercase: Whether to convert text to lowercase
        """
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_unicode = normalize_unicode
        self.remove_urls = remove_urls
        self.remove_email = remove_email
        self.lowercase = lowercase
    
    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """
        Clean and normalize documents.
        
        Args:
            documents: List of documents to clean
            
        Returns:
            List of cleaned documents
        """
        cleaned_docs = []
        
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata
            
            # Apply cleaning operations
            cleaned_content = self._clean_text(content)
            
            cleaned_docs.append(Document(
                page_content=cleaned_content,
                metadata=metadata
            ))
        
        return cleaned_docs
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return text
        
        # Normalize Unicode if enabled
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Remove URLs if enabled
        if self.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses if enabled
        if self.remove_email:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace if enabled
        if self.remove_extra_whitespace:
            # Replace multiple spaces with a single space
            text = re.sub(r'\s+', ' ', text)
            # Remove spaces at the beginning and end of lines
            text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)
            # Remove multiple newlines
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Convert to lowercase if enabled
        if self.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def clean_vietnamese_text(self, text: str) -> str:
        """
        Clean and normalize Vietnamese text.
        
        Args:
            text: Vietnamese text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return text
        
        # Basic cleaning
        text = self._clean_text(text)
        
        # Vietnamese-specific normalization
        # Normalize Vietnamese diacritics
        text = unicodedata.normalize('NFC', text)
        
        # Try to use underthesea if available
        try:
            from underthesea import word_tokenize
            # Tokenize and rejoin to ensure proper spacing
            text = " ".join(word_tokenize(text))
        except ImportError:
            pass
        
        return text