"""
Text document loader for the RAG system.
"""
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from langchain.document_loaders import TextLoader
from langchain.schema import Document

class TXTLoader:
    """Loader for text documents."""
    
    def __init__(self, use_langchain: bool = True, encoding: str = "utf-8"):
        """
        Initialize the text loader.
        
        Args:
            use_langchain: Whether to use LangChain's loader or a custom implementation
            encoding: Text encoding to use
        """
        self.use_langchain = use_langchain
        self.encoding = encoding
    
    def load(self, file_path: str) -> List[Document]:
        """
        Load a text document.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of Document objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        if self.use_langchain:
            return self._load_with_langchain(file_path)
        else:
            return self._load_custom(file_path)
    
    def _load_with_langchain(self, file_path: str) -> List[Document]:
        """Load text using LangChain's loader."""
        try:
            loader = TextLoader(file_path, encoding=self.encoding)
            documents = loader.load()
            
            # Add additional metadata
            for doc in documents:
                doc.metadata["file_path"] = file_path
                doc.metadata["file_type"] = "txt"
                doc.metadata["file_name"] = os.path.basename(file_path)
            
            return documents
        except UnicodeDecodeError:
            # Try with different encodings if UTF-8 fails
            for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    loader = TextLoader(file_path, encoding=encoding)
                    documents = loader.load()
                    
                    # Add additional metadata
                    for doc in documents:
                        doc.metadata["file_path"] = file_path
                        doc.metadata["file_type"] = "txt"
                        doc.metadata["file_name"] = os.path.basename(file_path)
                        doc.metadata["encoding"] = encoding
                    
                    return documents
                except:
                    continue
            
            # If all encodings fail, raise the original error
            raise
    
    def _load_custom(self, file_path: str) -> List[Document]:
        """Custom text loading implementation."""
        documents = []
        
        try:
            # Try with the specified encoding first
            encodings_to_try = [self.encoding]
            
            # Add fallback encodings
            if self.encoding.lower() != "utf-8":
                encodings_to_try.append("utf-8")
            encodings_to_try.extend(["latin-1", "cp1252", "iso-8859-1"])
            
            text = None
            used_encoding = None
            
            for encoding in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        text = file.read()
                    used_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                raise UnicodeDecodeError(f"Failed to decode {file_path} with any encoding")
            
            # Extract metadata
            metadata = {
                "file_path": file_path,
                "file_type": "txt",
                "file_name": os.path.basename(file_path),
                "encoding": used_encoding,
                "size_bytes": os.path.getsize(file_path)
            }
            
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata=metadata
                ))
        
        except Exception as e:
            raise Exception(f"Error loading text file {file_path}: {e}")
        
        return documents