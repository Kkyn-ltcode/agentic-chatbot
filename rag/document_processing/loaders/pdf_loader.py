"""
PDF document loader for the RAG system.
"""
import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pypdf
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

class PDFLoader:
    """Loader for PDF documents."""
    
    def __init__(self, use_langchain: bool = True):
        """
        Initialize the PDF loader.
        
        Args:
            use_langchain: Whether to use LangChain's loader or a custom implementation
        """
        self.use_langchain = use_langchain
    
    def load(self, file_path: str) -> List[Document]:
        """
        Load a PDF document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        if self.use_langchain:
            return self._load_with_langchain(file_path)
        else:
            return self._load_custom(file_path)
    
    def _load_with_langchain(self, file_path: str) -> List[Document]:
        """Load PDF using LangChain's loader."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Add additional metadata
        for doc in documents:
            doc.metadata["file_path"] = file_path
            doc.metadata["file_type"] = "pdf"
            doc.metadata["file_name"] = os.path.basename(file_path)
        
        return documents
    
    def _load_custom(self, file_path: str) -> List[Document]:
        """Custom PDF loading implementation."""
        documents = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf = pypdf.PdfReader(file)
                
                # Extract metadata from the PDF
                pdf_metadata = pdf.metadata
                metadata = {
                    "file_path": file_path,
                    "file_type": "pdf",
                    "file_name": os.path.basename(file_path),
                    "total_pages": len(pdf.pages)
                }
                
                # Add PDF metadata if available
                if pdf_metadata:
                    for key, value in pdf_metadata.items():
                        if key.startswith('/'):
                            key = key[1:]  # Remove leading slash
                        metadata[f"pdf_{key.lower()}"] = str(value)
                
                # Extract text from each page
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text.strip():
                        page_metadata = metadata.copy()
                        page_metadata["page"] = i + 1
                        
                        documents.append(Document(
                            page_content=text,
                            metadata=page_metadata
                        ))
        
        except Exception as e:
            raise Exception(f"Error loading PDF {file_path}: {e}")
        
        return documents