"""
Factory for document loaders.
"""
import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import mimetypes
from urllib.parse import urlparse

from langchain.schema import Document

from .pdf_loader import PDFLoader
from .docx_loader import DOCXLoader
from .text_loader import TXTLoader
from .web_loader import WebLoader

class LoaderFactory:
    """Factory for creating document loaders."""
    
    def __init__(self, use_langchain: bool = True):
        """
        Initialize the loader factory.
        
        Args:
            use_langchain: Whether to use LangChain's loaders or custom implementations
        """
        self.use_langchain = use_langchain
        self.loaders = {
            "pdf": PDFLoader(use_langchain),
            "docx": DOCXLoader(use_langchain),
            "txt": TXTLoader(use_langchain),
            "web": WebLoader(use_langchain)
        }
        
        # Initialize mimetypes
        mimetypes.init()
    
    def get_loader(self, source_type: str):
        """
        Get a loader for the specified source type.
        
        Args:
            source_type: Type of source (pdf, docx, txt, web)
            
        Returns:
            Appropriate loader
        """
        source_type = source_type.lower()
        if source_type not in self.loaders:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        return self.loaders[source_type]
    
    def load(self, source: str) -> List[Document]:
        """
        Load documents from a source.
        
        Args:
            source: Path to a file or a web URL
            
        Returns:
            List of Document objects
        """
        # Determine the source type
        if source.startswith(('http://', 'https://')):
            source_type = "web"
        else:
            # Get file extension
            _, ext = os.path.splitext(source)
            ext = ext.lstrip('.').lower()
            
            # Map extension to source type
            if ext == "pdf":
                source_type = "pdf"
            elif ext in ["docx", "doc"]:
                source_type = "docx"
            elif ext in ["txt", "text", "md", "markdown", "csv", "json", "xml", "html", "htm"]:
                source_type = "txt"
            else:
                # Try to determine type from mimetype
                mime_type, _ = mimetypes.guess_type(source)
                if mime_type:
                    if mime_type == "application/pdf":
                        source_type = "pdf"
                    elif mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                                      "application/msword"]:
                        source_type = "docx"
                    elif mime_type.startswith("text/"):
                        source_type = "txt"
                    else:
                        raise ValueError(f"Unsupported file type: {mime_type}")
                else:
                    raise ValueError(f"Unsupported file extension: {ext}")
        
        # Get the appropriate loader and load the document
        loader = self.get_loader(source_type)
        return loader.load(source)