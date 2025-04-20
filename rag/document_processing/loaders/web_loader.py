"""
Web content loader for the RAG system.
"""
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import html2text
from langchain.document_loaders import WebBaseLoader
from langchain.schema import Document
from urllib.parse import urlparse

class WebLoader:
    """Loader for web content."""
    
    def __init__(self, use_langchain: bool = True):
        """
        Initialize the web loader.
        
        Args:
            use_langchain: Whether to use LangChain's loader or a custom implementation
        """
        self.use_langchain = use_langchain
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_tables = False
    
    def load(self, url: str) -> List[Document]:
        """
        Load content from a web URL.
        
        Args:
            url: Web URL to load
            
        Returns:
            List of Document objects
        """
        if not url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL: {url}")
        
        if self.use_langchain:
            return self._load_with_langchain(url)
        else:
            return self._load_custom(url)
    
    def _load_with_langchain(self, url: str) -> List[Document]:
        """Load web content using LangChain's loader."""
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        # Add additional metadata
        for doc in documents:
            doc.metadata["url"] = url
            doc.metadata["file_type"] = "web"
            doc.metadata["domain"] = urlparse(url).netloc
        
        return documents
    
    def _load_custom(self, url: str) -> List[Document]:
        """Custom web content loading implementation."""
        documents = []
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Extract metadata
            metadata = {
                "url": url,
                "file_type": "web",
                "domain": urlparse(url).netloc,
                "status_code": response.status_code,
                "content_type": response.headers.get('Content-Type', '')
            }
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title and other metadata
            if soup.title:
                metadata["title"] = soup.title.string
            
            # Extract meta tags
            for meta in soup.find_all('meta'):
                name = meta.get('name') or meta.get('property')
                content = meta.get('content')
                if name and content:
                    metadata[f"meta_{name.lower().replace(':', '_')}"] = content
            
            # Extract main content
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Convert HTML to markdown
            text = self.html_converter.handle(str(soup))
            
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata=metadata
                ))
        
        except Exception as e:
            raise Exception(f"Error loading web content from {url}: {e}")
        
        return documents