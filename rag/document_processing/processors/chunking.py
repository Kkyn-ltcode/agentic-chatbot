"""
Document chunking strategies for the RAG system.
"""
from typing import Dict, List, Optional, Any, Callable
import re
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

class DocumentChunker:
    """Chunking strategies for documents."""
    
    def __init__(self, 
                 chunk_size: int = 512, 
                 chunk_overlap: int = 50,
                 chunking_method: str = "recursive"):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            chunking_method: Method to use for chunking (recursive, character, token)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_method = chunking_method
        
        # Initialize the appropriate text splitter
        if chunking_method == "recursive":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        elif chunking_method == "character":
            self.text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            )
        elif chunking_method == "token":
            self.text_splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            raise ValueError(f"Unsupported chunking method: {chunking_method}")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        return self.text_splitter.split_documents(documents)
    
    def chunk_by_heading(self, documents: List[Document], heading_level: int = 2) -> List[Document]:
        """
        Split documents by headings.
        
        Args:
            documents: List of documents to chunk
            heading_level: Maximum heading level to split on (1 = h1, 2 = h2, etc.)
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata
            
            # Define regex patterns for different heading formats
            markdown_pattern = r'^#{1,' + str(heading_level) + r'}\s+(.+)$'
            html_pattern = r'<h[1-' + str(heading_level) + r'](.*?)>(.*?)</h[1-' + str(heading_level) + r']>'
            
            # Find all headings
            markdown_headings = re.finditer(markdown_pattern, content, re.MULTILINE)
            html_headings = re.finditer(html_pattern, content, re.MULTILINE | re.DOTALL)
            
            # Combine and sort headings by position
            headings = []
            for match in markdown_headings:
                headings.append((match.start(), match.group(0), match.group(1)))
            for match in html_headings:
                headings.append((match.start(), match.group(0), match.group(2)))
            
            headings.sort()
            
            if not headings:
                # If no headings found, use the original document
                chunked_docs.append(doc)
                continue
            
            # Split content by headings
            for i, (start, heading_text, heading_title) in enumerate(headings):
                # Determine the end position
                end = headings[i+1][0] if i < len(headings) - 1 else len(content)
                
                # Extract the section content
                section_content = content[start:end]
                
                # Create a new document for this section
                section_metadata = metadata.copy()
                section_metadata["heading"] = heading_title.strip()
                section_metadata["section"] = i + 1
                
                chunked_docs.append(Document(
                    page_content=section_content,
                    metadata=section_metadata
                ))
        
        return chunked_docs
    
    def chunk_by_paragraph(self, documents: List[Document]) -> List[Document]:
        """
        Split documents by paragraphs.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata
            
            # Split by double newlines (paragraphs)
            paragraphs = re.split(r'\n\s*\n', content)
            
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    para_metadata = metadata.copy()
                    para_metadata["paragraph"] = i + 1
                    
                    chunked_docs.append(Document(
                        page_content=paragraph,
                        metadata=para_metadata
                    ))
        
        return chunked_docs
    
    def chunk_with_custom_strategy(self, 
                                  documents: List[Document], 
                                  strategy: Callable[[str], List[str]]) -> List[Document]:
        """
        Split documents using a custom strategy.
        
        Args:
            documents: List of documents to chunk
            strategy: Function that takes a string and returns a list of chunks
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata
            
            # Apply the custom chunking strategy
            chunks = strategy(content)
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk"] = i + 1
                    
                    chunked_docs.append(Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    ))
        
        return chunked_docs