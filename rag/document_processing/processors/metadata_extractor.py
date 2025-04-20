"""
Metadata extraction for documents.
"""
import re
import os
from typing import Dict, List, Any, Optional
from langchain.schema import Document
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("metadata_extractor")

class MetadataExtractor:
    """Extract and enhance document metadata."""
    
    def __init__(self, 
                 extract_title: bool = True,
                 extract_dates: bool = True,
                 extract_authors: bool = True,
                 extract_keywords: bool = False):
        """
        Initialize the metadata extractor.
        
        Args:
            extract_title: Whether to extract document title
            extract_dates: Whether to extract dates from document
            extract_authors: Whether to extract author information
            extract_keywords: Whether to extract keywords
        """
        self.extract_title = extract_title
        self.extract_dates = extract_dates
        self.extract_authors = extract_authors
        self.extract_keywords = extract_keywords
    
    def extract_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Extract metadata from documents.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of documents with enhanced metadata
        """
        processed_docs = []
        
        for doc in documents:
            # Create a copy of the metadata
            metadata = doc.metadata.copy()
            
            # Add processing timestamp
            metadata["processed_at"] = datetime.now().isoformat()
            
            # Extract title if enabled and not already present
            if self.extract_title and "title" not in metadata:
                title = self._extract_title(doc.page_content)
                if title:
                    metadata["title"] = title
            
            # Extract dates if enabled
            if self.extract_dates:
                dates = self._extract_dates(doc.page_content)
                if dates:
                    metadata["extracted_dates"] = dates
            
            # Extract authors if enabled and not already present
            if self.extract_authors and "author" not in metadata:
                authors = self._extract_authors(doc.page_content)
                if authors:
                    metadata["authors"] = authors
            
            # Extract keywords if enabled
            if self.extract_keywords:
                keywords = self._extract_keywords(doc.page_content)
                if keywords:
                    metadata["keywords"] = keywords
            
            # Create a new document with enhanced metadata
            processed_docs.append(Document(
                page_content=doc.page_content,
                metadata=metadata
            ))
        
        return processed_docs
    
    def _extract_title(self, text: str) -> Optional[str]:
        """
        Extract title from document text.
        
        Args:
            text: Document text
            
        Returns:
            Extracted title or None
        """
        # Try to find title in the first few lines
        lines = text.split('\n')
        for i in range(min(5, len(lines))):
            line = lines[i].strip()
            if line and len(line) < 100 and not line.startswith('#') and not line.startswith('http'):
                return line
        
        # Try to find markdown or HTML headings
        title_patterns = [
            r'^# (.+)$',  # Markdown h1
            r'<h1.*?>(.*?)</h1>',  # HTML h1
            r'<title.*?>(.*?)</title>'  # HTML title
        ]
        
        for pattern in title_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                return matches[0].strip()
        
        return None
    
    def _extract_dates(self, text: str) -> List[str]:
        """
        Extract dates from document text.
        
        Args:
            text: Document text
            
        Returns:
            List of extracted dates
        """
        # Common date patterns
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY/MM/DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
            r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b'  # DD Month YYYY
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return list(set(dates))  # Remove duplicates
    
    def _extract_authors(self, text: str) -> List[str]:
        """
        Extract author information from document text.
        
        Args:
            text: Document text
            
        Returns:
            List of extracted authors
        """
        # Look for common author patterns
        author_patterns = [
            r'(?:Author|By|Written by)[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)',
            r'(?:©|Copyright)(?:\s+\d{4})?\s+by\s+([A-Z][a-z]+ [A-Z][a-z]+)'
        ]
        
        authors = []
        for pattern in author_patterns:
            matches = re.findall(pattern, text)
            authors.extend(matches)
        
        return list(set(authors))  # Remove duplicates
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from document text.
        
        Args:
            text: Document text
            
        Returns:
            List of extracted keywords
        """
        # Look for keyword sections
        keyword_patterns = [
            r'(?:Keywords|Tags|Key terms)[:\s]+((?:[a-zA-Z]+(?:,\s*|$))+)',
        ]
        
        keywords = []
        for pattern in keyword_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Split by comma and clean up
                keywords.extend([k.strip() for k in match.split(',') if k.strip()])
        
        return list(set(keywords))  # Remove duplicates