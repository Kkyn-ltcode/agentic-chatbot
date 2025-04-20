"""
Document processing pipeline for the RAG system.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from datetime import datetime
import hashlib
from langchain.schema import Document

from .loaders.loader_factory import LoaderFactory
from .processors.chunking import DocumentChunker
from .processors.cleaning import DocumentCleaner
from .processors.metadata_extractor import MetadataExtractor
from .processors.language_detector import LanguageDetector
from .storage.document_store import DocumentStore
from .storage.metadata_db import MetadataDB
from .storage.version_control import DocumentVersionControl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("document_pipeline")

class DocumentProcessingPipeline:
    """Pipeline for processing documents."""
    
    def __init__(self, 
                 base_dir: str,
                 use_langchain: bool = True,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 chunking_method: str = "recursive",
                 enable_versioning: bool = True):
        """
        Initialize the document processing pipeline.
        
        Args:
            base_dir: Base directory for storing processed documents
            use_langchain: Whether to use LangChain's loaders
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            chunking_method: Method for chunking documents
            enable_versioning: Whether to enable document versioning
        """
        self.base_dir = base_dir
        self.use_langchain = use_langchain
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_method = chunking_method
        self.enable_versioning = enable_versioning
        
        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        # Initialize components
        self.loader_factory = LoaderFactory(use_langchain=use_langchain)
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_method=chunking_method
        )
        self.cleaner = DocumentCleaner()
        self.metadata_extractor = MetadataExtractor()
        self.language_detector = LanguageDetector()
        
        # Initialize storage components
        self.document_store = DocumentStore(os.path.join(base_dir, "document_store"))
        self.metadata_db = MetadataDB(os.path.join(base_dir, "metadata.db"))
        
        if enable_versioning:
            self.version_control = DocumentVersionControl(os.path.join(base_dir, "version_control"))
        else:
            self.version_control = None
        
        # Processing stats
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "processing_errors": 0
        }
    
    def process_document(self, 
                         source: str, 
                         doc_id: Optional[str] = None,
                         custom_metadata: Optional[Dict[str, Any]] = None,
                         skip_chunking: bool = False) -> List[str]:
        """
        Process a document through the pipeline.
        
        Args:
            source: Path to a file or a web URL
            doc_id: Optional document identifier
            custom_metadata: Optional custom metadata
            skip_chunking: Whether to skip document chunking
            
        Returns:
            List of document IDs for the processed chunks
        """
        try:
            logger.info(f"Processing document: {source}")
            
            # Generate a document ID if not provided
            if doc_id is None:
                doc_id = f"doc_{hashlib.md5(source.encode('utf-8')).hexdigest()[:10]}_{int(datetime.now().timestamp())}"
            
            # Step 1: Load the document
            documents = self.loader_factory.load(source)
            if not documents:
                logger.warning(f"No content loaded from {source}")
                return []
            
            logger.info(f"Loaded {len(documents)} document(s) from {source}")
            
            # Add custom metadata if provided
            if custom_metadata:
                for doc in documents:
                    doc.metadata.update(custom_metadata)
            
            # Step 2: Clean the documents
            documents = self.cleaner.clean_documents(documents)
            
            # Step 3: Detect language
            documents = self.language_detector.detect_language(documents)
            
            # Step 4: Extract metadata
            documents = self.metadata_extractor.extract_metadata(documents)
            
            # Step 5: Chunk the documents if needed
            if not skip_chunking:
                original_docs = documents.copy()
                documents = self.chunker.chunk_documents(documents)
                logger.info(f"Created {len(documents)} chunks from {len(original_docs)} original document(s)")
                
                # Store original documents in version control if enabled
                if self.enable_versioning and self.version_control:
                    for i, doc in enumerate(original_docs):
                        original_doc_id = f"{doc_id}_original_{i}"
                        self.version_control.add_version(original_doc_id, doc)
            
            # Step 6: Store the documents
            chunk_ids = []
            for i, doc in enumerate(documents):
                # Generate a chunk ID
                chunk_doc_id = f"{doc_id}_chunk_{i}" if not skip_chunking else doc_id
                
                # Add source information to metadata
                doc.metadata["source"] = source
                doc.metadata["parent_doc_id"] = doc_id
                
                # Store the document
                self.document_store.add_document(chunk_doc_id, doc)
                
                # Store metadata in the database
                self.metadata_db.add_document(chunk_doc_id, doc.metadata)
                
                # Store in version control if enabled
                if self.enable_versioning and self.version_control:
                    self.version_control.add_version(chunk_doc_id, doc)
                
                chunk_ids.append(chunk_doc_id)
            
            # Update stats
            self.stats["documents_processed"] += 1
            self.stats["chunks_created"] += len(documents)
            
            logger.info(f"Successfully processed document {source} into {len(chunk_ids)} chunks")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error processing document {source}: {e}", exc_info=True)
            self.stats["processing_errors"] += 1
            return []
    
    def process_documents(self, 
                          sources: List[str],
                          custom_metadata: Optional[Dict[str, Any]] = None,
                          skip_chunking: bool = False) -> Dict[str, List[str]]:
        """
        Process multiple documents through the pipeline.
        
        Args:
            sources: List of file paths or web URLs
            custom_metadata: Optional custom metadata
            skip_chunking: Whether to skip document chunking
            
        Returns:
            Dictionary mapping source to list of document IDs
        """
        results = {}
        
        for source in sources:
            doc_ids = self.process_document(
                source=source,
                custom_metadata=custom_metadata,
                skip_chunking=skip_chunking
            )
            results[source] = doc_ids
        
        return results
    
    def process_directory(self, 
                          directory: str,
                          recursive: bool = True,
                          file_extensions: Optional[List[str]] = None,
                          custom_metadata: Optional[Dict[str, Any]] = None,
                          skip_chunking: bool = False) -> Dict[str, List[str]]:
        """
        Process all documents in a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to process subdirectories
            file_extensions: List of file extensions to process
            custom_metadata: Optional custom metadata
            skip_chunking: Whether to skip document chunking
            
        Returns:
            Dictionary mapping source to list of document IDs
        """
        if not os.path.isdir(directory):
            logger.error(f"Directory not found: {directory}")
            return {}
        
        # Default file extensions if not provided
        if file_extensions is None:
            file_extensions = [".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm"]
        
        # Normalize extensions
        file_extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in file_extensions]
        
        # Find all files
        sources = []
        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    _, ext = os.path.splitext(file_path)
                    if ext.lower() in file_extensions:
                        sources.append(file_path)
        else:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(file_path)
                    if ext.lower() in file_extensions:
                        sources.append(file_path)
        
        logger.info(f"Found {len(sources)} files to process in {directory}")
        
        # Process all files
        return self.process_documents(
            sources=sources,
            custom_metadata=custom_metadata,
            skip_chunking=skip_chunking
        )
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document from the store.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document or None if not found
        """
        return self.document_store.get_document(doc_id)
    
    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Metadata dictionary or None if not found
        """
        return self.metadata_db.get_metadata(doc_id)
    
    def search_by_metadata(self, query: Dict[str, Any], limit: int = 100) -> List[str]:
        """
        Search for documents by metadata.
        
        Args:
            query: Metadata query dictionary
            limit: Maximum number of results
            
        Returns:
            List of matching document IDs
        """
        return self.metadata_db.search_metadata(query, limit)
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the system.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if successful, False otherwise
        """
        # Delete from document store
        doc_store_result = self.document_store.delete_document(doc_id)
        
        # Delete from metadata database
        metadata_result = self.metadata_db.delete_document(doc_id)
        
        # Delete from version control if enabled
        version_result = True
        if self.enable_versioning and self.version_control:
            version_result = self.version_control.delete_all_versions(doc_id)
        
        return doc_store_result and metadata_result and version_result
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get document processing statistics.
        
        Returns:
            Dictionary of processing statistics
        """
        return self.stats
    
    def clear_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "processing_errors": 0
        }