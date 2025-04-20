"""
Document storage for the RAG system.
"""
import os
import json
import pickle
from typing import Dict, List, Optional, Any
from langchain.schema import Document
import logging
from datetime import datetime
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("document_store")

class DocumentStore:
    """Storage for processed documents."""
    
    def __init__(self, store_dir: str, use_pickle: bool = True):
        """
        Initialize the document store.
        
        Args:
            store_dir: Directory to store documents
            use_pickle: Whether to use pickle for document storage
        """
        self.store_dir = store_dir
        self.use_pickle = use_pickle
        
        # Create store directory if it doesn't exist
        os.makedirs(store_dir, exist_ok=True)
        
        # Create index file if it doesn't exist
        self.index_file = os.path.join(store_dir, "document_index.json")
        if not os.path.exists(self.index_file):
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
    
    def add_document(self, doc_id: str, document: Document) -> bool:
        """
        Add a document to the store.
        
        Args:
            doc_id: Document identifier
            document: Document to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create document directory
            doc_dir = os.path.join(self.store_dir, doc_id)
            os.makedirs(doc_dir, exist_ok=True)
            
            # Store document
            if self.use_pickle:
                doc_file = os.path.join(doc_dir, "document.pkl")
                with open(doc_file, 'wb') as f:
                    pickle.dump(document, f)
            else:
                # Store as JSON (note: this might lose some Document functionality)
                doc_file = os.path.join(doc_dir, "document.json")
                with open(doc_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "page_content": document.page_content,
                        "metadata": document.metadata
                    }, f, indent=2)
            
            # Store content separately for easy access
            content_file = os.path.join(doc_dir, "content.txt")
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(document.page_content)
            
            # Store metadata separately for easy access
            metadata_file = os.path.join(doc_dir, "metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(document.metadata, f, indent=2)
            
            # Update index
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            index[doc_id] = {
                "added_at": datetime.now().isoformat(),
                "content_length": len(document.page_content),
                "metadata_keys": list(document.metadata.keys())
            }
            
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2)
            
            logger.info(f"Added document {doc_id} to store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {e}", exc_info=True)
            return False
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document from the store.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document or None if not found
        """
        doc_dir = os.path.join(self.store_dir, doc_id)
        
        if not os.path.exists(doc_dir):
            logger.warning(f"Document {doc_id} not found")
            return None
        
        try:
            if self.use_pickle:
                doc_file = os.path.join(doc_dir, "document.pkl")
                if os.path.exists(doc_file):
                    with open(doc_file, 'rb') as f:
                        return pickle.load(f)
            else:
                doc_file = os.path.join(doc_dir, "document.json")
                if os.path.exists(doc_file):
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        doc_data = json.load(f)
                        return Document(
                            page_content=doc_data["page_content"],
                            metadata=doc_data["metadata"]
                        )
            
            # If document file not found, try to reconstruct from content and metadata
            content_file = os.path.join(doc_dir, "content.txt")
            metadata_file = os.path.join(doc_dir, "metadata.json")
            
            if os.path.exists(content_file) and os.path.exists(metadata_file):
                with open(content_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                return Document(
                    page_content=content,
                    metadata=metadata
                )
            
            logger.warning(f"Document {doc_id} files not found")
            return None
            
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}", exc_info=True)
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the store.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if successful, False otherwise
        """
        doc_dir = os.path.join(self.store_dir, doc_id)
        
        if not os.path.exists(doc_dir):
            logger.warning(f"Document {doc_id} not found")
            return False
        
        try:
            # Remove document directory
            shutil.rmtree(doc_dir)
            
            # Update index
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            if doc_id in index:
                del index[doc_id]
            
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2)
            
            logger.info(f"Deleted document {doc_id} from store")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}", exc_info=True)
            return False
    
    def list_documents(self) -> List[str]:
        """
        List all document IDs in the store.
        
        Returns:
            List of document IDs
        """
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            return list(index.keys())
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}", exc_info=True)
            return []
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the store.
        
        Returns:
            Number of documents
        """
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            return len(index)
            
        except Exception as e:
            logger.error(f"Error getting document count: {e}", exc_info=True)
            return 0
    
    def clear(self) -> bool:
        """
        Clear all documents from the store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all document IDs
            doc_ids = self.list_documents()
            
            # Delete each document
            for doc_id in doc_ids:
                self.delete_document(doc_id)
            
            # Reset index
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            
            logger.info(f"Cleared document store")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing document store: {e}", exc_info=True)
            return False