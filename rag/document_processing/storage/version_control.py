"""
Version control for documents in the RAG system.
"""
import os
import json
import shutil
from typing import Dict, List, Optional, Any
from langchain.schema import Document
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("version_control")

class DocumentVersionControl:
    """Version control for documents."""
    
    def __init__(self, versions_dir: str):
        """
        Initialize the document version control.
        
        Args:
            versions_dir: Directory to store document versions
        """
        self.versions_dir = versions_dir
        
        # Create versions directory if it doesn't exist
        os.makedirs(versions_dir, exist_ok=True)
        
        # Create versions index file if it doesn't exist
        self.index_file = os.path.join(versions_dir, "versions_index.json")
        if not os.path.exists(self.index_file):
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
    
    def add_version(self, doc_id: str, document: Document, version_note: str = "") -> str:
        """
        Add a new version of a document.
        
        Args:
            doc_id: Document identifier
            document: Document to store
            version_note: Optional note about this version
            
        Returns:
            Version identifier
        """
        try:
            # Create document versions directory
            doc_versions_dir = os.path.join(self.versions_dir, doc_id)
            os.makedirs(doc_versions_dir, exist_ok=True)
            
            # Generate version ID based on timestamp
            version_id = f"v_{int(datetime.now().timestamp())}"
            version_dir = os.path.join(doc_versions_dir, version_id)
            os.makedirs(version_dir, exist_ok=True)
            
            # Store document content
            content_file = os.path.join(version_dir, "content.txt")
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(document.page_content)
            
            # Store document metadata
            metadata_file = os.path.join(version_dir, "metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(document.metadata, f, indent=2)
            
            # Store version info
            version_info = {
                "version_id": version_id,
                "created_at": datetime.now().isoformat(),
                "version_note": version_note,
                "content_length": len(document.page_content),
                "metadata_keys": list(document.metadata.keys())
            }
            
            info_file = os.path.join(version_dir, "version_info.json")
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(version_info, f, indent=2)
            
            # Update versions index
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            if doc_id not in index:
                index[doc_id] = []
            
            index[doc_id].append(version_info)
            
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2)
            
            logger.info(f"Added version {version_id} for document {doc_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Error adding version for document {doc_id}: {e}", exc_info=True)
            return ""
    
    def get_version(self, doc_id: str, version_id: str) -> Optional[Document]:
        """
        Get a specific version of a document.
        
        Args:
            doc_id: Document identifier
            version_id: Version identifier
            
        Returns:
            Document or None if not found
        """
        version_dir = os.path.join(self.versions_dir, doc_id, version_id)
        
        if not os.path.exists(version_dir):
            logger.warning(f"Version {version_id} of document {doc_id} not found")
            return None
        
        try:
            # Load content
            content_file = os.path.join(version_dir, "content.txt")
            with open(content_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Load metadata
            metadata_file = os.path.join(version_dir, "metadata.json")
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return Document(
                page_content=content,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error getting version {version_id} of document {doc_id}: {e}", exc_info=True)
            return None
    
    def list_versions(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        List all versions of a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            List of version information dictionaries
        """
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            return index.get(doc_id, [])
            
        except Exception as e:
            logger.error(f"Error listing versions for document {doc_id}: {e}", exc_info=True)
            return []
    
    def delete_version(self, doc_id: str, version_id: str) -> bool:
        """
        Delete a specific version of a document.
        
        Args:
            doc_id: Document identifier
            version_id: Version identifier
            
        Returns:
            True if successful, False otherwise
        """
        version_dir = os.path.join(self.versions_dir, doc_id, version_id)
        
        if not os.path.exists(version_dir):
            logger.warning(f"Version {version_id} of document {doc_id} not found")
            return False
        
        try:
            # Remove version directory
            shutil.rmtree(version_dir)
            
            # Update versions index
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            if doc_id in index:
                index[doc_id] = [v for v in index[doc_id] if v["version_id"] != version_id]
                
                # Remove document entry if no versions left
                if not index[doc_id]:
                    del index[doc_id]
            
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2)
            
            logger.info(f"Deleted version {version_id} of document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting version {version_id} of document {doc_id}: {e}", exc_info=True)
            return False
    
    def delete_all_versions(self, doc_id: str) -> bool:
        """
        Delete all versions of a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if successful, False otherwise
        """
        doc_versions_dir = os.path.join(self.versions_dir, doc_id)
        
        if not os.path.exists(doc_versions_dir):
            logger.warning(f"No versions found for document {doc_id}")
            return False
        
        try:
            # Remove document versions directory
            shutil.rmtree(doc_versions_dir)
            
            # Update versions index
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            if doc_id in index:
                del index[doc_id]
            
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2)
            
            logger.info(f"Deleted all versions of document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting all versions of document {doc_id}: {e}", exc_info=True)
            return False
    
    def get_latest_version(self, doc_id: str) -> Optional[Document]:
        """
        Get the latest version of a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document or None if not found
        """
        try:
            versions = self.list_versions(doc_id)
            
            if not versions:
                logger.warning(f"No versions found for document {doc_id}")
                return None
            
            # Sort versions by creation timestamp (descending)
            versions.sort(key=lambda v: v["created_at"], reverse=True)
            
            # Get latest version
            latest_version_id = versions[0]["version_id"]
            
            return self.get_version(doc_id, latest_version_id)
            
        except Exception as e:
            logger.error(f"Error getting latest version of document {doc_id}: {e}", exc_info=True)
            return None