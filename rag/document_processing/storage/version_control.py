"""
Version control for documents in the RAG system.
"""
import os
import json
import pickle
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import hashlib
from datetime import datetime
from langchain.schema import Document

class DocumentVersionControl:
    """Version control for documents."""
    
    def __init__(self, storage_dir: str):
        """
        Initialize the document version control.
        
        Args:
            storage_dir: Directory to store document versions
        """
        self.storage_dir = storage_dir
        self.versions_dir = os.path.join(storage_dir, "versions")
        
        # Create directories if they don't exist
        os.makedirs(self.versions_dir, exist_ok=True)
        
        # Index file maps document IDs to their versions
        self.index_file = os.path.join(storage_dir, "version_index.json")
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the version index from disk."""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_index(self) -> None:
        """Save the version index to disk."""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2)
    
    def _compute_hash(self, text: str) -> str:
        """Compute a hash for a text string."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def add_version(self, doc_id: str, document: Document) -> str:
        """
        Add a new version of a document.
        
        Args:
            doc_id: Document identifier
            document: Document to store
            
        Returns:
            Version identifier
        """
        # Create document directory if it doesn't exist
        doc_dir = os.path.join(self.versions_dir, doc_id)
        os.makedirs(doc_dir, exist_ok=True)
        
        # Compute content hash
        content_hash = self._compute_hash(document.page_content)
        
        # Generate version ID
        timestamp = int(datetime.now().timestamp())
        version_id = f"v_{timestamp}"
        
        # Save document version
        version_path = os.path.join(doc_dir, f"{version_id}.pickle")
        with open(version_path, 'wb') as f:
            pickle.dump(document, f)
        
        # Save metadata separately for easier access
        metadata_path = os.path.join(doc_dir, f"{version_id}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            # Add additional metadata
            metadata = document.metadata.copy()
            metadata["content_hash"] = content_hash
            metadata["doc_id"] = doc_id
            metadata["version_id"] = version_id
            metadata["created_at"] = datetime.now().isoformat()
            metadata["content_length"] = len(document.page_content)
            
            json.dump(metadata, f, indent=2)
        
        # Update index
        if doc_id not in self.index:
            self.index[doc_id] = {
                "versions": [],
                "latest_version": None
            }
        
        version_info = {
            "version_id": version_id,
            "content_hash": content_hash,
            "created_at": metadata["created_at"],
            "version_path": version_path,
            "metadata_path": metadata_path
        }
        
        self.index[doc_id]["versions"].append(version_info)
        self.index[doc_id]["latest_version"] = version_id
        self._save_index()
        
        return version_id
    
    def get_version(self, doc_id: str, version_id: Optional[str] = None) -> Optional[Document]:
        """
        Get a specific version of a document.
        
        Args:
            doc_id: Document identifier
            version_id: Version identifier (None for latest version)
            
        Returns:
            Document or None if not found
        """
        if doc_id not in self.index:
            return None
        
        # Determine which version to retrieve
        if version_id is None:
            version_id = self.index[doc_id]["latest_version"]
            if version_id is None:
                return None
        
        # Find the version info
        version_info = None
        for v in self.index[doc_id]["versions"]:
            if v["version_id"] == version_id:
                version_info = v
                break
        
        if version_info is None:
            return None
        
        # Load the document
        version_path = version_info["version_path"]
        if not os.path.exists(version_path):
            return None
        
        try:
            with open(version_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading document version {doc_id}/{version_id}: {e}")
            return None
    
    def get_version_metadata(self, doc_id: str, version_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific version of a document.
        
        Args:
            doc_id: Document identifier
            version_id: Version identifier (None for latest version)
            
        Returns:
            Metadata dictionary or None if not found
        """
        if doc_id not in self.index:
            return None
        
        # Determine which version to retrieve
        if version_id is None:
            version_id = self.index[doc_id]["latest_version"]
            if version_id is None:
                return None
        
        # Find the version info
        version_info = None
        for v in self.index[doc_id]["versions"]:
            if v["version_id"] == version_id:
                version_info = v
                break
        
        if version_info is None:
            return None
        
        # Load the metadata
        metadata_path = version_info["metadata_path"]
        if not os.path.exists(metadata_path):
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading version metadata {doc_id}/{version_id}: {e}")
            return None
    
    def list_versions(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        List all versions of a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            List of version info dictionaries
        """
        if doc_id not in self.index:
            return []
        
        return self.index[doc_id]["versions"]
    
    def delete_version(self, doc_id: str, version_id: str) -> bool:
        """
        Delete a specific version of a document.
        
        Args:
            doc_id: Document identifier
            version_id: Version identifier
            
        Returns:
            True if successful, False otherwise
        """
        if doc_id not in self.index:
            return False
        
        # Find the version info
        version_info = None
        version_index = -1
        for i, v in enumerate(self.index[doc_id]["versions"]):
            if v["version_id"] == version_id:
                version_info = v
                version_index = i
                break
        
        if version_info is None:
            return False
        
        # Delete files
        version_path = version_info["version_path"]
        metadata_path = version_info["metadata_path"]
        
        if os.path.exists(version_path):
            os.remove(version_path)
        
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        # Update index
        self.index[doc_id]["versions"].pop(version_index)
        
        # Update latest version if needed
        if self.index[doc_id]["latest_version"] == version_id:
            if self.index[doc_id]["versions"]:
                # Set the most recent version as the latest
                self.index[doc_id]["versions"].sort(key=lambda v: v["created_at"], reverse=True)
                self.index[doc_id]["latest_version"] = self.index[doc_id]["versions"][0]["version_id"]
            else:
                self.index[doc_id]["latest_version"] = None
        
        self._save_index()
        return True
    
    def delete_all_versions(self, doc_id: str) -> bool:
        """
        Delete all versions of a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if successful, False otherwise
        """
        if doc_id not in self.index:
            return False
        
        # Delete all version files
        doc_dir = os.path.join(self.versions_dir, doc_id)
        if os.path.exists(doc_dir):
            for file_name in os.listdir(doc_dir):
                file_path = os.path.join(doc_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            # Remove the directory
            os.rmdir(doc_dir)
        
        # Update index
        del self.index[doc_id]
        self._save_index()
        
        return True
    
    def compare_versions(self, doc_id: str, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """
        Compare two versions of a document.
        
        Args:
            doc_id: Document identifier
            version_id1: First version identifier
            version_id2: Second version identifier
            
        Returns:
            Dictionary with comparison results
        """
        # Get the documents
        doc1 = self.get_version(doc_id, version_id1)
        doc2 = self.get_version(doc_id, version_id2)
        
        if doc1 is None or doc2 is None:
            return {"error": "One or both versions not found"}
        
        # Get metadata
        meta1 = self.get_version_metadata(doc_id, version_id1)
        meta2 = self.get_version_metadata(doc_id, version_id2)
        
        # Compare content length
        len1 = len(doc1.page_content)
        len2 = len(doc2.page_content)
        
        # Simple diff stats
        words1 = set(doc1.page_content.split())
        words2 = set(doc2.page_content.split())
        
        common_words = words1.intersection(words2)
        unique_words1 = words1 - words2
        unique_words2 = words2 - words1
        
        return {
            "version1": version_id1,
            "version2": version_id2,
            "content_length_diff": len2 - len1,
            "common_word_count": len(common_words),
            "unique_words_v1": len(unique_words1),
            "unique_words_v2": len(unique_words2),
            "created_at_v1": meta1.get("created_at") if meta1 else None,
            "created_at_v2": meta2.get("created_at") if meta2 else None
        }