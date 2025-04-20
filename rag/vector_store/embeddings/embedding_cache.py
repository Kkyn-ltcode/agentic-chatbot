"""
Caching system for embeddings to avoid redundant computation.
"""
import os
import json
from typing import Dict, List, Optional, Any
import numpy as np
import hashlib
from pathlib import Path

from ...config.config import config

class EmbeddingCache:
    """Cache for document and query embeddings."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory to store the cache
        """
        self.cache_dir = cache_dir or os.path.join(
            config.get("embedding.cache_dir"), "cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Index file maps document IDs to their hash values
        self.index_file = os.path.join(self.cache_dir, "cache_index.json")
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, str]:
        """Load the cache index from disk."""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_index(self) -> None:
        """Save the cache index to disk."""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2)
    
    def _compute_hash(self, text: str) -> str:
        """Compute a hash for a text string."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_document_embedding(self, doc_id: str, text: str) -> Optional[np.ndarray]:
        """
        Get a document embedding from cache if available.
        
        Args:
            doc_id: Document identifier
            text: Document text (used to check if content has changed)
            
        Returns:
            Cached embedding or None if not found
        """
        # Check if document is in cache
        text_hash = self._compute_hash(text)
        
        if doc_id in self.index and self.index[doc_id] == text_hash:
            # Document exists in cache and hasn't changed
            cache_path = os.path.join(self.cache_dir, f"{doc_id}.npy")
            if os.path.exists(cache_path):
                try:
                    return np.load(cache_path)
                except Exception as e:
                    print(f"Error loading cached embedding: {e}")
        
        return None
    
    def save_document_embedding(self, doc_id: str, text: str, embedding: np.ndarray) -> None:
        """
        Save a document embedding to cache.
        
        Args:
            doc_id: Document identifier
            text: Document text
            embedding: Document embedding
        """
        text_hash = self._compute_hash(text)
        cache_path = os.path.join(self.cache_dir, f"{doc_id}.npy")
        
        # Save the embedding
        np.save(cache_path, embedding)
        
        # Update the index
        self.index[doc_id] = text_hash
        self._save_index()
    
    def get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """
        Get a query embedding from cache if available.
        
        Args:
            query: Query text
            
        Returns:
            Cached embedding or None if not found
        """
        query_hash = self._compute_hash(query)
        cache_path = os.path.join(self.cache_dir, f"query_{query_hash}.npy")
        
        if os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception as e:
                print(f"Error loading cached query embedding: {e}")
        
        return None
    
    def save_query_embedding(self, query: str, embedding: np.ndarray) -> None:
        """
        Save a query embedding to cache.
        
        Args:
            query: Query text
            embedding: Query embedding
        """
        query_hash = self._compute_hash(query)
        cache_path = os.path.join(self.cache_dir, f"query_{query_hash}.npy")
        
        # Save the embedding
        np.save(cache_path, embedding)
    
    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        for file_path in Path(self.cache_dir).glob("*.npy"):
            os.remove(file_path)
        
        self.index = {}
        self._save_index()
        print("Embedding cache cleared")