"""
FAISS vector index implementation for efficient similarity search.
"""
import os
import json
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import faiss
from pathlib import Path

from ...config.config import config

class FAISSIndex:
    """FAISS vector index for document embeddings."""
    
    def __init__(self, dimension: Optional[int] = None, index_path: Optional[str] = None):
        """
        Initialize the FAISS index.
        
        Args:
            dimension: Dimension of the embeddings
            index_path: Path to save/load the index
        """
        self.dimension = dimension or config.get("embedding.dimension", 384)
        self.index_path = index_path or config.get("vector_store.index_path")
        self.metadata_path = os.path.join(self.index_path, "metadata.json")
        
        # Create directory if it doesn't exist
        os.makedirs(self.index_path, exist_ok=True)
        
        # Initialize index and metadata
        self.index = None
        self.metadata = {}
        self.doc_ids = []
        
        # Try to load existing index
        self._load_or_create_index()
    
    def _load_or_create_index(self) -> None:
        """Load an existing index or create a new one."""
        index_file = os.path.join(self.index_path, "faiss_index.bin")
        
        if os.path.exists(index_file):
            try:
                self.index = faiss.read_index(index_file)
                print(f"Loaded FAISS index with {self.index.ntotal} vectors")
                
                # Load metadata
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.metadata = data.get("metadata", {})
                        self.doc_ids = data.get("doc_ids", [])
                
                return
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                print("Creating new index...")
        
        # Create a new index
        self._create_new_index()
    
    def _create_new_index(self) -> None:
        """Create a new FAISS index."""
        # Create a new L2 index (can be changed to inner product if using normalized vectors)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = {}
        self.doc_ids = []
        print(f"Created new FAISS index with dimension {self.dimension}")
    
    def _save_index(self) -> None:
        """Save the index and metadata to disk."""
        if self.index is None:
            return
        
        index_file = os.path.join(self.index_path, "faiss_index.bin")
        faiss.write_index(self.index, index_file)
        
        # Save metadata
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": self.metadata,
                "doc_ids": self.doc_ids
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Saved FAISS index with {self.index.ntotal} vectors")
    
    def add_documents(self, doc_ids: List[str], embeddings: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add documents to the index.
        
        Args:
            doc_ids: List of document IDs
            embeddings: Document embeddings as a numpy array
            metadata: Optional metadata for each document
        """
        if self.index is None:
            self._create_new_index()
        
        if len(doc_ids) != embeddings.shape[0]:
            raise ValueError("Number of document IDs must match number of embeddings")
        
        # Add embeddings to the index
        self.index.add(embeddings.astype(np.float32))
        
        # Store document IDs and metadata
        start_idx = len(self.doc_ids)
        self.doc_ids.extend(doc_ids)
        
        if metadata:
            for i, doc_id in enumerate(doc_ids):
                self.metadata[doc_id] = metadata[i] if i < len(metadata) else {}
        else:
            for doc_id in doc_ids:
                self.metadata[doc_id] = {}
        
        # Save the updated index
        self._save_index()
    
    def search(self, query_embedding: np.ndarray, top_k: Optional[int] = None) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            Tuple of (document IDs, similarity scores, metadata)
        """
        if self.index is None or self.index.ntotal == 0:
            return [], [], []
        
        k = top_k or config.get("vector_store.similarity_top_k", 5)
        k = min(k, self.index.ntotal)  # Can't retrieve more than we have
        
        # Ensure the query embedding is the right shape and type
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert to document IDs and metadata
        result_ids = [self.doc_ids[idx] for idx in indices[0]]
        result_scores = [float(1.0 / (1.0 + dist)) for dist in distances[0]]  # Convert distance to similarity score
        result_metadata = [self.metadata.get(doc_id, {}) for doc_id in result_ids]
        
        return result_ids, result_scores, result_metadata
    
    def delete_documents(self, doc_ids: List[str]) -> None:
        """
        Delete documents from the index.
        
        Args:
            doc_ids: List of document IDs to delete
        """
        if self.index is None or self.index.ntotal == 0:
            return
        
        # FAISS doesn't support direct deletion, so we need to rebuild the index
        # Get all embeddings
        all_embeddings = faiss.rev_swig_ptr(self.index.get_xb(), self.index.ntotal * self.dimension)
        all_embeddings = all_embeddings.reshape(self.index.ntotal, self.dimension)
        
        # Create a new index
        new_index = faiss.IndexFlatL2(self.dimension)
        new_doc_ids = []
        new_metadata = {}
        
        # Add back all documents except the ones to delete
        for i, doc_id in enumerate(self.doc_ids):
            if doc_id not in doc_ids:
                new_index.add(all_embeddings[i:i+1])
                new_doc_ids.append(doc_id)
                new_metadata[doc_id] = self.metadata.get(doc_id, {})
        
        # Replace the old index
        self.index = new_index
        self.doc_ids = new_doc_ids
        self.metadata = new_metadata
        
        # Save the updated index
        self._save_index()
    
    def clear(self) -> None:
        """Clear the index."""
        self._create_new_index()
        self._save_index()