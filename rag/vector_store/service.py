"""
Vector store service that combines embeddings and indices.
"""
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import os

from .embeddings.vietnamese_embeddings import VietnameseEmbeddings
from .embeddings.embedding_cache import EmbeddingCache
from .indices.index_manager import IndexManager
from ..config.config import config

class VectorStoreService:
    """Service for managing document embeddings and vector search."""
    
    def __init__(self):
        """Initialize the vector store service."""
        # Initialize components
        self.embedding_model = VietnameseEmbeddings()
        self.embedding_cache = EmbeddingCache()
        self.index_manager = IndexManager()
        
        # Default index name
        self.default_index = "default"
    
    def add_documents(self, 
                      documents: List[str], 
                      doc_ids: List[str], 
                      metadata: Optional[List[Dict[str, Any]]] = None,
                      index_name: Optional[str] = None) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            doc_ids: List of document IDs
            metadata: Optional metadata for each document
            index_name: Name of the index to use
        """
        if len(documents) != len(doc_ids):
            raise ValueError("Number of documents must match number of document IDs")
        
        # Get or create embeddings
        embeddings = []
        for i, (doc, doc_id) in enumerate(zip(documents, doc_ids)):
            # Check cache first
            cached_embedding = self.embedding_cache.get_document_embedding(doc_id, doc)
            
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                # Generate new embedding
                embedding = self.embedding_model.embed_documents([doc])[0]
                self.embedding_cache.save_document_embedding(doc_id, doc, embedding)
                embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Add to index
        index = self.index_manager.get_index(
            index_name or self.default_index,
            dimension=self.embedding_model.dimension
        )
        index.add_documents(doc_ids, embeddings_array, metadata)
    
    def search(self, 
               query: str, 
               top_k: Optional[int] = None,
               index_name: Optional[str] = None) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            index_name: Name of the index to search
            
        Returns:
            Tuple of (document IDs, similarity scores, metadata)
        """
        # Check cache for query embedding
        query_embedding = self.embedding_cache.get_query_embedding(query)
        
        if query_embedding is None:
            # Generate new embedding
            query_embedding = self.embedding_model.embed_query(query)
            self.embedding_cache.save_query_embedding(query, query_embedding)
        
        # Search the index
        index = self.index_manager.get_index(index_name or self.default_index)
        return index.search(query_embedding, top_k)
    
    def delete_documents(self, 
                         doc_ids: List[str],
                         index_name: Optional[str] = None) -> None:
        """
        Delete documents from the vector store.
        
        Args:
            doc_ids: List of document IDs to delete
            index_name: Name of the index
        """
        index = self.index_manager.get_index(index_name or self.default_index)
        index.delete_documents(doc_ids)
    
    def list_indices(self) -> List[str]:
        """
        List all available indices.
        
        Returns:
            List of index names
        """
        return self.index_manager.list_indices()
    
    def clear_index(self, index_name: Optional[str] = None) -> None:
        """
        Clear an index.
        
        Args:
            index_name: Name of the index to clear
        """
        index = self.index_manager.get_index(index_name or self.default_index)
        index.clear()
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.embedding_cache.clear_cache()