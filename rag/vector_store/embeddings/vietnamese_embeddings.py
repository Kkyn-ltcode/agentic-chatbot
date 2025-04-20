"""
Vietnamese embedding models for the RAG system.
"""
import os
from typing import List, Dict, Any, Optional, Union
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

from ...config.config import config

class VietnameseEmbeddings:
    """Vietnamese embedding model wrapper."""
    
    def __init__(self, model_name: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize the Vietnamese embedding model.
        
        Args:
            model_name: Name of the embedding model to use
            cache_dir: Directory to cache the model
        """
        self.model_name = model_name or config.get("embedding.model_name", "bge-small-en-vi")
        self.cache_dir = cache_dir or config.get("embedding.cache_dir")
        self.dimension = config.get("embedding.dimension", 384)
        self.normalize = config.get("embedding.normalize_embeddings", True)
        
        # Load the model
        self.model = self._load_model()
        
        # Check if we're running on GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        print(f"Loaded Vietnamese embedding model: {self.model_name} on {self.device}")
    
    def _load_model(self) -> SentenceTransformer:
        """Load the embedding model."""
        try:
            # Try to load from Hugging Face
            return SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir
            )
        except Exception as e:
            # Fallback to a default model if the specified one isn't available
            print(f"Error loading model {self.model_name}: {e}")
            print("Falling back to multilingual-e5-small model")
            return SentenceTransformer(
                "intfloat/multilingual-e5-small",
                cache_folder=self.cache_dir
            )
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of text documents to embed
            
        Returns:
            Array of embeddings
        """
        if not documents:
            return np.array([])
        
        # Process in batches for large document sets
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=len(batch) > 10
            )
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query string.
        
        Args:
            query: Query text to embed
            
        Returns:
            Query embedding
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        return embedding
    
    def similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate similarity scores between a query and documents.
        
        Args:
            query_embedding: Query embedding
            doc_embeddings: Document embeddings
            
        Returns:
            Array of similarity scores
        """
        if len(doc_embeddings) == 0:
            return np.array([])
        
        # Ensure embeddings are normalized if using cosine similarity
        if self.normalize:
            return np.dot(doc_embeddings, query_embedding)
        else:
            # Normalize for cosine similarity
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            return np.dot(doc_norms, query_norm)