"""
Manager for multiple FAISS indices.
"""
import os
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from pathlib import Path

from .faiss_index import FAISSIndex
from ...config.config import config

class IndexManager:
    """Manager for multiple FAISS indices."""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the index manager.
        
        Args:
            base_path: Base path for storing indices
        """
        self.base_path = base_path or config.get("vector_store.index_path")
        os.makedirs(self.base_path, exist_ok=True)
        
        # Dictionary of indices
        self.indices: Dict[str, FAISSIndex] = {}
    
    def get_index(self, index_name: str, dimension: Optional[int] = None) -> FAISSIndex:
        """
        Get or create an index.
        
        Args:
            index_name: Name of the index
            dimension: Dimension of the embeddings
            
        Returns:
            FAISS index
        """
        if index_name in self.indices:
            return self.indices[index_name]
        
        # Create a new index
        index_path = os.path.join(self.base_path, index_name)
        index = FAISSIndex(dimension=dimension, index_path=index_path)
        self.indices[index_name] = index
        
        return index
    
    def list_indices(self) -> List[str]:
        """
        List all available indices.
        
        Returns:
            List of index names
        """
        # Check directories in the base path
        indices = []
        for item in Path(self.base_path).iterdir():
            if item.is_dir() and (item / "faiss_index.bin").exists():
                indices.append(item.name)
        
        return indices
    
    def delete_index(self, index_name: str) -> bool:
        """
        Delete an index.
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            True if successful, False otherwise
        """
        if index_name in self.indices:
            del self.indices[index_name]
        
        index_path = os.path.join(self.base_path, index_name)
        if os.path.exists(index_path):
            try:
                for file_path in Path(index_path).glob("*"):
                    os.remove(file_path)
                os.rmdir(index_path)
                return True
            except Exception as e:
                print(f"Error deleting index {index_name}: {e}")
        
        return False