"""
Configuration management for the RAG system.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path("/Users/nguyen/Documents/Work/agentic")
RAG_DIR = BASE_DIR / "rag"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, DATA_DIR / "documents", MODELS_DIR / "embeddings"]:
    dir_path.mkdir(exist_ok=True, parents=True)

class Config:
    """Configuration manager for the RAG system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from a JSON file or default values."""
        self.config_path = config_path or os.path.join(RAG_DIR, "config", "default_config.json")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Default configuration
            default_config = {
                "embedding": {
                    "model_name": "bge-small-en-vi",
                    "cache_dir": str(MODELS_DIR / "embeddings"),
                    "dimension": 384,
                    "normalize_embeddings": True
                },
                "vector_store": {
                    "index_type": "faiss",
                    "index_path": str(DATA_DIR / "indices"),
                    "similarity_top_k": 5
                },
                "document_processing": {
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                    "documents_dir": str(DATA_DIR / "documents")
                },
                "retrieval": {
                    "use_hybrid_search": True,
                    "rerank_results": True,
                    "max_documents": 10
                },
                "vietnamese": {
                    "use_vietnamese_tokenizer": True,
                    "remove_vietnamese_stopwords": True,
                    "normalize_unicode": True
                }
            }
            # Save default config
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            return default_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value and save to file."""
        keys = key.split('.')
        config = self.config
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        
        # Save updated config
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the entire configuration as a dictionary."""
        return self.config.copy()


# Create a singleton instance
config = Config()