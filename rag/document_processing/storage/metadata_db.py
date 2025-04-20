"""
Metadata database for the RAG system.
"""
import os
import json
import sqlite3
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("metadata_db")

class MetadataDB:
    """Database for document metadata."""
    
    def __init__(self, db_path: str):
        """
        Initialize the metadata database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create metadata table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            key TEXT,
            value TEXT,
            value_type TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
        )
        ''')
        
        # Create index on doc_id and key
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metadata_doc_id ON metadata (doc_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metadata_key ON metadata (key)')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Initialized metadata database at {self.db_path}")
    
    def add_document(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Add document metadata to the database.
        
        Args:
            doc_id: Document identifier
            metadata: Document metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if document already exists
            cursor.execute('SELECT doc_id FROM documents WHERE doc_id = ?', (doc_id,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing document
                cursor.execute('UPDATE documents SET updated_at = CURRENT_TIMESTAMP WHERE doc_id = ?', (doc_id,))
                
                # Delete existing metadata
                cursor.execute('DELETE FROM metadata WHERE doc_id = ?', (doc_id,))
            else:
                # Insert new document
                cursor.execute('INSERT INTO documents (doc_id) VALUES (?)', (doc_id,))
            
            # Insert metadata
            for key, value in metadata.items():
                # Determine value type
                if value is None:
                    value_type = 'null'
                    value_str = 'null'
                elif isinstance(value, (int, float)):
                    value_type = 'number'
                    value_str = str(value)
                elif isinstance(value, bool):
                    value_type = 'boolean'
                    value_str = str(value).lower()
                elif isinstance(value, (list, dict)):
                    value_type = 'json'
                    value_str = json.dumps(value)
                else:
                    value_type = 'string'
                    value_str = str(value)
                
                cursor.execute(
                    'INSERT INTO metadata (doc_id, key, value, value_type) VALUES (?, ?, ?, ?)',
                    (doc_id, key, value_str, value_type)
                )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added metadata for document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding metadata for document {doc_id}: {e}", exc_info=True)
            return False
    
    def get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Metadata dictionary or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if document exists
            cursor.execute('SELECT doc_id FROM documents WHERE doc_id = ?', (doc_id,))
            if not cursor.fetchone():
                conn.close()
                logger.warning(f"Document {doc_id} not found")
                return None
            
            # Get metadata
            cursor.execute('SELECT key, value, value_type FROM metadata WHERE doc_id = ?', (doc_id,))
            rows = cursor.fetchall()
            
            metadata = {}
            for key, value_str, value_type in rows:
                # Convert value based on type
                if value_type == 'null':
                    metadata[key] = None
                elif value_type == 'number':
                    try:
                        if '.' in value_str:
                            metadata[key] = float(value_str)
                        else:
                            metadata[key] = int(value_str)
                    except ValueError:
                        metadata[key] = value_str
                elif value_type == 'boolean':
                    metadata[key] = value_str.lower() == 'true'
                elif value_type == 'json':
                    metadata[key] = json.loads(value_str)
                else:
                    metadata[key] = value_str
            
            conn.close()
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting metadata for document {doc_id}: {e}", exc_info=True)
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete document metadata.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete document (cascade will delete metadata)
            cursor.execute('DELETE FROM documents WHERE doc_id = ?', (doc_id,))
            
            deleted = cursor.rowcount > 0
            
            conn.commit()
            conn.close()
            
            if deleted:
                logger.info(f"Deleted metadata for document {doc_id}")
            else:
                logger.warning(f"Document {doc_id} not found for deletion")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting metadata for document {doc_id}: {e}", exc_info=True)
            return False
    
    def search_metadata(self, query: Dict[str, Any], limit: int = 100) -> List[str]:
        """
        Search for documents by metadata.
        
        Args:
            query: Metadata query dictionary
            limit: Maximum number of results
            
        Returns:
            List of matching document IDs
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if not query:
                # Return all documents if query is empty
                cursor.execute('SELECT doc_id FROM documents LIMIT ?', (limit,))
                doc_ids = [row[0] for row in cursor.fetchall()]
                conn.close()
                return doc_ids
            
            # Build query
            conditions = []
            params = []
            
            for key, value in query.items():
                # Determine value type and format
                if value is None:
                    conditions.append("(key = ? AND value_type = 'null')")
                    params.append(key)
                elif isinstance(value, (int, float)):
                    conditions.append("(key = ? AND value = ? AND value_type = 'number')")
                    params.extend([key, str(value)])
                elif isinstance(value, bool):
                    conditions.append("(key = ? AND value = ? AND value_type = 'boolean')")
                    params.extend([key, str(value).lower()])
                elif isinstance(value, (list, dict)):
                    conditions.append("(key = ? AND value = ? AND value_type = 'json')")
                    params.extend([key, json.dumps(value)])
                else:
                    conditions.append("(key = ? AND value = ? AND value_type = 'string')")
                    params.extend([key, str(value)])
            
            # Get document IDs that match all conditions
            query_str = f'''
            SELECT doc_id FROM documents WHERE doc_id IN (
                SELECT doc_id FROM metadata
                WHERE {" OR ".join(conditions)}
                GROUP BY doc_id
                HAVING COUNT(DISTINCT key) = ?
            )
            LIMIT ?
            '''
            
            params.extend([len(query), limit])
            cursor.execute(query_str, params)
            
            doc_ids = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error searching metadata: {e}", exc_info=True)
            return []
    
    def list_documents(self) -> List[str]:
        """
        List all document IDs in the database.
        
        Returns:
            List of document IDs
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT doc_id FROM documents')
            doc_ids = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}", exc_info=True)
            return []
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the database.
        
        Returns:
            Number of documents
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM documents')
            count = cursor.fetchone()[0]
            
            conn.close()
            return count
            
        except Exception as e:
            logger.error(f"Error getting document count: {e}", exc_info=True)
            return 0
    
    def clear(self) -> bool:
        """
        Clear all documents from the database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM documents')
            
            conn.commit()
            conn.close()
            
            logger.info("Cleared metadata database")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing metadata database: {e}", exc_info=True)
            return False