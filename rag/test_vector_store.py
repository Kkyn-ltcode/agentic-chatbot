"""
Test script for the vector store service.
"""
import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from rag.vector_store.service import VectorStoreService

def test_vector_store():
    """Test the vector store service."""
    print("Initializing vector store service...")
    service = VectorStoreService()
    
    # Test documents (in Vietnamese)
    documents = [
        "Hà Nội là thủ đô của Việt Nam, một thành phố với lịch sử lâu đời.",
        "Thành phố Hồ Chí Minh là trung tâm kinh tế lớn nhất của Việt Nam.",
        "Đà Nẵng là thành phố cảng nằm ở miền Trung Việt Nam.",
        "Huế từng là kinh đô của Việt Nam dưới triều Nguyễn.",
        "Nha Trang nổi tiếng với những bãi biển đẹp và là điểm du lịch nổi tiếng."
    ]
    
    doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    
    metadata = [
        {"city": "Hanoi", "region": "North"},
        {"city": "Ho Chi Minh City", "region": "South"},
        {"city": "Da Nang", "region": "Central"},
        {"city": "Hue", "region": "Central"},
        {"city": "Nha Trang", "region": "Central"}
    ]
    
    # Add documents to the vector store
    print("Adding documents to the vector store...")
    service.add_documents(documents, doc_ids, metadata)
    
    # Test search
    print("\nTesting search functionality...")
    queries = [
        "Thủ đô của Việt Nam",
        "Thành phố ở miền Trung",
        "Địa điểm du lịch biển"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        doc_ids, scores, meta = service.search(query, top_k=3)
        
        for i, (doc_id, score, m) in enumerate(zip(doc_ids, scores, meta)):
            print(f"  Result {i+1}: {doc_id} (Score: {score:.4f})")
            print(f"    City: {m.get('city')}, Region: {m.get('region')}")
    
    # Test deleting a document
    print("\nTesting document deletion...")
    service.delete_documents(["doc1"])
    
    print("\nSearching after deletion:")
    doc_ids, scores, meta = service.search("Thủ đô của Việt Nam", top_k=3)
    for i, (doc_id, score, m) in enumerate(zip(doc_ids, scores, meta)):
        print(f"  Result {i+1}: {doc_id} (Score: {score:.4f})")
        print(f"    City: {m.get('city')}, Region: {m.get('region')}")
    
    # Test clearing cache
    print("\nTesting cache clearing...")
    service.clear_cache()
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_vector_store()