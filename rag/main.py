"""
Main entry point for the RAG system.
"""
import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from rag.vector_store.service import VectorStoreService
from rag.config.config import config

def main():
    """Main entry point for the RAG system."""
    parser = argparse.ArgumentParser(description="Vietnamese RAG System")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add documents command
    add_parser = subparsers.add_parser("add", help="Add documents to the vector store")
    add_parser.add_argument("--files", nargs="+", required=True, help="Files to add")
    add_parser.add_argument("--index", default="default", help="Index name")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for documents")
    search_parser.add_argument("query", help="Query text")
    search_parser.add_argument("--index", default="default", help="Index name")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    
    # List indices command
    list_parser = subparsers.add_parser("list-indices", help="List available indices")
    
    # Clear index command
    clear_parser = subparsers.add_parser("clear-index", help="Clear an index")
    clear_parser.add_argument("--index", default="default", help="Index name")
    
    # Clear cache command
    cache_parser = subparsers.add_parser("clear-cache", help="Clear the embedding cache")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize the vector store service
    service = VectorStoreService()
    
    # Execute the command
    if args.command == "add":
        # TODO: Implement document loading and processing
        print(f"Adding documents from {args.files} to index {args.index}")
        print("This functionality will be implemented in Phase 2")
    
    elif args.command == "search":
        print(f"Searching for: {args.query}")
        doc_ids, scores, metadata = service.search(args.query, args.top_k, args.index)
        
        if not doc_ids:
            print("No results found.")
        else:
            print(f"\nTop {len(doc_ids)} results:")
            for i, (doc_id, score, meta) in enumerate(zip(doc_ids, scores, metadata)):
                print(f"  Result {i+1}: {doc_id} (Score: {score:.4f})")
                for key, value in meta.items():
                    print(f"    {key}: {value}")
    
    elif args.command == "list-indices":
        indices = service.list_indices()
        if not indices:
            print("No indices found.")
        else:
            print("Available indices:")
            for index in indices:
                print(f"  - {index}")
    
    elif args.command == "clear-index":
        print(f"Clearing index: {args.index}")
        service.clear_index(args.index)
        print("Index cleared successfully.")
    
    elif args.command == "clear-cache":
        print("Clearing embedding cache...")
        service.clear_cache()
        print("Cache cleared successfully.")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()