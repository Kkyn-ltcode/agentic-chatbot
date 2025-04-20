"""
Command-line interface for the document processing pipeline.
"""
import os
import sys
import argparse
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

from .pipeline import DocumentProcessingPipeline

def main():
    """Main entry point for the document processing CLI."""
    parser = argparse.ArgumentParser(description="Document Processing Pipeline CLI")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process documents")
    process_parser.add_argument("source", help="File path, URL, or directory to process")
    process_parser.add_argument("--recursive", action="store_true", help="Process directories recursively")
    process_parser.add_argument("--extensions", nargs="+", help="File extensions to process")
    process_parser.add_argument("--skip-chunking", action="store_true", help="Skip document chunking")
    process_parser.add_argument("--metadata", help="JSON string or file path with custom metadata")
    process_parser.add_argument("--output-dir", default="./processed_data", help="Output directory for processed documents")
    process_parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size")
    process_parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents by metadata")
    search_parser.add_argument("query", help="JSON string or file path with metadata query")
    search_parser.add_argument("--limit", type=int, default=100, help="Maximum number of results")
    search_parser.add_argument("--output-dir", default="./processed_data", help="Directory with processed documents")
    
    # Get command
    get_parser = subparsers.add_parser("get", help="Get document by ID")
    get_parser.add_argument("doc_id", help="Document ID")
    get_parser.add_argument("--output-dir", default="./processed_data", help="Directory with processed documents")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete document by ID")
    delete_parser.add_argument("doc_id", help="Document ID")
    delete_parser.add_argument("--output-dir", default="./processed_data", help="Directory with processed documents")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get processing statistics")
    stats_parser.add_argument("--output-dir", default="./processed_data", help="Directory with processed documents")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize pipeline
    pipeline = DocumentProcessingPipeline(
        base_dir=args.output_dir,
        chunk_size=args.chunk_size if hasattr(args, "chunk_size") else 512,
        chunk_overlap=args.chunk_overlap if hasattr(args, "chunk_overlap") else 50
    )
    
    # Execute command
    if args.command == "process":
        # Parse custom metadata
        custom_metadata = None
        if args.metadata:
            if os.path.isfile(args.metadata):
                with open(args.metadata, 'r', encoding='utf-8') as f:
                    custom_metadata = json.load(f)
            else:
                try:
                    custom_metadata = json.loads(args.metadata)
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON metadata: {args.metadata}")
                    return
        
        # Process source
        if os.path.isdir(args.source):
            # Process directory
            results = pipeline.process_directory(
                directory=args.source,
                recursive=args.recursive,
                file_extensions=args.extensions,
                custom_metadata=custom_metadata,
                skip_chunking=args.skip_chunking
            )
            print(f"Processed {len(results)} files")
            print(f"Created {pipeline.stats['chunks_created']} document chunks")
            print(f"Errors: {pipeline.stats['processing_errors']}")
        else:
            # Process single file or URL
            doc_ids = pipeline.process_document(
                source=args.source,
                custom_metadata=custom_metadata,
                skip_chunking=args.skip_chunking
            )
            print(f"Processed document: {args.source}")
            print(f"Created {len(doc_ids)} document chunks")
            for i, doc_id in enumerate(doc_ids):
                print(f"  Chunk {i+1}: {doc_id}")
    
    elif args.command == "search":
        # Parse query
        query = None
        if os.path.isfile(args.query):
            with open(args.query, 'r', encoding='utf-8') as f:
                query = json.load(f)
        else:
            try:
                query = json.loads(args.query)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON query: {args.query}")
                return
        
        # Search documents
        doc_ids = pipeline.search_by_metadata(query, args.limit)
        print(f"Found {len(doc_ids)} matching documents")
        
        for i, doc_id in enumerate(doc_ids):
            metadata = pipeline.get_document_metadata(doc_id)
            print(f"\nDocument {i+1}: {doc_id}")
            if metadata:
                for key in ["file_name", "file_path", "language", "content_length"]:
                    if key in metadata:
                        print(f"  {key}: {metadata[key]}")
    
    elif args.command == "get":
        # Get document
        document = pipeline.get_document(args.doc_id)
        if document:
            print(f"Document: {args.doc_id}")
            print(f"Metadata: {document.metadata}")
            print("\nContent:")
            print(document.page_content[:500] + "..." if len(document.page_content) > 500 else document.page_content)
        else:
            print(f"Document not found: {args.doc_id}")
    
    elif args.command == "delete":
        # Delete document
        success = pipeline.delete_document(args.doc_id)
        if success:
            print(f"Document deleted: {args.doc_id}")
        else:
            print(f"Failed to delete document: {args.doc_id}")
    
    elif args.command == "stats":
        # Get stats
        stats = pipeline.get_processing_stats()
        print("Processing Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()