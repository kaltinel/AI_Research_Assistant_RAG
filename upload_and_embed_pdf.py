#!/usr/bin/env python3
"""
Standalone CLI script for uploading and embedding PDF files into a vector database.

Usage:
    python upload_and_embed_pdf.py /path/to/your/document.pdf

This script:
1. Takes an absolute path to a PDF file
2. Compresses the PDF using pikepdf
3. Converts PDF to text using pdfplumber
4. Generates embeddings using Ollama
5. Stores embeddings in ChromaDB vector database
"""

import argparse
import logging
import pathlib
import time
import sys
import gc
from typing import Optional

import pikepdf
import pdfplumber
import chromadb
from chromadb.config import Settings
import ollama


# Configuration - adapt these to your needs
EMBEDDING_MODEL = "qwen3:0.6b"
CHROMA_DB_PATH = "./chromadb_storage"
COLLECTION_NAME = "pdf_documents"
MAX_CHUNKS = 5
TARGET_CHUNK_SIZE = 4000

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compress_pdf(input_path: str, output_path: Optional[str] = None) -> str:
    """Compress a PDF file using pikepdf while maintaining quality."""
    import tempfile
    import shutil
    
    if output_path is None:
        output_path = input_path
    
    try:
        logger.info(f"Compressing PDF: {input_path}")
        
        # Use temporary file approach if input and output are the same
        if input_path == output_path:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Open and save to temporary file
            pdf = pikepdf.open(input_path)
            pdf.save(temp_path, 
                     compress_streams=True,  
                     preserve_pdfa=True,     
                     object_stream_mode=pikepdf.ObjectStreamMode.generate,  
                     linearize=True)
            pdf.close()
            
            # Replace original file with compressed version
            shutil.move(temp_path, output_path)
        else:
            # Different input and output paths - direct save
            pdf = pikepdf.open(input_path)
            pdf.save(output_path, 
                     compress_streams=True,  
                     preserve_pdfa=True,     
                     object_stream_mode=pikepdf.ObjectStreamMode.generate,  
                     linearize=True)
            pdf.close()
        
        logger.info(f"PDF compression completed: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"PDF compression error: {str(e)}")
        return input_path


def convert_pdf_to_text(pdf_path: str) -> tuple[bool, str, str]:
    """Convert a PDF file to text with memory management."""
    logger.info(f"Starting PDF text extraction for: {pdf_path}")
    
    try:
        # Check file size for memory management
        file_size = pathlib.Path(pdf_path).stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"PDF file size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 100:
            return False, "", f"PDF file too large ({file_size_mb:.1f}MB, max 100MB)"
        
        extracted_text = ""
        total_pages = 0
        pages_with_content = 0
        
        # Process PDF with memory management
        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"PDF loaded - {total_pages} pages detected")
            
            if total_pages == 0:
                return False, "", "PDF file contains no pages"
            
            if total_pages > 1000:
                logger.warning(f"PDF has {total_pages} pages - too large")
                return False, "", f"PDF too large ({total_pages} pages). Maximum supported is 1000 pages."
            
            # Process pages in batches
            batch_size = 5
            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                logger.info(f"Processing pages {batch_start+1}-{batch_end}")
                
                for page_idx in range(batch_start, batch_end):
                    page_num = page_idx + 1
                    
                    try:
                        page = pdf.pages[page_idx]
                        logger.debug(f"Processing page {page_num}/{total_pages}")
                        
                        # Extract text from page
                        page_text = page.extract_text()
                        
                        if page_text and page_text.strip():
                            extracted_text += f"\n\n=== Page {page_num} ===\n\n"
                            extracted_text += page_text.strip()
                            pages_with_content += 1
                            
                            # Try to extract tables if available
                            try:
                                tables = page.extract_tables()
                                if tables:
                                    extracted_text += f"\n\n--- Tables on Page {page_num} ---\n\n"
                                    for table_num, table in enumerate(tables, 1):
                                        if table:
                                            extracted_text += f"Table {table_num}:\n"
                                            for row in table:
                                                if row:
                                                    row_text = "\t".join([str(cell).strip() if cell is not None else "" for cell in row])
                                                    if row_text.strip():
                                                        extracted_text += row_text + "\n"
                                            extracted_text += "\n"
                            except Exception as table_error:
                                logger.debug(f"Could not extract tables from page {page_num}: {str(table_error)}")
                        else:
                            logger.debug(f"No text found on page {page_num}")
                            extracted_text += f"\n\n=== Page {page_num} ===\n\n[No readable text found on this page]\n"
                        
                        # Clear page reference
                        del page
                            
                    except Exception as page_error:
                        logger.warning(f"Error processing page {page_num}: {str(page_error)}")
                        extracted_text += f"\n\n=== Page {page_num} ===\n\n[Error processing this page: {str(page_error)}]\n"
                
                # Force garbage collection after each batch
                gc.collect()
                logger.debug(f"Completed batch {batch_start+1}-{batch_end}, memory cleaned")
                time.sleep(0.1)
        
        logger.info(f"Text extraction complete! {pages_with_content}/{total_pages} pages contained readable text")
        
        if len(extracted_text) < 50:
            logger.warning("Minimal content extracted from PDF")
            return False, "", "Minimal text content was extracted from this PDF"
        
        logger.info(f"Successfully converted PDF to text: {len(extracted_text)} characters extracted")
        gc.collect()
        
        return True, extracted_text, ""
        
    except Exception as e:
        logger.error(f"Error in PDF conversion: {str(e)}")
        return False, "", f"Error processing PDF file: {str(e)}"


def unwrap_embedding(embedding_response):
    """Unwrap nested embedding structure from Ollama."""
    try:
        # Handle EmbedResponse object from Ollama
        if hasattr(embedding_response, 'embedding'):
            embedding = embedding_response.embedding
        elif hasattr(embedding_response, 'embeddings'):
            embedding = embedding_response.embeddings
        # Handle dictionary response
        elif isinstance(embedding_response, dict):
            if "embeddings" in embedding_response:
                embedding = embedding_response["embeddings"]
            elif "embedding" in embedding_response:
                embedding = embedding_response["embedding"]
            else:
                raise ValueError(f"No embedding found in response. Available keys: {list(embedding_response.keys())}")
        # Handle direct list
        elif isinstance(embedding_response, list):
            embedding = embedding_response
        else:
            raise ValueError(f"Unexpected response type: {type(embedding_response)}")
        
        # Handle nested list structures
        while isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
            embedding = embedding[0]
        
        # Validate that we have a proper embedding vector
        if not isinstance(embedding, list) or len(embedding) == 0:
            raise ValueError(f"Invalid embedding format: {type(embedding)} with length {len(embedding) if hasattr(embedding, '__len__') else 'unknown'}")
        
        # Check if all elements are numbers
        if not all(isinstance(x, (int, float)) for x in embedding[:5]):  # Check first 5 elements
            raise ValueError(f"Embedding contains non-numeric values: {embedding[:5]}")
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error unwrapping embedding: {str(e)}")
        logger.error(f"Response type: {type(embedding_response)}")
        logger.error(f"Response content: {embedding_response}")
        raise


def create_optimized_chunks(text: str, max_chunks: int = 5, target_size: int = 4000) -> list[str]:
    """Create optimized chunks with maximum limit for speed."""
    if len(text) <= target_size:
        return [text]
    
    # Calculate optimal chunk size to stay within max_chunks limit
    optimal_size = max(target_size, len(text) // max_chunks)
    overlap = min(300, optimal_size // 10)  # 10% overlap, max 300 chars
    
    chunks = []
    start = 0
    
    while start < len(text) and len(chunks) < max_chunks:
        end = start + optimal_size
        
        # For the last chunk, take everything remaining
        if len(chunks) == max_chunks - 1:
            end = len(text)
        
        # Find good breaking point if not the last chunk
        if end < len(text):
            for break_char in ['\n\n', '\n', '. ', '? ', '! ']:
                break_pos = text.rfind(break_char, start, end)
                if break_pos > start + optimal_size // 2:
                    end = break_pos + len(break_char)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = max(start + optimal_size - overlap, end - overlap)
        
        if start >= len(text):
            break
    
    # If we still have remaining text and haven't reached max_chunks, add it to the last chunk
    if start < len(text) and chunks:
        chunks[-1] += text[start:]
    
    return chunks


def get_chromadb_client():
    """Get ChromaDB client."""
    chromadb_path = pathlib.Path(CHROMA_DB_PATH)
    chromadb_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chromadb_path), settings=Settings(allow_reset=True))


def embed_and_store_document(pdf_path: str, text_content: str) -> tuple[bool, str]:
    """Embed text content and store in ChromaDB."""
    logger.info(f"Starting embedding process for: {pdf_path}")
    
    try:
        # Create document ID from file path
        doc_id = pathlib.Path(pdf_path).stem
        
        # Initialize ChromaDB
        logger.debug("Initializing ChromaDB client...")
        client = get_chromadb_client()
        logger.debug("Creating/getting collection...")
        try:
            collection = client.get_or_create_collection(name=COLLECTION_NAME)
            logger.debug(f"Collection '{COLLECTION_NAME}' initialized successfully")
        except Exception as collection_error:
            logger.error(f"Error creating/getting collection: {str(collection_error)}")
            logger.error(f"Collection error type: {type(collection_error)}")
            raise
        
        # Remove existing entries for this document
        try:
            logger.debug("Checking for existing documents...")
            existing_docs = collection.get(where={"document_path": pdf_path})
            logger.debug(f"Found {len(existing_docs.get('ids', []))} existing documents")
            if existing_docs["ids"]:
                logger.debug(f"Deleting {len(existing_docs['ids'])} existing entries")
                collection.delete(ids=existing_docs["ids"])
                logger.info(f"Removed {len(existing_docs['ids'])} existing entries")
        except Exception as cleanup_error:
            logger.error(f"Error during existing document cleanup: {str(cleanup_error)}")
            logger.error(f"Cleanup error type: {type(cleanup_error)}")
            # Continue processing even if cleanup fails
            pass
        
        # Create base metadata
        base_metadata = {
            "document_path": pdf_path,
            "document_name": pathlib.Path(pdf_path).name,
            "embedding_timestamp": time.time(),
            "text_length": len(text_content)
        }
        
        # Choose embedding strategy based on document size
        if len(text_content) <= 15000:  # Small to medium documents - single embedding
            logger.info("Using single document embedding strategy")
            
            logger.info("Generating embedding for complete document...")
            response = ollama.embed(
                model=EMBEDDING_MODEL,
                input=text_content,
                options={"timeout": 90}
            )
            logger.debug(f"Ollama response type: {type(response)}")
            
            embedding = unwrap_embedding(response)
            logger.info(f"Embedding generated successfully, dimension: {len(embedding)}")
            
            # Create metadata for complete document
            complete_metadata = {
                **base_metadata,
                "chunk_type": "complete_document",
                "is_complete": True
            }
            
            # Prepare data for ChromaDB
            logger.info("Adding embedding to ChromaDB...")
            try:
                collection.add(
                    embeddings=[embedding],
                    documents=[text_content],
                    metadatas=[complete_metadata],
                    ids=[f"{doc_id}_complete"]
                )
                logger.info("Successfully added embedding to ChromaDB")
            except Exception as add_error:
                logger.error(f"ChromaDB add error: {str(add_error)}")
                logger.error(f"Embedding type: {type(embedding)}, length: {len(embedding)}")
                logger.error(f"Metadata keys: {list(complete_metadata.keys())}")
                raise
            
            logger.info("Single embedding completed successfully")
            chunks_created = 1
            
        else:  # Large documents - optimized chunking
            logger.info("Using optimized chunking strategy for large document")
            
            chunks = create_optimized_chunks(text_content, max_chunks=MAX_CHUNKS, target_size=TARGET_CHUNK_SIZE)
            logger.info(f"Created {len(chunks)} optimized chunks")
            
            # Batch process chunks
            chunk_embeddings = []
            chunk_documents = []
            chunk_metadatas = []
            chunk_ids = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                try:
                    logger.info(f"Generating embedding for chunk {i+1}...")
                    chunk_response = ollama.embed(
                        model=EMBEDDING_MODEL,
                        input=chunk,
                        options={"timeout": 60}
                    )
                    logger.debug(f"Chunk {i+1} response type: {type(chunk_response)}")
                    
                    chunk_embedding = unwrap_embedding(chunk_response)
                    logger.info(f"Chunk {i+1} embedding generated, dimension: {len(chunk_embedding)}")
                    
                    # Create metadata for each chunk
                    chunk_metadata = {
                        **base_metadata,
                        "chunk_type": "optimized_chunk",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_text_length": len(chunk),
                        "is_complete": False
                    }
                    
                    chunk_embeddings.append(chunk_embedding)
                    chunk_documents.append(chunk)
                    chunk_metadatas.append(chunk_metadata)
                    chunk_ids.append(f"{doc_id}_chunk_{i}")
                    
                except Exception as e:
                    logger.error(f"Failed to embed chunk {i}: {str(e)}")
                    continue
            
            # Batch add all chunks at once
            if chunk_embeddings:
                logger.info(f"Adding {len(chunk_embeddings)} chunks to ChromaDB...")
                try:
                    collection.add(
                        embeddings=chunk_embeddings,
                        documents=chunk_documents,
                        metadatas=chunk_metadatas,
                        ids=chunk_ids
                    )
                    logger.info(f"Added {len(chunk_embeddings)} chunks successfully")
                except Exception as batch_add_error:
                    logger.error(f"ChromaDB batch add error: {str(batch_add_error)}")
                    logger.error(f"Number of embeddings: {len(chunk_embeddings)}")
                    logger.error(f"Number of documents: {len(chunk_documents)}")
                    logger.error(f"Number of metadatas: {len(chunk_metadatas)}")
                    logger.error(f"Number of ids: {len(chunk_ids)}")
                    if chunk_embeddings:
                        logger.error(f"First embedding type: {type(chunk_embeddings[0])}, length: {len(chunk_embeddings[0])}")
                    raise
            
            chunks_created = len(chunk_embeddings)
        
        logger.info(f"Embedding completed! Created {chunks_created} embeddings")
        
        # Verify embeddings were stored
        try:
            verify_docs = collection.get(where={"document_path": pdf_path})
            logger.info(f"Verification: {len(verify_docs['ids'])} documents stored")
        except Exception as e:
            logger.warning(f"Could not verify embeddings: {str(e)}")
        
        return True, f"Successfully embedded document with {chunks_created} chunks"
        
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        return False, f"Embedding failed: {str(e)}"


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Upload and embed PDF files into a vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python upload_and_embed_pdf.py /path/to/document.pdf
  python upload_and_embed_pdf.py ~/Documents/manual.pdf
        """
    )
    
    parser.add_argument(
        "pdf_path",
        help="Absolute path to the PDF file to process"
    )
    
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Skip PDF compression step"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate PDF path
    pdf_path = pathlib.Path(args.pdf_path)
    
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    if not pdf_path.is_file():
        logger.error(f"Path is not a file: {args.pdf_path}")
        sys.exit(1)
    
    if not str(pdf_path).lower().endswith('.pdf'):
        logger.error(f"File is not a PDF: {args.pdf_path}")
        sys.exit(1)
    
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Step 1: Compress PDF (optional)
    if not args.no_compress:
        compress_pdf(str(pdf_path))
    
    # Step 2: Convert PDF to text
    logger.info("Converting PDF to text...")
    success, text_content, error_message = convert_pdf_to_text(str(pdf_path))
    
    if not success:
        logger.error(f"PDF conversion failed: {error_message}")
        sys.exit(1)
    
    logger.info(f"Successfully extracted {len(text_content)} characters of text")
    
    # Step 3: Embed and store in vector database
    logger.info("Embedding document...")
    embed_success, embed_message = embed_and_store_document(str(pdf_path), text_content)
    
    if not embed_success:
        logger.error(f"Embedding failed: {embed_message}")
        sys.exit(1)
    
    logger.info(f"Success! {embed_message}")
    logger.info(f"Document '{pdf_path.name}' has been embedded and stored in the vector database")


if __name__ == "__main__":
    main()
