#!/usr/bin/env python3
"""
Document Query and RAG (Retrieval-Augmented Generation) Script

This script implements a complete RAG system that allows you to query documents 
that have been embedded using the upload_and_embed_pdf.py script.

=== RAG PIPELINE OVERVIEW ===

1. RETRIEVAL PHASE:
   - User provides a natural language query
   - Query is converted to vector embedding using qwen3:0.6b model
   - ChromaDB performs semantic search to find most similar document chunks
   - Results are ranked by cosine similarity

2. CONTEXT BUILDING PHASE:
   - Retrieved document chunks are combined into a structured prompt
   - Context includes user query + relevant documents + instructions for AI
   - Content is truncated if needed to fit model limits

3. GENERATION PHASE:
   - Complete context is sent to qwen3:0.6b chat model
   - AI generates response based on document content
   - Response is contextually grounded in the retrieved documents

=== TECHNICAL ARCHITECTURE ===

Dependencies:
- ollama: For embedding generation and chat completion
- chromadb: For vector similarity search
- pathlib: For file system operations

Key Components:
- search_documents(): Handles vector search and retrieval
- build_context_message(): Formats documents into AI prompt
- generate_response(): Gets final answer from language model
- query_documents_interactive(): Provides chat-like interface

=== USAGE EXAMPLES ===

Single query mode:
    python query_documents.py "What is the main topic of the document?"

Interactive chat mode:
    python query_documents.py --interactive

With verbose logging:
    python query_documents.py --verbose "How does the system work?"

=== PREREQUISITES ===

1. Ollama container must be running with qwen3:0.6b model
2. Documents must be embedded first using upload_and_embed_pdf.py
3. ChromaDB storage must exist at ./chromadb_storage/
"""

import argparse
import logging
import pathlib
import sys
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
import ollama


# =================================================================
# CONFIGURATION CONSTANTS
# =================================================================
# These settings control the RAG system behavior and should match 
# the configuration used in upload_and_embed_pdf.py

# Model configuration - both use the same small, CPU-friendly model
EMBEDDING_MODEL = "qwen3:0.6b"     # Model for converting text to vectors
CHAT_MODEL = "qwen3:0.6b"          # Model for generating responses

# Database configuration - must match upload script
CHROMA_DB_PATH = "./chromadb_storage"    # Where vector embeddings are stored
COLLECTION_NAME = "pdf_documents"        # ChromaDB collection name

# Search and context limits
MAX_RESULTS = 3                    # Number of document chunks to retrieve per query
MAX_CONTEXT_CHARS = 8000          # Maximum characters to send to AI model (to fit context window)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_chromadb_client():
    """
    Get ChromaDB client (matches upload script configuration).
    
    This function establishes connection to the persistent ChromaDB database
    where our embedded document vectors are stored.
    """
    # Check if the ChromaDB storage directory exists
    # This is where our vector embeddings are stored persistently
    chromadb_path = pathlib.Path(CHROMA_DB_PATH)
    if not chromadb_path.exists():
        logger.error(f"ChromaDB storage not found at {chromadb_path}")
        logger.error("Please run upload_and_embed_pdf.py first to embed some documents")
        sys.exit(1)
    
    # Create and return a persistent ChromaDB client
    # This connects to the same database used by upload_and_embed_pdf.py
    return chromadb.PersistentClient(path=str(chromadb_path), settings=Settings(allow_reset=True))


def unwrap_embedding(embedding_response):
    """
    Unwrap nested embedding structure from Ollama - matches ai_assistant.py logic.
    
    Ollama sometimes returns embeddings wrapped in nested lists like [[[1,2,3,...]]]
    This function flattens it to get the actual vector [1,2,3,...]
    """
    # Extract the "embeddings" field from Ollama's response
    embedding = embedding_response["embeddings"]
    
    # Handle nested list structures by unwrapping layers
    # Keep unwrapping until we get to the actual vector (not nested lists)
    while isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
        embedding = embedding[0]
    
    # Return the actual embedding vector (list of numbers)
    return embedding


def search_documents(query_text: str, max_results: int = MAX_RESULTS) -> List[Dict[str, Any]]:
    """
    Search for relevant documents using semantic similarity.
    
    This is the core RAG (Retrieval) function that:
    1. Converts user query to vector embedding
    2. Searches ChromaDB for most similar document chunks
    3. Returns ranked results by similarity
    
    Args:
        query_text: The user's query text (e.g., "What is the main topic?")
        max_results: Maximum number of results to return (default: 3)
        
    Returns:
        List of dictionaries containing document chunks and metadata
    """
    logger.info(f"Searching for: '{query_text}'")
    
    try:
        # STEP 1: Convert user query to vector embedding
        # This uses the same model that was used to embed the documents
        logger.debug("Generating query embedding...")
        embedding_response = ollama.embed(model=EMBEDDING_MODEL, input=query_text)
        query_embedding = unwrap_embedding(embedding_response)
        
        # STEP 2: Connect to ChromaDB and get the document collection
        # This is where all our PDF document chunks are stored as vectors
        client = get_chromadb_client()
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        
        # STEP 3: Perform vector similarity search
        # ChromaDB compares our query vector to all document vectors
        # and returns the most similar ones ranked by distance
        logger.debug(f"Searching ChromaDB for {max_results} similar documents...")
        results = collection.query(
            query_embeddings=[query_embedding],  # Our query as a vector
            n_results=max_results,               # How many top results to return
            include=['documents', 'metadatas', 'distances']  # What data to include
        )
        
        # STEP 4: Process and format the search results
        documents = []
        if results["ids"] and results["ids"][0]:  # Check if we got any results
            for i in range(len(results["ids"][0])):
                # Extract information for each found document chunk
                doc = {
                    "id": results["ids"][0][i],                    # Unique chunk ID
                    "content": results["documents"][0][i],         # The actual text content
                    "metadata": results["metadatas"][0][i] if results["metadatas"][0] else {},  # Document info
                    "distance": results["distances"][0][i] if results["distances"][0] else None  # Similarity score
                }
                documents.append(doc)
                
                # STEP 5: Log what we found for debugging
                doc_name = doc["metadata"].get("document_name", "Unknown")
                chunk_info = ""
                if "chunk_index" in doc["metadata"]:
                    chunk_info = f" (chunk {doc['metadata']['chunk_index'] + 1})"
                
                # Calculate similarity: lower distance = higher similarity
                # Note: ChromaDB distance can be negative for some metrics
                similarity_score = 1 - doc['distance'] if doc['distance'] is not None else 0
                logger.info(f"Found: {doc_name}{chunk_info} (similarity: {similarity_score:.3f})")
        
        logger.info(f"Retrieved {len(documents)} relevant document chunks")
        return documents
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        return []  # Return empty list if search fails


def build_context_message(query: str, documents: List[Dict[str, Any]]) -> str:
    """
    Build a context message for the AI chat using retrieved documents.
    
    This function takes the retrieved document chunks and formats them into
    a structured prompt that tells the AI model how to use the document content
    to answer the user's question.
    
    Args:
        query: The user's original query
        documents: List of retrieved document chunks from vector search
        
    Returns:
        Formatted context message for the AI model (the full RAG prompt)
    """
    # CASE 1: No relevant documents found
    if not documents:
        return f"""You are a helpful assistant to help user with their scientific questions. The user asked: "{query}"

I couldn't find any relevant documents in the database to help answer this question. 
Please provide a general response and suggest that the user might need to upload relevant documents first."""

    # CASE 2: We have relevant documents - build the RAG context
    
    # STEP 1: Combine all document chunks into one text block
    combined_content = ""
    doc_sources = []  # Track which documents we're using
    
    for doc in documents:
        content = doc["content"]  # The actual text from the document chunk
        doc_name = doc["metadata"].get("document_name", "Unknown Document")
        
        # Keep track of unique document sources for attribution
        if doc_name not in doc_sources:
            doc_sources.append(doc_name)
        
        # Add each document chunk with a clear separator
        combined_content += f"\n\n--- Document: {doc_name} ---\n"
        combined_content += content
    
    # STEP 2: Truncate content if it's too long for the AI model
    # This prevents the context from exceeding model limits
    if len(combined_content) > MAX_CONTEXT_CHARS:
        logger.info(f"Truncating context from {len(combined_content)} to {MAX_CONTEXT_CHARS} characters")
        combined_content = combined_content[:MAX_CONTEXT_CHARS-100] + "...\n\n[Content truncated for length]"
    
    # STEP 3: Build the complete RAG prompt
    # This tells the AI model exactly how to use the document content
    context_message = f"""You are a knowledgeable AI assistant with access to document content, to help users with their scientific questions. Answer the user's question based on the provided documents.

USER QUESTION: "{query}"

RELEVANT DOCUMENTS:
{combined_content}

INSTRUCTIONS:
- Answer the question using information from the provided documents
- If the documents don't contain enough information, say so clearly
- Cite specific information from the documents when possible
- Be concise but thorough
- If multiple documents are referenced, mention which document contains specific information

SOURCES: {', '.join(doc_sources)}"""

    logger.info(f"Built context with {len(combined_content)} characters from {len(doc_sources)} document(s)")
    return context_message


def generate_response(context_message: str) -> str:
    """
    Generate AI response using the qwen model (on CPU) with document context.
    
    This is the final step in the RAG pipeline - the "Generation" part.
    It sends the context (user query + relevant documents) to the AI model
    and gets back a contextually-aware response.
    
    Args:
        context_message: The complete RAG prompt with query and retrieved documents
        
    Returns:
        Generated response from the AI model based on the document context
    """
    logger.info("Generating AI response...")
    
    try:
        # STEP 1: Send the context to the qwen model via Ollama
        # The context_message contains: user query + relevant document chunks + instructions
        response = ollama.chat(
            model=CHAT_MODEL,  # Using qwen3:0.6b for both embedding and chat
            messages=[{"role": "user", "content": context_message}],
            options={
                # Model parameters for response generation
                "temperature": 0.7,      # Controls creativity (0=deterministic, 1=creative)
                "top_p": 0.9,           # Nucleus sampling, consider top 90% of probability mass
                "top_k": 40,            # Consider only top 40 most likely next tokens
                "num_predict": 2048,    # Maximum number of tokens to generate
                "num_ctx": 4096         # Context window size (how much text model can see)
            }
        )
        
        # STEP 2: Extract the generated text from the response
        generated_text = response["message"]["content"]
        logger.info("AI response generated successfully")
        return generated_text
        
    except Exception as e:
        # STEP 3: Handle any errors gracefully
        logger.error(f"Error generating response: {str(e)}")
        return f"I apologize, but I encountered an error while generating a response: {str(e)}"


def query_documents_interactive():
    """Interactive mode for continuous querying."""
    print("\n" + "="*60)
    print("Document Query System in Interactive Mode")
    print("="*60)
    print("Type your questions about the embedded documents.")
    print("Type 'quit', 'exit', or 'q' to stop.")
    print("Type 'help' for available commands.")
    print("-"*60)
    
    while True:
        try:
            user_input = input("\nQuery: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if user_input.lower() == 'help':
                print("""
Available commands:
- Type any question to search the documents
- 'quit', 'exit', 'q' - Exit the program
- 'help' - Show this help message

Example queries:
- "What is the main topic discussed?"
- "What are the main methods used in the document?"
- "What are the key requirements to conduct this research?"
""")
                continue
            
            # Process the query
            print(f"\nSearching documents...")
            documents = search_documents(user_input)
            
            if not documents:
                print("No relevant documents found. Try a different query or check if documents are embedded.")
                continue
            
            print(f"Found {len(documents)} relevant chunks. Generating response...")
            
            # Build context and generate response
            context_message = build_context_message(user_input, documents)
            response = generate_response(context_message)
            
            # Display response
            print("\n" + "="*60)
            print("AI Response:")
            print("="*60)
            print(response)
            print("-"*60)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    """
    Main CLI function - orchestrates the complete RAG pipeline.
    
    This function handles command-line arguments and coordinates:
    1. Argument parsing and validation
    2. ChromaDB connection checking 
    3. Single query or interactive mode execution
    4. The complete RAG flow: Retrieval → Context Building → Generation
    """
    # STEP 1: Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Query embedded documents using RAG (Retrieval-Augmented Generation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query mode
  python query_documents.py "What is the main topic of the document?"
  
  # Interactive mode
  python query_documents.py --interactive
  
  # With verbose logging
  python query_documents.py --verbose "What are the key requirements to conduct this research?"
        """
    )
    
    # Define command-line arguments
    parser.add_argument(
        "query",
        nargs="?",  # Optional positional argument
        help="Query text to search for in the documents"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive query mode for continuous questions"
    )
    
    parser.add_argument(
        "--max-results", "-n",
        type=int,
        default=MAX_RESULTS,
        help=f"Maximum number of document chunks to retrieve (default: {MAX_RESULTS})"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging to see detailed search process"
    )
    
    # STEP 2: Parse arguments and configure logging
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # STEP 3: Validate that we have embedded documents to search
    chromadb_path = pathlib.Path(CHROMA_DB_PATH)
    if not chromadb_path.exists():
        logger.error(f"ChromaDB storage not found at {chromadb_path}")
        logger.error("Please run upload_and_embed_pdf.py first to embed some documents")
        sys.exit(1)
    
    # STEP 4: Choose execution mode based on arguments
    
    # Interactive mode - continuous question/answer session
    if args.interactive:
        query_documents_interactive()
        return
    
    # Single query mode - answer one question and exit
    if not args.query:
        logger.error("Please provide a query or use --interactive mode")
        parser.print_help()
        sys.exit(1)
    
    logger.info(f"Processing query: '{args.query}'")
    
    # STEP 5: Execute the complete RAG pipeline
    
    # RAG STEP 1: RETRIEVAL - Search for relevant documents
    documents = search_documents(args.query, max_results=args.max_results)
    
    if not documents:
        logger.error("No relevant documents found. Try a different query or check if documents are embedded.")
        sys.exit(1)
    
    # RAG STEP 2: CONTEXT BUILDING - Combine documents into prompt
    context_message = build_context_message(args.query, documents)
    
    # RAG STEP 3: GENERATION - Get AI response based on context
    response = generate_response(context_message)
    
    # STEP 6: Display the final results to the user
    print("\n" + "="*80)
    print("AI Response:")
    print("="*80)
    print(response)
    print("-"*80)
    print(f"Based on {len(documents)} relevant document chunk(s)")


if __name__ == "__main__":
    main()
