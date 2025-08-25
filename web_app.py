#!/usr/bin/env python3
"""
Web Interface for LLM-Embed RAG System

This Flask web application provides a user-friendly browser interface for:
1. Asking questions about embedded documents (chat interface)
2. Uploading new PDF documents 
3. Viewing document upload status and progress

The web app reuses the core logic from query_documents.py and upload_and_embed_pdf.py
but wraps it in an easy-to-use web interface accessible from any browser.

Usage:
    python3 web_app.py
    
Then open your browser to: http://localhost:5000
"""

import os
import json
import pathlib
import logging
import threading
import time
from typing import List, Dict, Any, Optional

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

# Import our existing RAG functionality
# We'll reuse the core functions from our command-line scripts
import chromadb
from chromadb.config import Settings
import ollama

# =================================================================
# CONFIGURATION 
# =================================================================

# Model configuration - same as our standalone scripts
EMBEDDING_MODEL = "qwen3:0.6b"
CHAT_MODEL = "qwen3:0.6b"

# Database configuration  
CHROMA_DB_PATH = "./chromadb_storage"
COLLECTION_NAME = "pdf_documents"

# Web app configuration
UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

# Search configuration
MAX_RESULTS = 3
MAX_CONTEXT_CHARS = 8000

# =================================================================
# FLASK APP SETUP
# =================================================================

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'  # Change this in production!
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global storage for tracking upload progress
upload_progress = {}

# =================================================================
# CORE RAG FUNCTIONS (reused from our scripts)
# =================================================================

def get_chromadb_client():
    """Get ChromaDB client for document search."""
    chromadb_path = pathlib.Path(CHROMA_DB_PATH)
    if not chromadb_path.exists():
        return None
    return chromadb.PersistentClient(path=str(chromadb_path), settings=Settings(allow_reset=True))

def unwrap_embedding(embedding_response):
    """Unwrap nested embedding structure from Ollama."""
    embedding = embedding_response["embeddings"]
    while isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
        embedding = embedding[0]
    return embedding

def search_documents(query_text: str, max_results: int = MAX_RESULTS) -> List[Dict[str, Any]]:
    """Search for relevant documents using semantic similarity."""
    try:
        # Generate embedding for the query
        embedding_response = ollama.embed(model=EMBEDDING_MODEL, input=query_text)
        query_embedding = unwrap_embedding(embedding_response)
        
        # Get ChromaDB client and collection
        client = get_chromadb_client()
        if not client:
            return []
            
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        
        # Perform similarity search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        documents = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                doc = {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"][0] else {},
                    "distance": results["distances"][0][i] if results["distances"][0] else None
                }
                documents.append(doc)
        
        return documents
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        return []

def build_context_message(query: str, documents: List[Dict[str, Any]]) -> str:
    """Build context message for AI chat using retrieved documents."""
    if not documents:
        return f"""You are a helpful assistant to help user with their scientific questions. The user asked: "{query}"

I couldn't find any relevant documents in the database to help answer this question. 
Please provide a general response and suggest that the user might need to upload relevant documents first."""

    # Combine document content
    combined_content = ""
    doc_sources = []
    
    for doc in documents:
        content = doc["content"]
        doc_name = doc["metadata"].get("document_name", "Unknown Document")
        
        if doc_name not in doc_sources:
            doc_sources.append(doc_name)
        
        combined_content += f"\n\n--- Document: {doc_name} ---\n"
        combined_content += content
    
    # Truncate if too long
    if len(combined_content) > MAX_CONTEXT_CHARS:
        combined_content = combined_content[:MAX_CONTEXT_CHARS-100] + "...\n\n[Content truncated for length]"
    
    # Build context message
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

    return context_message

def generate_response(context_message: str) -> str:
    """Generate AI response using the qwen model with document context."""
    try:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": context_message}],
            options={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 2048,
                "num_ctx": 4096
            }
        )
        
        return response["message"]["content"]
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"I apologize, but I encountered an error while generating a response: {str(e)}"

# =================================================================
# HELPER FUNCTIONS
# =================================================================

def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_document_count():
    """Get count of embedded documents."""
    try:
        client = get_chromadb_client()
        if not client:
            return 0
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        return collection.count()
    except:
        return 0

# =================================================================
# WEB ROUTES
# =================================================================

@app.route('/')
def index():
    """Main chat interface page."""
    doc_count = get_document_count()
    return render_template('index.html', doc_count=doc_count)

@app.route('/upload')
def upload_page():
    """Document upload page."""
    return render_template('upload.html')

@app.route('/api/query', methods=['POST'])
def api_query():
    """API endpoint for asking questions."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Search for relevant documents
        documents = search_documents(query)
        
        if not documents:
            response_text = "I couldn't find any relevant documents to answer your question. Please make sure you have uploaded and processed some PDF documents first."
            return jsonify({
                'response': response_text,
                'sources': [],
                'found_documents': 0
            })
        
        # Build context and generate response
        context_message = build_context_message(query, documents)
        response_text = generate_response(context_message)
        
        # Extract source information
        sources = []
        for doc in documents:
            doc_name = doc["metadata"].get("document_name", "Unknown")
            chunk_info = ""
            if "chunk_index" in doc["metadata"]:
                chunk_info = f" (chunk {doc['metadata']['chunk_index'] + 1})"
            sources.append(f"{doc_name}{chunk_info}")
        
        return jsonify({
            'response': response_text,
            'sources': sources,
            'found_documents': len(documents)
        })
        
    except Exception as e:
        logger.error(f"Error in query API: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """API endpoint for uploading PDF files."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(filepath)
        
        # Start background processing
        upload_id = f"upload_{timestamp}"
        upload_progress[upload_id] = {
            'status': 'uploaded',
            'filename': filename,
            'message': 'File uploaded successfully. Processing will start shortly.'
        }
        
        # Start processing in background thread
        thread = threading.Thread(target=process_pdf_background, args=(filepath, filename, upload_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'upload_id': upload_id,
            'message': 'File uploaded successfully. Processing started.'
        })
        
    except Exception as e:
        logger.error(f"Error in upload API: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_status/<upload_id>')
def api_upload_status(upload_id):
    """API endpoint to check upload/processing status."""
    if upload_id in upload_progress:
        return jsonify(upload_progress[upload_id])
    else:
        return jsonify({'error': 'Upload ID not found'}), 404

def process_pdf_background(filepath: str, original_filename: str, upload_id: str):
    """Process PDF in background thread (simplified version of upload_and_embed_pdf.py logic)."""
    try:
        # Update status
        upload_progress[upload_id] = {
            'status': 'processing',
            'filename': original_filename,
            'message': 'Converting PDF to text...'
        }
        
        # This is a simplified version - in a real implementation you'd want to 
        # import and use the full logic from upload_and_embed_pdf.py
        # For now, we'll just simulate processing
        
        import subprocess
        import sys
        
        # Call our existing upload script
        result = subprocess.run([
            sys.executable, 'upload_and_embed_pdf.py', filepath
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            upload_progress[upload_id] = {
                'status': 'completed',
                'filename': original_filename,
                'message': 'Document processed and embedded successfully!'
            }
        else:
            upload_progress[upload_id] = {
                'status': 'error',
                'filename': original_filename,
                'message': f'Error processing document: {result.stderr}'
            }
        
        # Clean up uploaded file after processing
        try:
            os.remove(filepath)
        except:
            pass
            
    except Exception as e:
        upload_progress[upload_id] = {
            'status': 'error',
            'filename': original_filename,
            'message': f'Processing error: {str(e)}'
        }

# =================================================================
# MAIN EXECUTION
# =================================================================

if __name__ == '__main__':
    # Check if ChromaDB exists
    chromadb_path = pathlib.Path(CHROMA_DB_PATH)
    if not chromadb_path.exists():
        logger.warning("ChromaDB storage not found. You'll need to upload some documents first.")
    
    # Check if Ollama is accessible
    try:
        ollama.list()
        logger.info("Ollama connection successful")
    except Exception as e:
        logger.error(f"Cannot connect to Ollama: {e}")
        logger.error("Make sure Ollama is running with: docker ps")
    
    print("\n" + "="*60)
    print("LLM-Embed Web Interface Starting...")
    print("="*60)
    print("Open your browser to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-"*60)
    
    # Run Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000)
