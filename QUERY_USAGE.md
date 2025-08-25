# Document Query System - RAG (Retrieval-Augmented Generation)

## Overview

The `query_documents.py` script implements a RAG system that allows you to query documents embedded using the `upload_and_embed_pdf.py` script. It performs semantic search in the ChromaDB vector database and uses the qwen model to generate contextual responses.

## How It Works

1. **User Input**: Takes a natural language query from the user
2. **Text-to-Vector**: Converts the query to an embedding using qwen3:0.6b
3. **Semantic Search**: Searches ChromaDB for similar document chunks
4. **Context Building**: Combines relevant chunks into context
5. **AI Response**: Uses qwen3:0.6b to generate an answer based on the context

## Prerequisites

1. **Documents must be embedded first**:
   ```bash
   python3 upload_and_embed_pdf.py /path/to/document.pdf
   ```

2. **Ollama container running** with qwen3:0.6b model:
   ```bash
   docker ps  # Verify ollama container is running
   docker exec ollama ollama list  # Should show qwen3:0.6b
   ```

3. **ChromaDB storage exists** at `./chromadb_storage/`

## Usage Modes

### Single Query Mode

Ask a specific question and get an immediate response:

```bash
# Basic query
python3 query_documents.py "What is the main topic of the document?"

# With verbose logging
python3 query_documents.py --verbose "How does the system work?"

# Retrieve more document chunks for context
python3 query_documents.py --max-results 5 "What are the key requirements?"
```

### Interactive Mode

Start a conversation-like interface for multiple queries:

```bash
python3 query_documents.py --interactive
```

In interactive mode:
- Type questions naturally
- Get responses based on your embedded documents
- Type `quit`, `exit`, or `q` to stop
- Type `help` for available commands

## Example Queries

**Technical Documentation:**
- "What are the system requirements?"
- "How do I install the software?"
- "What are the configuration options?"

**Research Papers:**
- "What is the main hypothesis?"
- "What methodology was used?"
- "What were the key findings?"

**Manuals:**
- "How do I troubleshoot connection issues?"
- "What safety precautions should I follow?"
- "Where can I find the warranty information?"

## Configuration

The script uses these default settings:

```python
EMBEDDING_MODEL = "qwen3:0.6b"     # Model for query embeddings
CHAT_MODEL = "qwen3:0.6b"          # Model for response generation
CHROMA_DB_PATH = "./chromadb_storage"  # Database location
COLLECTION_NAME = "pdf_documents"   # Collection name
MAX_RESULTS = 3                     # Document chunks to retrieve
MAX_CONTEXT_CHARS = 8000           # Maximum context length
```

## Output Format

### Single Query Example

```
Processing query: 'What is the main topic of the document?'
Searching for: 'What is the main topic of the document?'
Found: technical_manual.pdf (chunk 1) (similarity: 0.823)
Found: technical_manual.pdf (chunk 3) (similarity: 0.756)
Retrieved 2 relevant document chunks
Generating AI response...

================================================================================
AI Response:
================================================================================
Based on the provided documents, the main topic of the document is...

[Detailed response based on document content]
--------------------------------------------------------------------------------
Based on 2 relevant document chunk(s)
```

### Interactive Mode Example

```
============================================================
Document Query System - Interactive Mode
============================================================
Type your questions about the embedded documents.
Type 'quit', 'exit', or 'q' to stop.
------------------------------------------------------------

Query: What is the installation process?

Searching documents...
Found 3 relevant chunks. Generating response...

============================================================
AI Response:
============================================================
The installation process involves the following steps:
1. Download the software package...
2. Run the installer with administrator privileges...
[etc.]
------------------------------------------------------------
```

## Features

**Semantic Search:**
- Finds relevant content based on meaning, not just keywords
- Ranks results by similarity to your query
- Retrieves multiple document chunks for comprehensive context

**Intelligent Context Building:**
- Combines relevant chunks from multiple documents
- Truncates content if too long while preserving important information
- Tracks source documents for attribution

**AI Response Generation:**
- Uses document context to provide accurate answers
- Cites specific information from documents
- Indicates when information is not available in the documents

**User-Friendly Interface:**
- Simple command-line interface
- Interactive mode for multiple queries
- Verbose logging for debugging
- Clear error messages and guidance

## Performance Tips

**Optimize Query Results:**
- Use specific, detailed questions for better results
- Include key terms that might appear in your documents
- Try different phrasings if you don't get good results

**Adjust Parameters:**
- Increase `--max-results` for more comprehensive context
- Use `--verbose` to see which documents are being found
- Multiple shorter documents often work better than one very long document

## Troubleshooting

### Common Issues

1. **"ChromaDB storage not found"**
   ```bash
   # You need to embed documents first
   python3 upload_and_embed_pdf.py /path/to/document.pdf
   ```

2. **"No relevant documents found"**
   - Try different keywords or phrasing
   - Check if your documents contain information related to the query
   - Verify documents were embedded successfully

3. **"Connection refused" when generating response**
   ```bash
   docker ps  # Check if ollama container is running
   docker start ollama  # Start if stopped
   ```

4. **Poor search results**
   - Use more specific queries
   - Include important keywords from your documents
   - Try increasing `--max-results` for more context

### Debug Mode

Use `--verbose` to see detailed information about the search process:

```bash
python3 query_documents.py --verbose "your question here"
```

This shows:
- Query embedding generation
- Document search results and similarity scores
- Context building process
- AI response generation status

## Integration

This script works seamlessly with:

**Document Embedding Pipeline:**
1. `upload_and_embed_pdf.py` - Embed PDF documents
2. `query_documents.py` - Query embedded documents

**Batch Processing:**
```bash
# Embed multiple documents
for pdf in *.pdf; do
    python3 upload_and_embed_pdf.py "$pdf"
done

# Query the collection
python3 query_documents.py --interactive
```

## Advanced Usage

**Custom Configuration:**
Edit the script to modify:
- Model selection (if you have other models available)
- Database path and collection names
- Context length and result limits
- AI response parameters (temperature, etc.)

**Programmatic Usage:**
The script functions can be imported and used in other Python programs:

```python
from query_documents import search_documents, generate_response, build_context_message

# Search for documents
docs = search_documents("your query")

# Generate response
context = build_context_message("your query", docs)
response = generate_response(context)
```

## Next Steps

After setting up the query system:

1. **Test with your documents**: Try various types of questions
2. **Optimize queries**: Experiment with different phrasings
3. **Scale up**: Embed more documents to build a comprehensive knowledge base
4. **Integrate**: Use the RAG system as part of larger applications
