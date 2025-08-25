# PDF Upload and Embedding Script Usage

## Overview

The `upload_and_embed_pdf.py` script is a standalone CLI tool that processes PDF files and stores them in a vector database for AI-powered document search and retrieval.

## Features

**PDF Processing:**
- PDF compression using pikepdf to optimize file size
- Text extraction using pdfplumber with table support
- Memory-efficient batch processing for large documents
- Intelligent chunking for optimal embedding performance

**Vector Database Integration:**
- Embeddings generated using Ollama (qwen3:0.6b model)
- Storage in ChromaDB with persistent collections
- Automatic duplicate handling and cleanup
- Metadata tracking for document management

**CLI Interface:**
- Simple command-line interface with help documentation
- Verbose logging options for debugging
- Optional PDF compression skipping
- Comprehensive error handling

## Prerequisites

1. **Ollama must be running** with required models:
   ```bash
   docker ps  # Verify ollama container is running
   docker exec ollama ollama list  # Should show qwen3:0.6b model
   ```

2. **Python virtual environment** with dependencies:
   ```bash
   source .venv/bin/activate  # Activate virtual environment
   ```

## Usage

### Basic Usage

```bash
python3 upload_and_embed_pdf.py /path/to/your/document.pdf
```

### Advanced Options

```bash
# Skip PDF compression (faster but larger files)
python3 upload_and_embed_pdf.py --no-compress /path/to/document.pdf

# Enable verbose logging for debugging
python3 upload_and_embed_pdf.py --verbose /path/to/document.pdf

# Combine options
python3 upload_and_embed_pdf.py --no-compress --verbose /path/to/document.pdf
```

### Examples

```bash
# Process a technical manual
python3 upload_and_embed_pdf.py ~/Documents/user_manual.pdf

# Process an academic paper with verbose output
python3 upload_and_embed_pdf.py -v ~/Downloads/research_paper.pdf

# Process a large document without compression
python3 upload_and_embed_pdf.py --no-compress ~/Documents/large_report.pdf
```

## What the Script Does

### Step 1: PDF Validation
- Checks if file exists and is a valid PDF
- Validates file size (max 100MB)
- Ensures readable format

### Step 2: PDF Compression (Optional)
- Uses pikepdf to optimize PDF file size
- Maintains document quality while reducing storage requirements
- Can be skipped with `--no-compress` flag

### Step 3: Text Extraction
- Extracts text content using pdfplumber
- Processes pages in batches to manage memory
- Extracts tables and structured content
- Handles various PDF formats and encodings

### Step 4: Embedding Strategy
- **Small documents (≤15,000 chars)**: Single embedding for complete document
- **Large documents (>15,000 chars)**: Intelligent chunking with overlap
- Maximum 5 chunks per document for optimal performance
- Each chunk maintains context and metadata

### Step 5: Vector Database Storage
- Stores embeddings in ChromaDB collection named "pdf_documents"
- Automatically removes existing entries for the same document
- Includes comprehensive metadata:
  - Document path and name
  - Embedding timestamp
  - Text length and chunk information
  - Processing strategy used

## Output and Verification

### Success Output
```
Processing PDF: /path/to/document.pdf
Converting PDF to text...
Successfully extracted 25,432 characters of text
Embedding document...
Success! Successfully embedded document with 3 chunks
Document 'document.pdf' has been embedded and stored in the vector database
```

### Verification
The script creates a local ChromaDB database at `./chromadb_storage/` containing your document embeddings.

## Configuration

### Default Settings
```python
EMBEDDING_MODEL = "qwen3:0.6b"
CHROMA_DB_PATH = "./chromadb_storage"
COLLECTION_NAME = "pdf_documents"
MAX_CHUNKS = 5
TARGET_CHUNK_SIZE = 4000
```

### Customization
You can modify these settings at the top of the script:
- Change embedding model (requires model to be available in Ollama)
- Adjust chunk size and count for different document types
- Modify database storage location
- Change collection name for organization

## File Limitations

- **Maximum file size**: 100MB
- **Maximum pages**: 1,000 pages
- **Supported format**: PDF only
- **Memory requirement**: ~8GB RAM recommended for large documents

## Troubleshooting

### Common Issues

1. **"No module named 'pikepdf'"**
   ```bash
   source .venv/bin/activate
   pip install pikepdf pdfplumber chromadb ollama pandas
   ```

2. **"Connection refused" when embedding**
   ```bash
   docker ps  # Check if ollama container is running
   docker start ollama  # Start if stopped
   ```

3. **"Model not found: qwen3:0.6b"**
   ```bash
   docker exec ollama ollama pull qwen3:0.6b
   ```

4. **"PDF file too large"**
   - Reduce PDF file size using external tools
   - Split large PDFs into smaller sections
   - Use `--no-compress` to skip compression step

5. **Memory issues with large PDFs**
   - Close other applications to free RAM
   - Process smaller batches
   - Consider splitting the document

### Debug Mode
Use the `--verbose` flag to see detailed processing information:
```bash
python3 upload_and_embed_pdf.py --verbose /path/to/document.pdf
```

## Integration with AI Chat

Once documents are embedded, they can be used with AI chat systems that support ChromaDB vector search. The embeddings enable:

- **Semantic search**: Find relevant content based on meaning, not just keywords
- **Context retrieval**: Get relevant document sections for AI responses
- **RAG (Retrieval-Augmented Generation)**: Enhance AI responses with document knowledge

## Next Steps

After embedding your documents:

1. **Verify storage**: Check the `./chromadb_storage/` directory for your database
2. **Test search**: Use ChromaDB queries to test document retrieval
3. **Integrate with chat**: Connect the vector database to your AI chat application
4. **Scale up**: Process multiple documents to build a comprehensive knowledge base

## Support

For issues or questions:
- Check the verbose output for detailed error messages
- Verify Ollama container and models are working
- Ensure adequate system resources (RAM/storage)
- Review PDF file format and size constraints
