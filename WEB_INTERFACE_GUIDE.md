# Web Interface Guide

The LLM-Embed system includes a user-friendly web interface that makes it easy to upload documents and ask questions through your browser.

## Starting the Web Interface

1. **Make sure your environment is set up:**
```bash
source .venv/bin/activate
```

2. **Ensure Ollama is running:**
```bash
docker ps  # Should show ollama container running
```

3. **Start the web server:**
```bash
python3 web_app.py
```

4. **Open your browser to:**
```
http://localhost:5000
```

The web server will start on port 5000 and you'll see startup messages in the terminal.

## Using the Web Interface

### Main Chat Page

The main page (`http://localhost:5000`) provides a chat interface where you can:

**Ask Questions:**
- Type your question in the input box at the bottom
- Press Enter or click the send button
- See AI responses based on your uploaded documents
- View which documents were used as sources for each answer

**Track Statistics:**
- See how many documents you have uploaded
- Monitor how many questions you've asked
- Check system status (Ollama model and database status)

**Quick Actions:**
- Upload new documents
- Clear chat history
- Access example questions (when documents are available)

### Upload Page

The upload page (`http://localhost:5000/upload`) allows you to:

**Upload Documents:**
- Drag and drop PDF files onto the upload area
- Or click to browse and select files
- Upload multiple files at once
- Track upload and processing progress in real-time

**Monitor Processing:**
- See each file move through the processing pipeline:
  1. Upload: File uploaded to server
  2. Extract: Text extracted from PDF
  3. Chunk: Large documents split into sections  
  4. Embed: Text converted to vector embeddings
  5. Store: Embeddings saved to database

**Processing Status:**
- Uploading: File being uploaded
- Processing: Document being converted and embedded
- Completed: Ready for questions
- Error: Something went wrong (with error details)

## Features

### Real-time Processing
- Upload progress tracking
- Live status updates
- Background processing (you can navigate away while files process)

### Smart Chat Interface
- Message history
- Typing indicators
- Source attribution for answers
- Example questions when you have documents

### Responsive Design
- Works on desktop and mobile browsers
- Clean, modern interface
- Easy navigation between chat and upload

### File Management
- Automatic file validation
- Size and format checking
- Error handling with helpful messages
- Secure file processing

## Browser Requirements

The web interface works with modern browsers including:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

JavaScript must be enabled for full functionality.

## Security Notes

- Files are processed locally on your machine
- No data is sent to external services
- Upload directory is created automatically
- Files are cleaned up after processing
- Web interface runs on localhost only by default

## Troubleshooting

### Web Server Won't Start

**"Port already in use" error:**
```bash
# Kill any process using port 5000
lsof -ti:5000 | xargs kill -9
```

**Import errors:**
```bash
# Make sure you're in the virtual environment
source .venv/bin/activate
pip install -r requirements.txt
```

### Upload Issues

**File upload fails:**
- Check file size (max 100MB)
- Ensure file is a valid PDF
- Check disk space

**Processing gets stuck:**
- Check Ollama is running: `docker ps`
- Check logs in the terminal where you started web_app.py
- Try uploading a smaller file first

### Chat Issues

**"No relevant documents found":**
- Make sure you've uploaded and processed documents first
- Check that processing completed successfully
- Try different question phrasing

**AI responses are slow:**
- This is normal with CPU processing
- qwen3:0.6b is optimized for CPU but still takes time
- Responses typically take 10-30 seconds

### Browser Issues

**Interface not loading:**
- Check you're going to `http://localhost:5000`
- Ensure JavaScript is enabled
- Try refreshing the page
- Check browser console for errors (F12)

**Drag and drop not working:**
- Try clicking the upload area instead
- Check file types (PDF only)
- Ensure files aren't too large

## Advanced Usage

### Custom Port
To run on a different port, edit `web_app.py` and change:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

### Production Deployment
The current setup is for development. For production use:
- Change the Flask secret key
- Use a proper WSGI server like gunicorn
- Set up proper logging
- Configure firewalls and security

### Multiple Users
The current setup is designed for single-user local use. For multiple users, you'd need to add:
- User authentication
- File isolation
- Database separation
- Session management

## Integration with Command Line Tools

The web interface uses the same underlying functions as the command-line scripts:
- `upload_and_embed_pdf.py` - for document processing
- `query_documents.py` - for question answering

You can use both the web interface and command line tools with the same document database. Documents uploaded through either method will be available in both.

## Performance Tips

### For Better Upload Performance:
- Upload smaller files first to test the system
- Process one large file at a time
- Ensure adequate disk space and memory

### For Better Chat Performance:
- Use specific, focused questions
- If responses are slow, try shorter questions
- Clear chat history periodically to keep the interface responsive

### For System Performance:
- Close unnecessary applications while processing large files
- Monitor CPU and memory usage during uploads
- Keep the Ollama container running to avoid startup delays

The web interface provides a much more user-friendly way to interact with your document RAG system compared to command-line tools, making it accessible to users who prefer graphical interfaces.
