# LLM-Embed Setup Guide for macOS

This guide will walk you through setting up Ollama with Docker on macOS, including installing the required AI models for chat and embedding functionality.

## Prerequisites

- macOS (tested on macOS 14.6.0 and later)
- Terminal access
- Administrator privileges (for installations)

## Step 1: Install Homebrew (Package Manager)

If you don't have Homebrew installed, run this command in Terminal:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After installation, add Homebrew to your PATH:

```bash
echo >> ~/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Verify Homebrew is working:

```bash
brew --version
```

## Step 2: Install Docker Desktop

Install Docker Desktop using Homebrew:

```bash
brew install --cask docker
```

Start Docker Desktop:

```bash
open /Applications/Docker.app
```

Wait for Docker Desktop to fully start (you'll see the Docker icon in your menu bar), then verify it's working:

```bash
docker --version
docker ps
```

## Step 3: Run Ollama Container

Run the Ollama container with persistent storage:

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

Verify the container is running:

```bash
docker ps
```

You should see output showing the ollama container with status "Up".

## Step 4: Install Required AI Models

### Install Chat Model (qwen3:0.6b)

```bash
docker exec ollama ollama pull qwen3:0.6b
```

### Install Embedding Model (mxbai-embed-large)

```bash
docker exec ollama ollama pull mxbai-embed-large
```

## Step 5: Verify Installation

Check that both models are installed:

```bash
docker exec ollama ollama list
```

Test the chat model:

```bash
docker exec ollama ollama run qwen3:0.6b "Hello! Test message."
```

Test the API endpoint:

```bash
curl http://localhost:11434/api/tags
```

## Models Information

- **qwen3:0.6b** (~522 MB): Advanced and lightweight reasoning model for chat
- **mxbai-embed-large** (~669 MB): Embedding model for vector search and document context

## Quick Reference Commands

### Container Management
```bash
# Check container status
docker ps

# Stop Ollama container
docker stop ollama

# Start Ollama container
docker start ollama

# Restart Ollama container
docker restart ollama

# View container logs
docker logs ollama
```

### Model Management
```bash
# List installed models
docker exec ollama ollama list

# Pull a new model
docker exec ollama ollama pull <model-name>

# Remove a model
docker exec ollama ollama rm <model-name>

# Test chat with a model
docker exec ollama ollama run qwen3:0.6b "Your message here"
```

### API Testing
```bash
# List available models via API
curl http://localhost:11434/api/tags

# Test chat via API
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:0.6b",
  "prompt": "Hello, how are you?",
  "stream": false
}'

# Test embeddings via API
curl http://localhost:11434/api/embed -d '{
  "model": "mxbai-embed-large",
  "input": "Hello world"
}'
```

## Configuration

The application uses these models as configured in `config.yaml`:

```yaml
CHAT_MODEL: qwen3:0.6b
EMBEDDING_MODEL: mxbai-embed-large
```

## Troubleshooting

### Docker Desktop not starting
- Make sure Docker Desktop is running (check menu bar for Docker icon)
- Try restarting Docker Desktop: `open /Applications/Docker.app`

### Container won't start
- Check if port 11434 is already in use: `lsof -i :11434`
- Remove existing container: `docker rm -f ollama`
- Run the container command again

### Models won't download
- Check internet connection
- Ensure Docker container is running: `docker ps`
- Try pulling models individually
- Check disk space (models are large files)

### API not responding
- Verify container is running: `docker ps`
- Check if port is accessible: `curl http://localhost:11434/api/tags`
- Restart container: `docker restart ollama`

## Performance Notes

- **CPU vs GPU**: This setup uses CPU-only mode. For GPU acceleration on Apple Silicon Macs, consider installing Ollama natively instead of using Docker.
- **Memory Usage**: The models require significant RAM. Ensure your Mac has at least 8GB RAM for smooth operation.
- **Storage**: Each model requires several hundred MB of storage. Make sure to have adequate disk space.

## Next Steps

Once this setup is complete, you can:

1. Start your Flask application that uses these models
2. Test the RAG (Retrieval-Augmented Generation) functionality
3. Upload and process documents for embedding
4. Use the chat interface with document context

## Alternative Installation (Native Ollama)

If you prefer to install Ollama natively instead of Docker:

```bash
# Install Ollama directly
brew install ollama

# Start Ollama service
ollama serve

# Pull models (in a new terminal)
ollama pull qwen3:0.6b
ollama pull mxbai-embed-large
```