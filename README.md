# Local GitHub RAG

Local GitHub RAG is a Streamlit application that helps developers understand codebases through natural language queries using local LLMs. It clones GitHub repositories locally and implements RAG (Retrieval Augmented Generation) to provide context-aware responses about your code.

## Features

- 🔍 Clone and analyze GitHub repositories locally
- 💬 Ask questions about the codebase in natural language
- 🤖 Powered by local LLMs through Ollama
- 📝 Optional GitHub issues integration
- 🗄️ Persistent vector storage for quick repository switching
- 🔄 Support for multiple repositories
- 🎯 Efficient code context retrieval using multi-query generation

## Prerequisites

- Python 3.8+
- Ollama installed locally
- Git

## Required Models

The following models need to be available through Ollama:
- `qwen2.5-coder:14b` for code understanding and response generation
- `nomic-embed-text` for text embeddings

## Installation

```bash
# Clone the repository
git clone git@github.com:pixelsoccupied/local-github-rag.git
cd local-github-rag

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start the server
streamlit run main.py
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Choose one of three options:
   - Process a new repository by entering a GitHub URL
   - Use an already downloaded repository from the ./repos directory
   - Switch to a previously processed repository

3. Optional: Include GitHub issues in the analysis by providing a GitHub token

4. Start asking questions about your codebase!

## Environment Variables

- `GITHUB_TOKEN` (optional): For fetching GitHub issues and higher API rate limits

## How It Works

1. **Repository Processing**:
   - Clones the GitHub repository locally
   - Processes code files and optionally GitHub issues
   - Splits content into chunks for efficient retrieval
   - Creates embeddings using the nomic-embed-text model
   - Stores vectors in a local Chroma database

2. **Question Answering**:
   - Uses multi-query retrieval to find relevant code contexts
   - Generates comprehensive responses using the Qwen coder model
   - Provides specific code examples and issue references when applicable

## Supported File Types

- Python (.py)
- Markdown (.md)
- Text (.txt)
- Go (.go)
- JavaScript (.js)
- Java (.java)
- C++ (.cpp, .h)
- Rust (.rs)
- AsciiDoc (.adoc)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

- Built with Streamlit
- Powered by Ollama
- Uses LangChain for RAG implementation
- Vector storage by Chroma