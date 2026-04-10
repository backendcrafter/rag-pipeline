# RAG Pipeline

A Retrieval Augmented Generation pipeline built with LangChain and ChromaDB.

## How it works
1. `ingest.py` — loads documents, chunks them, embeds using HuggingFace, stores in ChromaDB
2. `query.py` — takes user query, retrieves relevant chunks, sends to Groq LLM for answer

## Tech Stack
- Python
- LangChain + ChromaDB
- HuggingFace Embeddings (all-mpnet-base-v2)
- Groq API (Llama 3.3 70B)

## How to Run
1. Add your Groq API key to `.env` file: `MY_API=your_key_here`
2. Install: `pip install langchain langchain-groq langchain-community langchain-chroma langchain-huggingface chromadb sentence-transformers python-dotenv`
3. Run `python ingest.py` once to build the vector store
4. Run `python query.py` to ask questions

## Author
Harsh Pratap Singh