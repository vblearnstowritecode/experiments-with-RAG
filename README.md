# Experiments with RAG

This repository contains my further experiments and implementations based on the excellent course **"Advanced Retrieval for AI"** from [DeepLearning.AI](https://learn.deeplearning.ai/courses/advanced-retrieval-for-ai).

## What's Inside

- **TCS RAG System**: A complete RAG implementation using the TCS Annual Report as a knowledge base
- **Evaluation Framework**: Systematic evaluation of RAG performance with metrics and analysis
- **Multiple RAG Approaches**: Basic RAG, Query Expansion, and Cross-encoder Re-ranking
- **Clean Python Module**: Modular RAG functions for easy testing and deployment
- **Comprehensive Testing**: 10-question evaluation set across difficulty levels

## Key Features

- ChromaDB for vector storage
- OpenAI GPT-4.1 integration for answer generation
- Three distinct RAG methodologies for comparison
- Cross-encoder re-ranking with sentence-transformers
- Judge LLM evaluation system for answer quality assessment
- Performance tracking and analysis
- Clean separation of learning materials and implementation

## RAG Approaches Implemented

1. **Basic RAG**: Simple semantic search with 3 chunks
2. **Query Expansion**: Multiple generated queries for comprehensive retrieval
3. **Cross-encoder Re-ranking**: Advanced ranking with `ms-marco-MiniLM-L-6-v2`

## Course Credit

This work builds upon concepts and techniques learned from the Advanced Retrieval for AI course by DeepLearning.AI. The course provided excellent foundational knowledge on embeddings, vector databases, and retrieval-augmented generation patterns.

## Current Status

✅ **Complete RAG Pipeline**: Fully operational with 1,324 document chunks
✅ **Three RAG Methods**: All approaches implemented and tested
✅ **Evaluation Framework**: 10-question test set with judge LLM scoring
✅ **Clean Architecture**: Modular `tcs_rag.py` with reusable functions
✅ **Performance Analysis**: Comprehensive evaluation and comparison system

## Getting Started

1. Install dependencies: `uv sync`
2. Set up your `.env` file with OpenAI API key
3. Run the main notebook: `jupyter lab tcs_rag_notebook.ipynb`
4. Test different approaches: `jupyter lab tcs_rag_evaluation.ipynb`
5. Use the clean module: `import tcs_rag; tcs_rag.basic_rag(question, collection, client)`

## Repository Structure

- `tcs_rag_notebook.ipynb` - Main RAG implementation and pipeline
- `tcs_rag_evaluation.ipynb` - Evaluation framework and testing
- `tcs_rag.py` - Clean Python module with reusable RAG functions
- `tcs_annual_report.md` - Extracted knowledge base content
- `pyproject.toml` - Dependencies and project configuration
- `tcs_rag_evaluation_*.csv` - Evaluation results and performance data