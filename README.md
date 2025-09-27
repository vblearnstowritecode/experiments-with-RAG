# Experiments with RAG

This repository contains my further experiments and implementations based on the excellent course **"Advanced Retrieval for AI"** from [DeepLearning.AI](https://learn.deeplearning.ai/courses/advanced-retrieval-for-ai).

## What's Inside

- **TCS RAG System**: A complete RAG implementation using the TCS Annual Report as a knowledge base
- **Evaluation Framework**: Systematic evaluation of RAG performance with metrics and analysis
- **Query Expansion**: Experiments with different query enhancement techniques
- **Embeddings Pipeline**: Two-stage chunking and semantic search implementation

## Key Features

- ChromaDB for vector storage
- OpenAI integration for answer generation
- Comprehensive evaluation metrics
- Performance tracking and analysis
- Clean separation of learning materials and implementation

## Course Credit

This work builds upon concepts and techniques learned from the Advanced Retrieval for AI course by DeepLearning.AI. The course provided excellent foundational knowledge on embeddings, vector databases, and retrieval-augmented generation patterns.

## Getting Started

1. Install dependencies: `uv sync`
2. Set up your `.env` file with OpenAI API key
3. Run the notebook: `jupyter lab tcs_rag_notebook.ipynb`
4. Explore evaluation results in `tcs_rag_evaluation.ipynb`

## Repository Structure

- `tcs_rag_notebook.ipynb` - Main RAG implementation
- `tcs_rag_evaluation.ipynb` - Evaluation and analysis
- `tcs_annual_report.md` - Extracted knowledge base content
- `pyproject.toml` - Dependencies and project configuration