# Financial & Legal RAG System: Apple & Tesla 10-K Analysis

This repository contains a Retrieval-Augmented Generation (RAG) system designed to answer complex financial and legal questions from Apple's 2024 10-K and Tesla's 2023 10-K filings. 

## Live Colab Notebook
**[INSERT YOUR PUBLIC COLAB LINK HERE ONCE GENERATED]**

## Repository Structure
* `data/`: Contains the raw SEC 10-K PDF filings.
* `design_report.md`: The concept note explaining architectural choices (Chunking, LLM, Re-ranking).
* `requirements.txt`: Python dependencies required to run the pipeline.

## Tech Stack
* **Parsing:** `pypdf`
* **Embeddings:** `BAAI/bge-small-en-v1.5` (via `sentence-transformers`)
* **Vector Store:** `ChromaDB`
* **Re-ranker:** `BAAI/bge-reranker-base` (Cross-Encoder)
* **LLM:** Meta Llama-3-8B-Instruct (Quantized, Local Inference)