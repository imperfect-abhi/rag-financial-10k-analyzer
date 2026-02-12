# 🏦 Financial & Legal RAG Analyzer: SEC 10-K Deep Dive

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Colab](https://img.shields.io/badge/Colab-Run%20Notebook-orange.svg)](https://colab.research.google.com/drive/1urFDnyjk1CJZUZvG01IOK8InohJYyB4O?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"RAG system engineered for precision in financial document like 10-K filings analysis."**

## 🚀 Live Solution
The complete, end-to-end modular pipeline is hosted on Google Colab.
**[👉 RAG Notebook](https://colab.research.google.com/drive/1urFDnyjk1CJZUZvG01IOK8InohJYyB4O?usp=sharing)**

---

## 📖 Project Overview
This project implements a sophisticated **Retrieval-Augmented Generation (RAG)** system designed to answer complex financial and legal queries from **Apple’s 2024 10-K** and **Tesla’s 2023 10-K** filings. 

The system is built with a "First Principles" approach, favoring modularity, theoretical depth, and extreme hardware optimization to run high-performance LLMs on consumer-grade (T4) GPUs.

### **Core Design Principles**
*   **Modular Architecture:** Independent components for Ingestion, Embedding, Retrieval, Re-ranking, and Generation.
*   **Theoretical Rigor:** Every module includes a "Post-Mortem" style theoretical deep dive in the notebook.
*   **Memory Hygiene:** Implements strategic CPU-offloading for retrieval models to maximize GPU VRAM for LLM reasoning.
*   **Deterministic Accuracy:** Strict prompt engineering and two-stage retrieval to eliminate hallucinations.

---

## 🛠️ Technical Stack
*   **LLM:** `Microsoft/Phi-3-mini-128k-instruct` (4-bit Quantization).
*   **Embedding Model:** `BAAI/bge-small-en-v1.5` (State-of-the-art retrieval).
*   **Re-ranker:** `BAAI/bge-reranker-base` (Cross-Encoder precision).
*   **Vector Store:** `ChromaDB` (Persistent, local storage).
*   **Parsing/Chunking:** `pypdf` & `langchain-text-splitters` (Recursive Character splitting).
*   **Environment:** Python 3.12 (with custom NumPy 1.26.4 binary fix).

---

## 🏗️ System Architecture
The pipeline follows a **Two-Stage Retrieval** strategy:

1.  **Ingestion:** PDFs are parsed page-by-page, retaining metadata (Page #, Section), and split into 1000-character recursive chunks.
2.  **Vector Search (Bi-Encoder):** A high-speed semantic search identifies the top 25 candidate chunks from the vector database.
3.  **Neural Re-ranking (Cross-Encoder):** The 25 candidates are re-scored using a Cross-Encoder to prioritize the most factually relevant top 5 chunks.
4.  **Generation (LLM):** The top chunks are injected into a strict system prompt. The LLM synthesizes the answer, providing inline citations and adhering to "Out-of-Scope" refusal rules.

---

## 📁 Repository Structure
```bash
├── data/                    # Raw SEC 10-K PDF filings
├── notebooks/               # Local copy of the Colab Notebook
├── design_report.md         # Detailed architectural specification & justifications
├── requirements.txt         # Pinned dependencies for environment reproducibility
└── README.md                # Project documentation