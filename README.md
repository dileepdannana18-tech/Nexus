# 🤖 NexusRAG

**NexusRAG** is an intelligent, containerized Agentic Retrieval-Augmented Generation (RAG) system built with **LangChain**, **LangGraph**, **Google Gemini API**, and **Streamlit**. It seamlessly connects document ingestion, semantic vector indexing, and agentic workflows to deliver precise, context-aware answers through an interactive web UI.

---

## 🚀 Features

* **LangGraph Workflow**: Orchestrates retrieval and generation steps dynamically as a stateful graph.
* **Google Gemini Integration**: Powered by `gemini-1.5-flash` for high-speed reasoning and text generation, coupled with Google GenAI embeddings (`gemini-embedding-2-preview`).
* **Efficient Vector Search**: Utilizes **FAISS** for fast and accurate similarity matching over document chunks.
* **Interactive Streamlit UI**: Provides an intuitive search interface, expandable source inspection, and recent query history tracking.
* **Dockerized Deployment**: Fully containerized for smooth, environment-agnostic execution.

---

## 🛠️ Tech Stack

* **Orchestration**: LangGraph, LangChain
* **LLM & Embeddings**: Google Gemini API (`gemini-1.5-flash`, `gemini-embedding-2-preview`)
* **Vector Store**: FAISS
* **Frontend**: Streamlit
* **Containerization**: Docker

---

## 📂 Project Structure

```text
NexusRAG/
├── src/
│   ├── config/              # Configuration and LLM initialization
│   ├── document_ingestion/  # Document loading and text splitting
│   ├── graph_builder/       # LangGraph workflow definition
│   ├── node/                # RAG graph nodes (retrieve & generate)
│   ├── state/               # LangGraph state schema definition
│   └── vectorstore/         # FAISS vector store management
├── .env                     # Environment variables (API keys)
├── Dockerfile               # Docker container configuration
├── requirements.txt         # Python dependencies
└── streamlit_app.py         # Main Streamlit UI entry point