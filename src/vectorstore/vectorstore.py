"""Vector store module for document embedding and retrieval"""

import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

class VectorStore:
    """Manages vector store operations"""
    
    def __init__(self):
        """Initialize vector store with Google embeddings"""
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        # Using the current supported embedding model identifier
        self.embedding = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview",
            google_api_key=api_key
        )
        self.vectorstore = None
        self.retriever = None
    
    def create_vectorstore(self, documents: List[Document]):
        """Create vector store from documents"""
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever()
    
    def get_retriever(self):
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever.invoke(query)