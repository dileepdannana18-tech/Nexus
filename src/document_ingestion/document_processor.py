"""Document processing module for loading and splitting documents"""

from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load document(s) from a URL"""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from all PDFs inside a directory"""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a single PDF file"""
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a TXT file"""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()
    
    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, PDF directories, PDF files, or TXT files

        Args:
            sources: List of URLs, file paths, or folder paths

        Returns:
            List of loaded documents
        """
        docs: List[Document] = []
        for src in sources:
            # 1. Handle URLs
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
                continue

            path = Path(src)
            
            # 2. Handle Directories (e.g., folder of PDFs)
            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))
            
            # 3. Handle Files based on extension
            elif path.is_file():
                suffix = path.suffix.lower()
                if suffix == ".pdf":
                    docs.extend(self.load_from_pdf(path))
                elif suffix == ".txt":
                    docs.extend(self.load_from_txt(path))
                else:
                    raise ValueError(f"Unsupported file extension: {suffix}")
            else:
                raise ValueError(
                    f"Unsupported or non-existent source path: {src}. "
                    "Use a valid URL, file path, or directory."
                )
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split documents
        """
        return self.splitter.split_documents(documents)
    
    def process_sources(self, sources: List[str]) -> List[Document]:
        """
        Complete pipeline to load and split documents from various sources
        
        Args:
            sources: List of URLs, file paths, or directories to process
            
        Returns:
            List of processed document chunks
        """
        docs = self.load_documents(sources)
        return self.split_documents(docs)