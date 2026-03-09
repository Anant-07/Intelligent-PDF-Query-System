"""
RAG Pipeline - Backend logic for Intelligent PDF Query System
Extracts and organizes PDF documents for semantic search
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv


class PDFProcessor:
    """Handles PDF document loading and processing"""
    
    def __init__(self, pdf_directory: str):
        self.pdf_dir = Path(pdf_directory)
    
    def process_all_pdfs(self) -> List[Document]:
        """Process all PDF files in the directory"""
        all_documents = []
        
        if not self.pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {self.pdf_dir}")
        
        pdf_files = list(self.pdf_dir.glob("**/*.pdf"))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in: {self.pdf_dir}")
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            
            try:
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()
                
                # Add metadata
                for doc in documents:
                    doc.metadata['source_file'] = pdf_file.name
                    doc.metadata['file_type'] = 'pdf'
                
                all_documents.extend(documents)
                print(f"Loaded {len(documents)} pages from {pdf_file.name}")
                
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {e}")
        
        print(f"\nTotal documents loaded: {len(all_documents)}")
        return all_documents


class TextSplitter:
    """Handles text chunking for better embedding performance"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        split_docs = self.splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        return split_docs


class EmbeddingManager:
    """Handles document embedding generation using Ollama embeddings"""
    
    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.model = None
        self.embedding_dim = None
        self._load_model()
    
    def _load_model(self):
        """Initialize Ollama embedding model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = OllamaEmbeddings(
                model=self.model_name,
                base_url=self.base_url
            )
            
            # Infer embedding dimension
            test_embedding = self.model.embed_query("test")
            self.embedding_dim = len(test_embedding)
            
            print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        if not texts:
            return np.empty((0, self.embedding_dim))
        
        embeddings = self.model.embed_documents(texts)
        return np.array(embeddings, dtype=np.float32)


class VectorStore:
    """Manages document embeddings in a FAISS vector store"""
    
    def __init__(
        self,
        index_path: str = "data/faiss_index",
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self.index_path = index_path
        self.embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)
        self.vectorstore = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Load or create FAISS index"""
        if os.path.exists(self.index_path):
            self.vectorstore = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"Vector store loaded. Documents: {self.vectorstore.index.ntotal}")
        else:
            os.makedirs(self.index_path, exist_ok=True)
            print("No existing index. New one will be created.")
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        if not documents:
            print("No documents to add")
            return
        
        texts = [doc.page_content for doc in documents]
        metadatas = []
        ids = []
        
        for i, doc in enumerate(documents):
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
                ids=ids,
            )
        else:
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        
        self.vectorstore.save_local(self.index_path)
        print(f"Added {len(documents)} documents. Total: {self.vectorstore.index.ntotal}")
    
    def is_ready(self) -> bool:
        """Check if vector store is ready"""
        return self.vectorstore is not None and self.vectorstore.index.ntotal > 0


class RAGRetriever:
    """Handles query-based retrieval from vector store"""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        
        if not self.vector_store.is_ready():
            return []
        
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        try:
            results = self.vector_store.vectorstore.similarity_search_with_score_by_vector(
                embedding=query_embedding,
                k=top_k,
            )
            
            retrieved_docs = []
            
            for rank, (doc, distance) in enumerate(results, start=1):
                similarity_score = 1 / (1 + distance)
                
                if similarity_score < score_threshold:
                    continue
                
                retrieved_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": similarity_score,
                    "distance": distance,
                    "rank": rank,
                })
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []


class RAGPipeline:
    """Complete RAG pipeline combining all components"""
    
    def __init__(
        self,
        pdf_directory: str = "PDFs",
        index_path: str = "data/faiss_index",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama-3.1-8b-instant",
        groq_api_key: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.pdf_directory = pdf_directory
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        
        # Initialize components
        self.embedding_manager = EmbeddingManager(model_name=embedding_model)
        self.vector_store = VectorStore(index_path=index_path, model_name=embedding_model)
        self.retriever = RAGRetriever(self.vector_store, self.embedding_manager)
        self.text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.llm = None
        
        if self.groq_api_key:
            self.llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name=llm_model,
                temperature=0.1,
                max_tokens=1024
            )
    
    def process_pdfs(self) -> bool:
        """Load and index all PDFs"""
        try:
            processor = PDFProcessor(self.pdf_directory)
            documents = processor.process_all_pdfs()
            
            if not documents:
                return False
            
            chunks = self.text_splitter.split_documents(documents)
            self.vector_store.add_documents(chunks)
            
            return True
            
        except Exception as e:
            print(f"Error processing PDFs: {e}")
            return False
    
    def query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Process a query through the RAG pipeline"""
        
        # Retrieve relevant documents
        results = self.retriever.retrieve(query, top_k=top_k)
        
        if not results:
            return {
                "answer": "No relevant documents found. Please process some PDFs first.",
                "context": [],
                "sources": []
            }
        
        # Build context from retrieved documents
        context = "\n\n".join([doc['content'] for doc in results])
        sources = list(set([doc['metadata'].get('source_file', 'Unknown') for doc in results]))
        
        # Generate answer using LLM if available
        answer = None
        if self.llm:
            prompt = f"""Use the following context to answer the question concisely.

Context:
{context}

Question: {query}

Answer:"""

            response = self.llm.invoke([prompt])
            answer = response.content
        else:
            # Return top results if no LLM
            answer = "LLM not configured. Here are the top relevant excerpts:\n\n"
            answer += "\n\n".join([f"--- Document {i+1} ---\n{doc['content'][:500]}..." 
                                   for i, doc in enumerate(results[:top_k])])
        
        return {
            "answer": answer,
            "context": results,
            "sources": sources
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "vector_store_ready": self.vector_store.is_ready(),
            "document_count": self.vector_store.vectorstore.index.ntotal if self.vector_store.is_ready() else 0,
            "llm_configured": self.llm is not None,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model if self.llm else None
        }
