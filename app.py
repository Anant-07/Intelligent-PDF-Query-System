"""
Intelligent PDF Query System - Streamlit Frontend
A RAG-powered application for querying PDF documents
"""

import streamlit as st
import os
from pathlib import Path

# Import the RAG pipeline
from rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="Intelligent PDF Query System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
        margin-top: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
    }
    .source-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background-color: #E0E0E0;
        border-radius: 1rem;
        font-size: 0.85rem;
        margin-right: 0.5rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #F5F5F5;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline(
    pdf_directory: str,
    index_path: str,
    embedding_model: str,
    llm_model: str,
    groq_api_key: str,
    chunk_size: int,
    chunk_overlap: int
) -> RAGPipeline:
    """Create and cache the RAG pipeline"""
    return RAGPipeline(
        pdf_directory=pdf_directory,
        index_path=index_path,
        embedding_model=embedding_model,
        llm_model=llm_model,
        groq_api_key=groq_api_key,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


def sidebar_configuration():
    """Render sidebar configuration options"""
    st.sidebar.title("⚙️ Configuration")
    
    st.sidebar.header("PDF Settings")
    pdf_dir = st.sidebar.text_input(
        "PDF Directory",
        value="PDFs",
        help="Directory containing PDF files to process"
    )
    
    st.sidebar.header("Embedding Settings")
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        ["nomic-embed-text", "mxbai-embed-large"],
        index=0,
        help="Ollama embedding model"
    )
    
    chunk_size = st.sidebar.slider(
        "Chunk Size",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100,
        help="Size of text chunks for embedding"
    )
    
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
        help="Overlap between text chunks"
    )
    
    st.sidebar.header("LLM Settings")
    llm_model = st.sidebar.selectbox(
        "LLM Model",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        index=0,
        help="Groq LLM model for generating answers"
    )
    
    groq_api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        help="Your Groq API key (or set GROQ_API_KEY in .env)"
    )
    
    # Load from .env if not provided
    if not groq_api_key:
        from dotenv import load_dotenv
        load_dotenv()
        groq_api_key = os.getenv("GROQ_API_KEY", "")
    
    index_path = st.sidebar.text_input(
        "Vector Store Index Path",
        value="data/faiss_index",
        help="Path to save/load FAISS index"
    )
    
    return {
        "pdf_directory": pdf_dir,
        "index_path": index_path,
        "embedding_model": embedding_model,
        "llm_model": llm_model,
        "groq_api_key": groq_api_key,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">📄 Intelligent PDF Query System</div>', unsafe_allow_html=True)
    
    # Get configuration from sidebar
    config = sidebar_configuration()
    
    # Initialize pipeline
    pipeline = get_pipeline(**config)
    
    # Check status
    status = pipeline.get_status()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["💬 Query", "📚 Document Management", "ℹ️ Status"])
    
    with tab1:
        # Query section
        st.markdown('<div class="sub-header">Ask questions about your PDFs</div>', unsafe_allow_html=True)
        
        # Check if vector store is ready
        if not status["vector_store_ready"]:
            st.markdown("""
            <div class="warning-box">
                ⚠️ No documents indexed yet. Please go to the "Document Management" tab to process your PDFs first.
            </div>
            """, unsafe_allow_html=True)
        
        # Chat input
        query = st.chat_input("Ask a question about your PDF documents...")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Process query
        if query:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.markdown(query)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Searching and generating answer..."):
                    result = pipeline.query(query, top_k=3)
                    
                    # Display answer
                    st.markdown("### Answer")
                    st.markdown(result["answer"])
                    
                    # Display sources
                    if result["sources"]:
                        st.markdown("### Sources")
                        for source in result["sources"]:
                            st.markdown(f'<span class="source-tag">📄 {source}</span>', unsafe_allow_html=True)
                    
                    # Show context (expandable)
                    with st.expander("📋 View Retrieved Context"):
                        for i, doc in enumerate(result["context"], 1):
                            st.markdown(f"**Document {i}** (Score: {doc['similarity_score']:.3f})")
                            st.markdown(f"Source: {doc['metadata'].get('source_file', 'Unknown')}")
                            content_preview = doc['content'][:500]
                            st.code(content_preview, language="text")
            
            # Add assistant response to history
            response_with_sources = f"{result['answer']}\n\n**Sources:** {', '.join(result['sources'])}"
            st.session_state.messages.append({"role": "assistant", "content": response_with_sources})
    
    with tab2:
        # Document management section
        st.markdown('<div class="sub-header">Manage PDF Documents</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Current Status")
            if status["vector_store_ready"]:
                st.markdown(f"""
                <div class="success-box">
                    ✅ Vector store is ready with <strong>{status['document_count']}</strong> documents indexed.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    ⚠️ No documents indexed yet.
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Actions")
            if st.button("🔄 Re-process PDFs", type="primary", use_container_width=True):
                with st.spinner("Processing PDFs..."):
                    success = pipeline.process_pdfs()
                    if success:
                        st.success("✅ PDFs processed successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to process PDFs. Check the directory.")
        
        # Show PDF directory info
        st.markdown("### PDF Directory")
        pdf_path = Path(config["pdf_directory"])
        if pdf_path.exists():
            pdf_files = list(pdf_path.glob("*.pdf"))
            st.markdown(f"**Directory:** `{pdf_path.absolute()}`")
            st.markdown(f"**PDF Files Found:** {len(pdf_files)}")
            
            if pdf_files:
                with st.expander("View PDF files"):
                    for f in pdf_files:
                        st.markdown(f"- 📄 {f.name}")
        else:
            st.error(f"PDF directory not found: {pdf_path}")
    
    with tab3:
        # Status section
        st.markdown('<div class="sub-header">System Status</div>', unsafe_allow_html=True)
        
        # Display status metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Documents Indexed", status["document_count"])
        
        with col2:
            st.metric("Vector Store Ready", "✅ Yes" if status["vector_store_ready"] else "❌ No")
        
        with col3:
            st.metric("LLM Configured", "✅ Yes" if status["llm_configured"] else "❌ No")
        
        with col4:
            st.metric("Embedding Model", status["embedding_model"])
        
        # Display configuration
        st.markdown("### Current Configuration")
        config_info = {
            "PDF Directory": config["pdf_directory"],
            "Index Path": config["index_path"],
            "Embedding Model": config["embedding_model"],
            "LLM Model": config["llm_model"],
            "Chunk Size": config["chunk_size"],
            "Chunk Overlap": config["chunk_overlap"]
        }
        
        for key, value in config_info.items():
            st.markdown(f"**{key}:** `{value}`")


if __name__ == "__main__":
    main()
