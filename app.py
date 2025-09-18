import streamlit as st
import os
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import re
import tempfile
from ibm_watson_machine_learning import APIClient
from dotenv import load_dotenv
import json
import time

# Load environment variables
load_dotenv()

class PDFProcessor:
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.index = None
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_path = tmp_file.name
            
            # Extract text
            doc = fitz.open(tmp_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def process_pdfs(self, pdf_files) -> List[str]:
        """Process multiple PDF files and return chunks"""
        all_chunks = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, pdf_file in enumerate(pdf_files):
            status_text.text(f"Processing {pdf_file.name}...")
            
            text = self.extract_text_from_pdf(pdf_file)
            if text:
                chunks = self.chunk_text(text)
                
                # Add metadata to chunks
                for chunk in chunks:
                    chunk_with_metadata = f"[Source: {pdf_file.name}]\n{chunk}"
                    all_chunks.append(chunk_with_metadata)
            
            progress_bar.progress((i + 1) / len(pdf_files))
        
        status_text.text("Processing complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        self.chunks = all_chunks
        return all_chunks

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.embeddings = None
    
    @st.cache_data
    def create_embeddings(_self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        with st.spinner("Creating embeddings..."):
            embeddings = _self.model.encode(chunks, show_progress_bar=True)
        return embeddings
    
    def build_faiss_index(self, chunks: List[str]):
        """Build FAISS index from text chunks"""
        self.embeddings = self.create_embeddings(chunks)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        st.success(f"FAISS index built with {len(chunks)} chunks")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for most relevant chunks"""
        if self.index is None:
            return []
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        return [(indices[0][i], scores[0][i]) for i in range(len(indices[0]))]

class WatsonxLLM:
    def __init__(self):
        self.client = None
        self.model_id = "mistralai/mixtral-8x7b-instruct-v01"
        self.project_id = None
        self.setup_client()
    
    def setup_client(self):
        """Setup IBM Watsonx client"""
        try:
            # Get credentials from environment variables
            api_key = os.getenv('IBM_WATSONX_API_KEY')
            url = os.getenv('IBM_WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')
            self.project_id = os.getenv('IBM_WATSONX_PROJECT_ID')
            
            if not api_key or not self.project_id:
                st.error("Please set IBM_WATSONX_API_KEY and IBM_WATSONX_PROJECT_ID in your .env file")
                return
            
            wml_credentials = {
                "url": url,
                "apikey": api_key
            }
            
            self.client = APIClient(wml_credentials)
            self.client.set.default_project(self.project_id)
            
        except Exception as e:
            st.error(f"Error setting up Watsonx client: {str(e)}")
    
    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """Generate answer using Watsonx"""
        if not self.client:
            return "Error: Watsonx client not initialized"
        
        # Prepare context
        context = "\n\n".join(context_chunks[:3])  # Use top 3 chunks
        
        # Create prompt
        prompt = f"""Based on the following academic content, please provide a detailed and accurate answer to the question. Use only the information provided in the context.

Context:
{context}

Question: {query}

Answer: Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, please state that clearly."""

        try:
            # Generate parameters
            parameters = {
                "max_new_tokens": 500,
                "temperature": 0.3,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
            
            # Generate response
            response = self.client.foundation_models.generate_text(
                model_id=self.model_id,
                prompt=prompt,
                parameters=parameters
            )
            
            return response
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"

class StudyMateApp:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.embedding_manager = EmbeddingManager()
        self.llm = WatsonxLLM()
        self.setup_page_config()
    
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="StudyMate - AI Academic Assistant",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render_header(self):
        """Render application header"""
        st.title("📚 StudyMate - AI Academic Assistant")
        st.markdown("---")
        st.markdown("""
        **Upload your academic PDFs and ask questions about them!**
        
        StudyMate uses advanced AI to help you understand your study materials through natural conversation.
        """)
    
    def render_sidebar(self):
        """Render sidebar with upload and settings"""
        with st.sidebar:
            st.header("📁 Document Upload")
            
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type="pdf",
                accept_multiple_files=True,
                help="Upload one or more PDF files to analyze"
            )
            
            if uploaded_files:
                st.success(f"{len(uploaded_files)} file(s) uploaded")
                
                if st.button("Process Documents", type="primary"):
                    with st.spinner("Processing documents..."):
                        chunks = self.pdf_processor.process_pdfs(uploaded_files)
                        
                        if chunks:
                            self.embedding_manager.build_faiss_index(chunks)
                            st.session_state.chunks = chunks
                            st.session_state.processed = True
                            st.success("Documents processed successfully!")
                        else:
                            st.error("No text found in uploaded documents")
            
            st.markdown("---")
            st.header("⚙️ Settings")
            
            # Search settings
            st.session_state.k_chunks = st.slider("Number of chunks to retrieve", 3, 10, 5)
            st.session_state.temperature = st.slider("Response creativity", 0.1, 1.0, 0.3)
            
            st.markdown("---")
            st.markdown("### 🔧 Setup Instructions")
            st.markdown("""
            1. Create a `.env` file with:
               - `IBM_WATSONX_API_KEY`
               - `IBM_WATSONX_PROJECT_ID`
               - `IBM_WATSONX_URL`
            2. Upload your PDF documents
            3. Process the documents
            4. Start asking questions!
            """)
    
    def render_main_interface(self):
        """Render main chat interface"""
        if 'processed' not in st.session_state:
            st.info("👆 Please upload and process your PDF documents first using the sidebar.")
            return
        
        st.header("💬 Ask Questions About Your Documents")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**🙋‍♂️ You:** {question}")
                st.markdown(f"**🤖 StudyMate:** {answer}")
                st.markdown("---")
        
        # Question input
        query = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main concepts explained in chapter 3?",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 5])
        
        with col1:
            ask_button = st.button("Ask", type="primary")
        
        with col2:
            clear_button = st.button("Clear History")
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        if ask_button and query:
            self.process_question(query)
    
    def process_question(self, query: str):
        """Process user question and generate answer"""
        if 'chunks' not in st.session_state:
            st.error("Please process documents first")
            return
        
        with st.spinner("Searching for relevant information..."):
            # Search for relevant chunks
            search_results = self.embedding_manager.search(
                query, 
                k=st.session_state.get('k_chunks', 5)
            )
            
            if not search_results:
                st.error("No relevant information found")
                return
            
            # Get relevant chunks
            relevant_chunks = []
            for idx, score in search_results:
                if idx < len(st.session_state.chunks):
                    relevant_chunks.append(st.session_state.chunks[idx])
        
        with st.spinner("Generating answer..."):
            # Generate answer
            answer = self.llm.generate_answer(query, relevant_chunks)
            
            # Add to chat history
            st.session_state.chat_history.append((query, answer))
            
            # Clear input and rerun
            st.rerun()
    
    def render_document_stats(self):
        """Render document statistics"""
        if 'chunks' in st.session_state:
            st.header("📊 Document Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Chunks", len(st.session_state.chunks))
            
            with col2:
                total_words = sum(len(chunk.split()) for chunk in st.session_state.chunks)
                st.metric("Total Words", f"{total_words:,}")
            
            with col3:
                avg_chunk_size = total_words / len(st.session_state.chunks) if st.session_state.chunks else 0
                st.metric("Avg Chunk Size", f"{avg_chunk_size:.0f} words")
    
    def run(self):
        """Main application runner"""
        self.render_header()
        self.render_sidebar()
        
        # Main content area
        tab1, tab2 = st.tabs(["💬 Chat", "📊 Statistics"])
        
        with tab1:
            self.render_main_interface()
        
        with tab2:
            self.render_document_stats()

# Initialize and run the app
if __name__ == "__main__":
    app = StudyMateApp()
    app.run()