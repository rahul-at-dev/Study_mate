import streamlit as st
import os
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import re
import tempfile
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class HuggingFaceLLM:
    def __init__(self):
        self.api_key = os.getenv('HUGGINGFACE_API_KEY')
        # Using a better free model for text generation
        self.model_id = "microsoft/DialoGPT-medium"
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"

        # Alternative models you can try:
        self.alternative_models = {
            "DialoGPT-medium": "microsoft/DialoGPT-medium",
            "GPT-2": "gpt2",
            "DistilGPT-2": "distilgpt2",
            "FLAN-T5-base": "google/flan-t5-base"
        }

    def is_configured(self) -> bool:
        """Check if API key is configured"""
        return self.api_key is not None and self.api_key != "" and self.api_key != "your_huggingface_token_here"

    def test_connection(self) -> Tuple[bool, str]:
        """Test connection to Hugging Face API"""
        if not self.is_configured():
            return False, "API key not configured"

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                "https://api-inference.huggingface.co/models/gpt2",
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                return True, "Connection successful"
            else:
                return False, f"HTTP {response.status_code}: {response.text}"
        except Exception as e:
            return False, f"Connection error: {str(e)}"

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """Generate answer using Hugging Face"""
        if not self.is_configured():
            return """
❌ **Hugging Face API key not configured**

**To get your FREE Hugging Face API key:**

1. Go to [huggingface.co](https://huggingface.co)
2. Sign up for a free account
3. Go to **Settings** → **Access Tokens**
4. Create a **new token** (select "Read" type)
5. Copy the token
6. Add it to your **.env** file: `HUGGINGFACE_API_KEY=your_token_here`
7. Restart the application

**Your token should look like:** `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
            """

        # Prepare context (keep it shorter for free API limits)
        context = "\n".join(context_chunks[:2])[:1000]  # Limit context length

        # Create a better prompt for academic content
        prompt = f"""Academic Content Summary:
{context}

Student Question: {query}

Helpful Answer:"""

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 150,  # Reasonable length for free tier
                    "temperature": 0.3,
                    "do_sample": True,
                    "top_p": 0.9,
                    "return_full_text": False
                },
                "options": {
                    "wait_for_model": True  # Wait if model is loading
                }
            }

            with st.spinner("🤖 Generating answer with Hugging Face AI..."):
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=60  # Longer timeout for free tier
                )

            if response.status_code == 200:
                result = response.json()

                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '').strip()

                    if generated_text:
                        return f"📚 **Answer based on your documents:**\n\n{generated_text}\n\n---\n*Powered by Hugging Face 🤗*"
                    else:
                        return "I couldn't generate a specific answer. Please try rephrasing your question or check if your documents contain relevant information."

                elif isinstance(result, dict):
                    if "error" in result:
                        if "loading" in result["error"].lower():
                            return "🔄 **Model is loading...** Please wait a moment and try again. Free models sometimes need time to start up."
                        else:
                            return f"❌ **Hugging Face Error:** {result['error']}"
                    else:
                        return "Received unexpected response format. Please try again."

                else:
                    return f"Unexpected response format: {str(result)}"

            elif response.status_code == 503:
                return "🔄 **Model is currently loading.** Please wait 30 seconds and try again. This is normal for free tier models."

            elif response.status_code == 429:
                return "⏳ **Rate limit exceeded.** Please wait a minute before asking another question."

            else:
                error_text = response.text
                return f"❌ **Error {response.status_code}:** {error_text}\n\nTry again in a few moments."

        except requests.exceptions.Timeout:
            return "⏰ **Request timed out.** The free tier can be slow. Please try again."

        except Exception as e:
            return f"❌ **Error generating answer:** {str(e)}\n\nPlease check your internet connection and try again."

    def change_model(self, model_key: str):
        """Change the AI model being used"""
        if model_key in self.alternative_models:
            self.model_id = self.alternative_models[model_key]
            self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
            return f"✅ Switched to {model_key}"
        return "❌ Unknown model"

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
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():  # Only add non-empty pages
                    text += f"[Page {page_num + 1}]\n{page_text}\n\n"
            doc.close()

            # Clean up temporary file
            os.unlink(tmp_path)

            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s.,!?;:()\[\]{}"\'-]', '', text)
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Split text into overlapping chunks (smaller for HuggingFace)"""
        # Clean text
        text = self.clean_text(text)

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
            if chunk and len(chunk) > 50:  # Only keep substantial chunks
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

                st.info(f"✅ {pdf_file.name}: {len(chunks)} chunks created")
            else:
                st.warning(f"⚠️ {pdf_file.name}: No text extracted")

            progress_bar.progress((i + 1) / len(pdf_files))

        status_text.text("✅ Processing complete!")
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
        with st.spinner("🧠 Creating embeddings for semantic search..."):
            embeddings = _self.model.encode(chunks, show_progress_bar=True)
        return embeddings

    def build_faiss_index(self, chunks: List[str]):
        """Build FAISS index from text chunks"""
        if not chunks:
            st.error("No chunks to process")
            return

        self.embeddings = self.create_embeddings(chunks)

        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

        st.success(f"🔍 FAISS index built with {len(chunks)} chunks")

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Search for most relevant chunks"""
        if self.index is None:
            return []

        # Create query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        return [(indices[0][i], scores[0][i]) for i in range(len(indices[0]))]

class StudyMateApp:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.embedding_manager = EmbeddingManager()
        self.llm = HuggingFaceLLM()
        self.setup_page_config()

    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="StudyMate - AI Academic Assistant (Hugging Face 🤗)",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def render_header(self):
        """Render application header"""
        st.title("📚 StudyMate - AI Academic Assistant")
        st.markdown("*Powered by Hugging Face 🤗 - Free AI for Everyone*")
        st.markdown("---")
        st.markdown("""
        **Upload your academic PDFs and ask questions about them!**

        StudyMate uses Hugging Face's free AI models to help you understand your study materials through natural conversation.
        """)

    def render_sidebar(self):
        """Render sidebar with upload and settings"""
        with st.sidebar:
            # Hugging Face Status
            st.header("🤗 Hugging Face Status")

            if self.llm.is_configured():
                is_connected, message = self.llm.test_connection()
                if is_connected:
                    st.success("✅ Connected to Hugging Face")
                else:
                    st.warning(f"⚠️ Connection issue: {message}")

                # Model selection
                st.subheader("🤖 AI Model")
                model_options = list(self.llm.alternative_models.keys())
                selected_model = st.selectbox(
                    "Choose AI Model:",
                    model_options,
                    index=0,
                    help="Different models have different strengths"
                )

                if st.button("Switch Model"):
                    result = self.llm.change_model(selected_model)
                    st.info(result)

            else:
                st.error("❌ Hugging Face API key not configured")
                st.markdown("""
                **Quick Setup:**
                1. Go to [huggingface.co](https://huggingface.co)
                2. Sign up (free)
                3. Get API token from Settings → Access Tokens
                4. Add to .env file: `HUGGINGFACE_API_KEY=your_token`
                5. Restart app
                """)

            st.markdown("---")
            st.header("📁 Document Upload")

            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type="pdf",
                accept_multiple_files=True,
                help="Upload academic PDFs: textbooks, papers, notes"
            )

            if uploaded_files:
                st.success(f"📄 {len(uploaded_files)} file(s) uploaded")

                # Show file details
                total_size = sum(file.size for file in uploaded_files)
                st.info(f"Total size: {total_size / (1024*1024):.1f} MB")

                if st.button("🚀 Process Documents", type="primary"):
                    with st.spinner("Processing documents..."):
                        chunks = self.pdf_processor.process_pdfs(uploaded_files)

                        if chunks:
                            self.embedding_manager.build_faiss_index(chunks)
                            st.session_state.chunks = chunks
                            st.session_state.processed = True
                            st.balloons()  # Celebration!
                            st.success("🎉 Documents processed successfully!")
                        else:
                            st.error("❌ No text found in uploaded documents")

            st.markdown("---")
            st.header("⚙️ Settings")

            # Search settings
            st.session_state.k_chunks = st.slider(
                "Chunks to retrieve",
                min_value=2,
                max_value=6,
                value=3,
                help="More chunks = more context but slower responses"
            )

            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                st.info("Free tier limitations:")
                st.write("• Models may take time to load")
                st.write("• Rate limits apply")
                st.write("• Shorter responses to save tokens")

            st.markdown("---")
            st.markdown("### 📖 Usage Tips")
            st.markdown("""
            **Best Questions:**
            - "What are the main points of chapter X?"
            - "Explain the concept of..."
            - "What is the difference between X and Y?"
            - "Summarize the methodology"

            **For Better Results:**
            - Upload related documents together
            - Ask specific questions
            - Be patient (free tier can be slow)
            """)

    def render_main_interface(self):
        """Render main chat interface"""
        if not self.llm.is_configured():
            st.warning("⚠️ Please configure your Hugging Face API key first (see sidebar)")
            st.info("👈 Check the sidebar for setup instructions")
            return

        if 'processed' not in st.session_state:
            st.info("👆 Please upload and process your PDF documents first using the sidebar.")

            # Show example of what StudyMate can do
            with st.expander("🎯 See what StudyMate can do"):
                st.markdown("""
                **Example Questions You Can Ask:**

                📖 *"What are the key concepts in chapter 3?"*

                🔍 *"Explain the research methodology used in this paper"*

                📊 *"What are the main findings and conclusions?"*

                🤔 *"How does theory X relate to concept Y?"*

                📝 *"Summarize the introduction section"*
                """)
            return

        st.header("💬 Ask Questions About Your Documents")

        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.container():
                # Question
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>🙋‍♂️ You:</strong> {question}
                </div>
                """, unsafe_allow_html=True)

                # Answer
                st.markdown(f"""
                <div style="background-color: #e8f4f8; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>🤖 StudyMate:</strong><br>{answer}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")

        # Question input
        with st.form("question_form", clear_on_submit=True):
            query = st.text_input(
                "Ask a question about your documents:",
                placeholder="e.g., What are the main concepts explained in chapter 3?",
                key="question_input"
            )

            col1, col2, col3 = st.columns([2, 2, 2])

            with col1:
                ask_button = st.form_submit_button("🚀 Ask", type="primary")

            with col2:
                if st.form_submit_button("🔄 Clear History"):
                    st.session_state.chat_history = []
                    st.rerun()

            with col3:
                if st.form_submit_button("💡 Example"):
                    example_questions = [
                        "What are the main points discussed?",
                        "Explain the key concepts",
                        "What is the conclusion?",
                        "Summarize the methodology",
                        "What are the important findings?"
                    ]
                    st.session_state.question_input = np.random.choice(example_questions)
                    st.rerun()

        if ask_button and query:
            self.process_question(query)

    def process_question(self, query: str):
        """Process user question and generate answer"""
        if 'chunks' not in st.session_state:
            st.error("Please process documents first")
            return

        start_time = time.time()

        with st.spinner("🔍 Searching for relevant information..."):
            # Search for relevant chunks
            search_results = self.embedding_manager.search(
                query,
                k=st.session_state.get('k_chunks', 3)
            )

            if not search_results:
                st.error("No relevant information found")
                return

            # Get relevant chunks
            relevant_chunks = []
            for idx, score in search_results:
                if idx < len(st.session_state.chunks):
                    relevant_chunks.append(st.session_state.chunks[idx])

            # Show search info
            search_time = time.time() - start_time
            st.info(f"🔍 Found {len(relevant_chunks)} relevant sections in {search_time:.1f}s")

        # Generate answer
        answer = self.llm.generate_answer(query, relevant_chunks)

        # Add timing info
        total_time = time.time() - start_time
        answer += f"\n\n⏱️ *Response generated in {total_time:.1f} seconds*"

        # Add to chat history
        st.session_state.chat_history.append((query, answer))

        # Rerun to show new message
        st.rerun()

    def render_document_stats(self):
        """Render document statistics"""
        if 'chunks' in st.session_state:
            st.header("📊 Document Statistics")

            # Basic stats
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📄 Total Chunks", len(st.session_state.chunks))

            with col2:
                total_words = sum(len(chunk.split()) for chunk in st.session_state.chunks)
                st.metric("📝 Total Words", f"{total_words:,}")

            with col3:
                avg_chunk_size = total_words / len(st.session_state.chunks) if st.session_state.chunks else 0
                st.metric("📏 Avg Chunk Size", f"{avg_chunk_size:.0f} words")

            with col4:
                # Estimate reading time
                reading_time = total_words / 200  # Average reading speed
                st.metric("⏱️ Reading Time", f"{reading_time:.0f} min")

            # Document breakdown
            st.subheader("📋 Document Breakdown")

            # Count chunks per document
            doc_stats = {}
            for chunk in st.session_state.chunks:
                # Extract source from chunk
                source_match = re.search(r'\[Source: (.+?)\]', chunk)
                if source_match:
                    source = source_match.group(1)
                    doc_stats[source] = doc_stats.get(source, 0) + 1

            # Display as chart
            if doc_stats:
                import pandas as pd
                df = pd.DataFrame(list(doc_stats.items()), columns=['Document', 'Chunks'])
                st.bar_chart(df.set_index('Document'))

                # Show table
                st.dataframe(df, use_container_width=True)
        else:
            st.info("📤 Upload and process documents to see statistics")

    def run(self):
        """Main application runner"""
        self.render_header()
        self.render_sidebar()

        # Main content area
        tab1, tab2 = st.tabs(["💬 Chat with Your Documents", "📊 Document Statistics"])

        with tab1:
            self.render_main_interface()

        with tab2:
            self.render_document_stats()

# Initialize and run the app
if __name__ == "__main__":
    app = StudyMateApp()
    app.run()
