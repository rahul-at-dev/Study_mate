"""
Utility functions for StudyMate application
"""

import re
import streamlit as st
from typing import List, Dict, Any
import logging
import time
from functools import wraps

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s.,!?;:()\[\]{}"\'-]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def validate_file(file) -> bool:
    """Validate uploaded file"""
    if file is None:
        return False
    
    # Check file type
    if not file.name.lower().endswith('.pdf'):
        st.error(f"File {file.name} is not a PDF file")
        return False
    
    # Check file size (10MB limit)
    if file.size > 10 * 1024 * 1024:
        st.error(f"File {file.name} is too large (max 10MB)")
        return False
    
    return True

def format_source_reference(chunk: str) -> str:
    """Extract and format source reference from chunk"""
    # Look for source reference in chunk
    source_match = re.search(r'\[Source: (.+?)\]', chunk)
    if source_match:
        return source_match.group(1)
    return "Unknown Source"

def highlight_keywords(text: str, keywords: List[str]) -> str:
    """Highlight keywords in text for better visibility"""
    if not keywords:
        return text
    
    for keyword in keywords:
        # Case-insensitive highlighting
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        text = pattern.sub(f'**{keyword}**', text)
    
    return text

def get_text_stats(text: str) -> Dict[str, Any]:
    """Get statistics about text"""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    paragraphs = text.split('\n\n')
    
    return {
        'characters': len(text),
        'words': len(words),
        'sentences': len([s for s in sentences if s.strip()]),
        'paragraphs': len([p for p in paragraphs if p.strip()]),
        'avg_words_per_sentence': len(words) / max(len(sentences), 1)
    }

def create_download_link(content: str, filename: str, link_text: str) -> str:
    """Create a download link for content"""
    import base64
    
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class StreamlitHandler:
    """Helper class for Streamlit UI operations"""
    
    @staticmethod
    def show_processing_status(message: str, progress: float = None):
        """Show processing status with optional progress"""
        if progress is not None:
            st.progress(progress)
        st.info(message)
    
    @staticmethod
    def show_error_with_details(error_message: str, details: str = None):
        """Show error message with expandable details"""
        st.error(error_message)
        if details:
            with st.expander("Error Details"):
                st.code(details)
    
    @staticmethod
    def show_success_with_stats(message: str, stats: Dict[str, Any] = None):
        """Show success message with optional statistics"""
        st.success(message)
        if stats:
            cols = st.columns(len(stats))
            for i, (key, value) in enumerate(stats.items()):
                with cols[i]:
                    st.metric(key.replace('_', ' ').title(), value)
    
    @staticmethod
    def create_info_expander(title: str, content: str):
        """Create an expandable info section"""
        with st.expander(title):
            st.markdown(content)

def chunk_text_advanced(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Advanced text chunking with better sentence boundary detection"""
    # Clean text first
    text = clean_text(text)
    
    # Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        # If adding this sentence would exceed chunk size
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
                current_size = len(current_chunk)
            else:
                current_chunk = sentence
                current_size = sentence_size
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_size += sentence_size
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """Extract key phrases from text using simple frequency analysis"""
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter out stop words
    filtered_words = [word for word in words if word not in stop_words]
    
    # Count frequency
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top phrases
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_phrases]]