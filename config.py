"""
Configuration file for StudyMate application
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # PDF Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Embedding Model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Search Configuration
    DEFAULT_K_CHUNKS: int = 5
    MAX_K_CHUNKS: int = 10
    
    # LLM Configuration
    DEFAULT_TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 500
    
    # IBM Watsonx
    WATSONX_MODEL_ID: str = "mistralai/mixtral-8x7b-instruct-v01"
    WATSONX_URL: str = "https://us-south.ml.cloud.ibm.com"
    
    # Streamlit Configuration
    PAGE_TITLE: str = "StudyMate - AI Academic Assistant"
    PAGE_ICON: str = "📚"
    LAYOUT: str = "wide"
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables"""
        config = cls()
        
        # Override with environment variables if available
        config.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', config.CHUNK_SIZE))
        config.CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', config.CHUNK_OVERLAP))
        config.EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', config.EMBEDDING_MODEL)
        config.DEFAULT_TEMPERATURE = float(os.getenv('DEFAULT_TEMPERATURE', config.DEFAULT_TEMPERATURE))
        config.MAX_TOKENS = int(os.getenv('MAX_TOKENS', config.MAX_TOKENS))
        config.WATSONX_MODEL_ID = os.getenv('WATSONX_MODEL_ID', config.WATSONX_MODEL_ID)
        config.WATSONX_URL = os.getenv('IBM_WATSONX_URL', config.WATSONX_URL)
        
        return config

# Global config instance
config = Config.from_env()