import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import streamlit as st 

# Load environment variables
load_dotenv()

# Application constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
MAX_TEXT_LENGTH = 28000
MODEL_NAME = "llama3-70b-8192"
TEMPERATURE = 0.4
MAX_TOKENS = 300

# Vector storage configuration
VECTOR_STORE_DIR = Path("vector_db")
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Groq API configuration
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    # api_key = st.secrets["GROQ_API_KEY"]       # For streamlit  cloud 
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    return Groq(api_key=api_key)







