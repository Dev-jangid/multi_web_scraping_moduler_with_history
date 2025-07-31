import uuid
from datetime import datetime
import streamlit as st
from vector_utils import create_vector_store, vector_store_exists, load_vector_store

def init_session_state():
    if 'sessions' not in st.session_state:
        st.session_state.sessions = {}
    if 'current_session' not in st.session_state:
        st.session_state.current_session = None
    if 'new_url' not in st.session_state:
        st.session_state.new_url = ""

def create_session(url, text, embedding_model):
    """Create a new session with vector store"""
    session_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create or load vector store
    if vector_store_exists(url):
        vector_store, chunks = load_vector_store(url)
    else:
        vector_store, chunks = create_vector_store(text, embedding_model, url)
    
    return {
        'id': session_id,
        'url': url,
        'text': text,
        'chunks': chunks,
        'vector_store': vector_store,
        'history': [],                   # Extra added for History----------------------------------------------------------------
        'created': timestamp,
        'last_accessed': timestamp
    }
    
    
# Extra added for history     
def add_to_history(session, user_input, bot_response):
    session['history'].append({
        'user': user_input,
        'bot': bot_response,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })
    session['last_accessed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return session


def get_sorted_sessions():
    return sorted(
        st.session_state.sessions.values(),
        key=lambda x: x['last_accessed'],
        reverse=True
    )
    
    
    
    
    



    
    
