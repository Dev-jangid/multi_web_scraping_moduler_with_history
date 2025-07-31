import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, get_groq_client, VECTOR_STORE_DIR
from session_manager import init_session_state, create_session, get_sorted_sessions, add_to_history    # Extra added for history "add_to_history"
from web_utils import fetch_website_content, process_content
from vector_utils import vector_store_exists
from groq_utils import generate_chat_response
from vector_utils import retrieve_context

# Initialize session state
init_session_state()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

# Initialize Groq client
try:
    client = get_groq_client()
except Exception as e:
    st.error(str(e))
    st.stop()

# Load embedding model
embedding_model = load_embedding_model()

# Streamlit app layout
st.set_page_config(
    page_title="WebChat Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar - Session Management
with st.sidebar:
    st.title("💬 Chat Sessions")
    
    with st.form("new_session_form", clear_on_submit=True):
        st.subheader("Start New Chat")
        url = st.text_input("Website URL:", value=st.session_state.new_url, 
                           placeholder="https://example.com")
        submit_new = st.form_submit_button("Start Chat")
        
        if submit_new and url:
            if not url.startswith('http'):
                st.error("Please enter a valid URL starting with http:// or https://")
            else:
                with st.spinner("Fetching website content..."):
                    try:
                        raw_text = fetch_website_content(url)
                        if not raw_text:
                            st.error("Failed to fetch content from URL")
                        else:
                            processed_text = process_content(raw_text)
                            
                            # Show cache status
                            if vector_store_exists(url):
                                st.info("Using cached vector database")
                            else:
                                st.info("Creating new vector database")
                            
                            new_session = create_session(url, processed_text, embedding_model)
                            session_id = new_session['id']
                            
                            st.session_state.sessions[session_id] = new_session
                            st.session_state.current_session = session_id
                            st.session_state.new_url = ""
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    st.divider()
    st.subheader("Your Sessions")
    
    if not st.session_state.sessions:
        st.info("No chat sessions yet")
    else:
        for session in get_sorted_sessions():
            display_url = session['url']
            if len(display_url) > 35:
                display_url = display_url[:15] + "..." + display_url[-15:]
                
            if st.button(
                f"{display_url}",
                key=f"session_{session['id']}",
                help=f"Created: {session['created']}"
            ):
                st.session_state.current_session = session['id']
                st.rerun()
        
        if st.session_state.current_session:
            if st.button("Delete Current Session", use_container_width=True):
                if st.session_state.current_session in st.session_state.sessions:
                    del st.session_state.sessions[st.session_state.current_session]
                    st.session_state.current_session = None
                    st.rerun()
    
    # Vector database management
    st.divider()
    st.subheader("Vector Database")
    if st.button("Clear All Vector Databases", use_container_width=True):
        for file in VECTOR_STORE_DIR.glob("*"):
            file.unlink()
        st.success("All vector databases cleared")

# Main Chat Area
st.title("WebChat Assistant")
st.caption("Chat with any website using AI")

if st.session_state.current_session:
    session = st.session_state.sessions[st.session_state.current_session]
    
    st.info(f"**Website:** {session['url']}  \n**Started:** {session['created']}")
    
    ####   Extra for history 
    with st.container():
        st.subheader("Chat History")
        
        if not session['history']:
            st.info("No messages yet. Start a conversation below.")
        
        for exchange in session['history']:
            with st.expander(f"You: {exchange['user']}", expanded=True):
                st.write(f"**Assistant:** {exchange['bot']}")
                st.caption(f"Sent at {exchange['timestamp']}")
    
    # Single Q&A interface
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Your question:", 
                                  placeholder="Ask about this website...",
                                  key="user_input")
        submit_chat = st.form_submit_button("Ask")
        
        if submit_chat and user_input:
            with st.spinner("Thinking..."):
                try:
                    # Retrieve context
                    context = retrieve_context(
                        user_input, 
                        session['vector_store'], 
                        session['chunks'], 
                        embedding_model
                    )
                    
                    # Generate response
                    bot_response = generate_chat_response(
                        client,
                        user_input, 
                        context,
                        session['history']    # Extra added for history
                    )
                    
                    # Display response
                    session = add_to_history(session, user_input, bot_response)     # Extra added for history 
                    st.subheader("Answer")
                    st.write(bot_response)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
else:
    st.info("👈 Start a new chat session by entering a website URL in the sidebar")
    
st.divider()
st.caption("WebChat Assistant v1.2 | Local vector database |chat history stored")








