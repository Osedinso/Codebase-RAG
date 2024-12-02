import streamlit as st
import os
from github import Github
from git import Repo
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="CodebaseGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .main-header {
            font-family: 'SF Pro Display', sans-serif;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 2rem;
        }
        .subheader {
            color: #64748B;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border: 1px solid #E2E8F0;
        }
        .user-message {
            background-color: #F8FAFC;
        }
        .bot-message {
            background-color: #F1F5F9;
            border-left: 4px solid #075eec;
        }
        .metadata {
            font-size: 0.8rem;
            color: #64748B;
        }
        .stButton>button {
            width: 100%;
            border-radius: 0.5rem;
            height: 3rem;
            background-color: #075eec;
            color: white;
        }
        .status-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #F1F5F9;
            margin-bottom: 1rem;
        }
        .error-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #FEE2E2;
            color: #991B1B;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize clients
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
pinecone_index = pc.Index("codebase-rag")
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=st.secrets["GROQ_API_KEY"])

# Existing functions remain the same
# [Keep all your existing functions here]

# Main UI
st.markdown("<h1 class='main-header'>🤖 CodebaseGPT</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Your AI-powered code understanding assistant. Enter a GitHub repository URL to begin exploring.</p>", unsafe_allow_html=True)

# Two-column layout for repository input
col1, col2 = st.columns([3, 1])
with col1:
    repo_url = st.text_input("", placeholder="Enter GitHub repository URL (e.g., https://github.com/username/repo)")
with col2:
    analyze_button = st.button("Analyze Repository")

if repo_url:
    if 'repo_processed' not in st.session_state:
        st.session_state.repo_processed = False
        st.session_state.chat_history = []

    if not st.session_state.repo_processed and analyze_button:
        try:
            with st.status("🔍 Processing repository...", expanded=True) as status:
                status.write("Cloning repository...")
                repo_path = clone_repository(repo_url)
                
                status.write("Extracting code files...")
                file_content = get_main_files_content(repo_path)
                
                status.write("Processing and storing embeddings...")
                documents = []
                for file in file_content:
                    code_chunks = get_code_chunks(file['content'])
                    for i, chunk in enumerate(code_chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={"source": file['name'], "chunk_id": i, "text": chunk}
                        )
                        documents.append(doc)
                
                vectorstore = PineconeVectorStore.from_documents(
                    documents=documents,
                    embedding=HuggingFaceEmbeddings(),
                    index_name="codebase-rag",
                    namespace=repo_url
                )
                
                st.session_state.repo_processed = True
                status.update(label="✅ Repository processed successfully!", state="complete")

        except Exception as e:
            st.error(f"Error processing repository: {str(e)}")
            st.session_state.repo_processed = False

    # Chat interface
    if st.session_state.repo_processed:
        st.markdown("### 💬 Chat with your codebase")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class='chat-message user-message'>
                        <b>You:</b><br>{message["content"]}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='chat-message bot-message'>
                        <b>CodebaseGPT:</b><br>{message["content"]}
                    </div>
                """, unsafe_allow_html=True)

        # Question input
        user_question = st.text_input("Ask a question about the codebase:", key="question_input")
        if user_question:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            with st.spinner("🤔 Analyzing your question..."):
                response = perform_rag(user_question, repo_url)
                # Add bot response to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
            # Rerun to update chat display
            st.experimental_rerun()

        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()

else:
    st.info("👋 Welcome! Please enter a GitHub repository URL to begin exploring your codebase.")
