#Streamlit Web Interface

import streamlit as st
import os
import sys
from pathlib import Path


from rag_system import RAGSystem, create_sample_documents


# Page configuration
st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stTitle { color: #1f77b4; }
    .answer-card{
        background:#ffffff;
        color:#222;
        border-radius:18px;
        padding:30px;
        margin-top:15px;
        margin-bottom:20px;

        box-shadow:
        0 10px 25px rgba(0,0,0,.08);

        border:1px solid #ececec;

        font-size:17px;

        line-height:1.9;

        max-height:650px;

        overflow-y:auto;
    }

    .answer-card *{
        color:#222 !important;
    }

    .answer-header{
        display:flex;
        justify-content:space-between;
        align-items:center;

        margin-bottom:20px;
    }

    .answer-title{

        font-size:24px;

        font-weight:700;

        color:#3b82f6;

    }

    .answer-source{

        background:#eff6ff;

        color:#2563eb;

        padding:5px 12px;

        border-radius:30px;

        font-size:13px;

        font-weight:600;

    }
    .source-box{
        background:#f8fafc;
        color:#374151;
        padding:18px;
        border-radius:10px;
        margin-bottom:12px;
        border-left:5px solid #16a34a;
        box-shadow:0 2px 8px rgba(0,0,0,.05);
    }

    .source-box *{
        color:#374151 !important;
    }
    .free-badge {
        background-color: #90EE90;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        color: #2d5016;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'llm_type' not in st.session_state:
    st.session_state.llm_type = 'groq'


def load_rag_system(llm_type: str = 'groq'):
    """Load or create RAG system with chosen LLM"""
    if st.session_state.rag_system is None or st.session_state.llm_type != llm_type:
        rag = RAGSystem(db_path='./vector_db', llm_type=llm_type)
        
        # load existing database
        if not rag.load_from_disk():
            st.info("Creating vector database from sample documents...")
            
            # Create sample documents 
            if not os.path.exists('documents'):
                create_sample_documents()
            
            # Ingest documents
            documents = [
                'documents/HR_Policy.txt',
                'documents/Product_Guide.txt',
                'documents/Compliance_Guide.txt'
            ]
            
            rag.ingest_documents(documents)
            st.success("✓ Vector database created successfully!")
        
        st.session_state.rag_system = rag
        st.session_state.documents_loaded = True
        st.session_state.llm_type = llm_type
    
    return st.session_state.rag_system


# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # st.markdown('<div class="free-badge">✓ 100% FREE - No API Keys!</div>', unsafe_allow_html=True)
    
    # LLM Selection
    st.subheader("Choose LLM")
    llm_choice = st.radio(
        "Select your preferred LLM:",
        options=['groq', 'ollama', 'huggingface'],
        format_func=lambda x: {
            'groq': ' Groq',
            'ollama': ' Ollama',
            'huggingface': ' Hugging Face'
        }[x]
    )
    
    # Show setup instructions based on choice
    if llm_choice == 'groq':
        st.info("""
        **Groq Setup:**
        1. Get key: https://console.groq.com/keys
        2. Set env variable:
           ```
           export GROQ_API_KEY='Enter_your_API_Key'
           ```
        """)
    
    elif llm_choice == 'ollama':
        st.info("""
        **Ollama Setup:**
        1. Download: https://ollama.ai
        2. Install and run:
           ```
           ollama serve
           ```
        3. No costs, runs on YOUR computer!
        """)
    
    else:  # huggingface
        st.info("""
        **Hugging Face Setup:**
        1. Install: `pip install transformers torch`
        2. Runs locally on your computer
        """)
    
    # Initialize with selected LLM
    rag = load_rag_system(llm_choice)
    
    st.divider()
    
    st.subheader("System Status")
    
    if st.session_state.documents_loaded:
        st.success("✓ System Ready")
        st.info(f"Documents loaded: {len(rag.loaded_documents)}")
        if rag.loaded_documents:
            with st.expander("View Loaded Documents"):
                for doc in rag.loaded_documents:
                    st.write(f"📄 {doc}")
    
    st.divider()
    
    st.subheader("Upload New Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process New Documents"):
            os.makedirs('uploaded_docs', exist_ok=True)
            file_paths = []
            
            progress_bar = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                file_path = f'uploaded_docs/{uploaded_file.name}'
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.info(f"Processing {len(file_paths)} documents...")
            rag.ingest_documents(file_paths)
            st.success("✓ Documents processed successfully!")
            st.rerun()
    
    st.divider()
    
    st.subheader("Retrieval Settings")
    k_results = st.slider(
        "Number of chunks to retrieve:",
        min_value=1,
        max_value=10,
        value=5,
        help="More chunks = more context but slower"
    )
    
    st.divider()
    
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Enterprise Knowledge Assistant**
        
        
        **LLM Options:**
        - **Groq**: Online
        - **Ollama**: Local 
        - **Hugging Face**: Local 
        
        **How it works:**
        1. Upload your documents
        2. System chunks and indexes them
        3. Ask questions in natural language
        4. Get answers grounded in YOUR documents
        5. See source citations
        """)


# Main content
st.title(" Enterprise Knowledge Assistant")
st.markdown("Ask questions about your documents!")

# Ensure RAG system is loaded
rag = load_rag_system(llm_choice)

# Question input
st.subheader(" Ask a Question")

col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the leave policy?",
        label_visibility="collapsed"
    )

with col2:
    search_button = st.button("🔍 Search", use_container_width=True)

# Sample questions
st.markdown("**Try these sample questions:**")
sample_questions = [
    "What is the employee leave policy?",
    "How many paid leaves do employees get?",
    "What is the refund policy?",
    "Tell me about data protection requirements",
    "What are the password requirements?",
]

cols = st.columns(2)
for idx, sample_q in enumerate(sample_questions):
    with cols[idx % 2]:
        if st.button(sample_q, use_container_width=True):
            question = sample_q
            search_button = True

# Process question
if question and search_button:
    with st.spinner(f" Using {llm_choice.upper()} to generate answer..."):
        result = rag.answer_question(question, k=k_results)
    answer = result["answer"]

    answer = answer.replace("\n", "<br>")
    answer = answer.replace("•", "&#8226;")
    # Display answer

    import html
    import re

    answer = result["answer"]

    # Remove excessive blank lines
    answer = re.sub(r"\n{3,}", "\n\n", answer)

    # Escape HTML
    answer = html.escape(answer)

    # Preserve line breaks
    answer = answer.replace("\n", "<br>")

    source = result["sources"][0]["document"] if result["sources"] else "Unknown"

    st.markdown(f"""
    <div class="answer-card">

    <div class="answer-header">

    <div class="answer-title">
    🤖 Answer
    </div>

    <div class="answer-source">
    📄 {source}
    </div>

    </div>

    {answer}

    </div>
    """, unsafe_allow_html=True)
    
    # Display sources
    if result['sources']:
        st.markdown("###  Sources")
        for i, source in enumerate(result['sources'], 1):
            similarity = source.get('similarity', 0)
            source_text = f"""
**{i}. {source['document']}** (Chunk {source['chunk']})
- Relevance: {similarity:.1%}
            """
            st.markdown(f'<div class="source-box">{source_text}</div>', unsafe_allow_html=True)
    else:
        st.warning(" No relevant sources found")
    
    # Display statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Retrieved Chunks", result['retrieved_chunks'])
    with col2:
        st.metric("Number of Sources", len(result['sources']))

else:
    # Show welcome message if no question
    st.success("""
     **Enterprise Knowledge Assistant**
    
     Welcome!
    
    **How to use:**
    1. Choose your LLM (left sidebar)
    2. Ask a question about your documents
    3. Get instant answers with sources
    
    **To add your documents:**
    1. Left sidebar → "Upload New Documents"
    2. Select PDF or TXT files
    3. Click "Process New Documents"
    4. Ask questions immediately!
    """)

# Footer
# st.divider()
# st.markdown("""
# <div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;">
#     <strong> Completely FREE!</strong><br>
#     Sentence Transformers + FAISS + Free LLM (Groq/Ollama/HF)<br>
#     No costs, no hidden charges, fully private!
# </div>
# """, unsafe_allow_html=True)
