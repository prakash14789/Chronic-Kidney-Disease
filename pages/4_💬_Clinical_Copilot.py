# pages/4_💬_Clinical_Copilot.py
import streamlit as st
import time
from ckd_app.dashboard.state import inject_custom_css, check_login, setup_sidebar
from ckd_app.utils.rag_engine import CKDRAGEngine

st.set_page_config(page_title="Clinical Copilot", page_icon="💬", layout="wide")

# Inject premium theme styling
inject_custom_css()
check_login()

# Sidebar Setup
sample_size, use_cv = setup_sidebar("Clinical Copilot")

# Retrieve or initialize the RAG Engine locally
if "rag_engine" not in st.session_state:
    with st.spinner("Initializing Clinical Knowledge Base..."):
        st.session_state.rag_engine = CKDRAGEngine()

rag_engine = st.session_state.rag_engine

# Sidebar Configuration for Hugging Face Token
with st.sidebar:
    st.divider()
    st.markdown("### 🔑 Hugging Face Config")
    hf_token_input = st.text_input(
        "Hugging Face API Token",
        type="password",
        value=st.session_state.get("hf_token", ""),
        help="Access the serverless LLM generation. Get a free token at: https://huggingface.co/settings/tokens"
    )
    if hf_token_input:
        st.session_state["hf_token"] = hf_token_input
        
    st.info("💡 **Local Embeddings active** using all-MiniLM-L6-v2. Vector search runs 100% offline.")

# Main Interface Header
st.title("💬 Clinical Copilot")
st.markdown("Guideline Retrieval-Augmented Generation (RAG) assistant powered by Hugging Face.")

# Show warning banner if token is missing
if not st.session_state.get("hf_token"):
    st.warning(
        "⚠️ **Hugging Face Hub API Token Missing**\n\n"
        "You can still search the guidelines! The system will retrieve matching document chunks directly from the local vector database, "
        "but it won't synthesize a chat response. Enter a token in the sidebar to enable full conversational generation."
    )

# Quick Sample Prompts
st.markdown("### 📋 Suggested Guideline Queries")
cols = st.columns(3)
sample_prompts = [
    "What are the KDIGO blood pressure targets for diabetic CKD patients?",
    "When should Metformin and SGLT2 inhibitors be initiated or discontinued?",
    "What are the dietary sodium and protein targets recommended by KDIGO?"
]

clicked_prompt = None
for idx, prompt_text in enumerate(sample_prompts):
    with cols[idx]:
        if st.button(prompt_text, key=f"sample_prompt_{idx}", use_container_width=True):
            clicked_prompt = prompt_text

# Initialize Chat Log
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your clinical kidney disease specialist copilot. I can answer questions about classification, staging, therapy management, and lifestyle recommendations using the official **KDIGO Clinical Practice Guidelines**.",
            "sources": []
        }
    ]

# Display Chat Logs
for msg in st.session_state.rag_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("🔍 Retrieved Guideline Sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f"- **{src['title']}** (Chunk {src['chunk']}) "
                        f"— *Match confidence:* `{src['score'] * 100:.1f}%`"
                    )

# Get current input (from quick buttons or standard chat input)
user_query = clicked_prompt or st.chat_input("Ask a KDIGO guideline query (e.g. 'What is eGFR category G3a?')...")

if user_query:
    # Render user query immediately
    st.session_state.rag_messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate and render assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching guidelines and generating response..."):
            start_time = time.time()
            
            # Fetch token
            token = st.session_state.get("hf_token")
            
            # Query the RAG Engine
            result = rag_engine.query(user_query, token=token)
            
            elapsed = time.time() - start_time
            answer = result["answer"]
            sources = result["sources"]

            # Display response
            st.markdown(answer)
            
            if sources:
                st.caption(f"⚡ Search completed in {elapsed:.2f}s using {len(sources)} reference chunks.")
                with st.expander("🔍 Retrieved Guideline Sources"):
                    for src in sources:
                        st.markdown(
                            f"- **{src['title']}** (Chunk {src['chunk']}) "
                            f"— *Match confidence:* `{src['score'] * 100:.1f}%`"
                        )
            
            # Save to chat log state
            st.session_state.rag_messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
            
    # Trigger a rerun if prompt clicked from button to clear the button click state
    if clicked_prompt:
        st.rerun()
