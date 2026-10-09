import streamlit as st
import os
import json
import random
import nltk

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="AI FAQ Chatbot | Multilingual Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern chat interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .badge-confidence {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #e3f2fd;
        color: #1976d2;
        margin-top: 4px;
    }
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
    }
    .suggestion-btn {
        margin: 2px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. AUTOMATIC NLTK DATA DOWNLOADER (Cloud-safe)
# ==============================================================================
@st.cache_resource(show_spinner="Initializing NLP Engine & Corpora...")
def ensure_nltk_corpora():
    packages = ['punkt', 'punkt_tab', 'wordnet', 'stopwords', 'omw-1.4']
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return True

ensure_nltk_corpora()

from faq_engine import FAQEngine
from llm_engine import OllamaLLM

# ==============================================================================
# 3. SESSION STATE INITIALIZATION
# ==============================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_DB = os.path.join(DATA_DIR, "general_faq.json")

if "faq_engine" not in st.session_state:
    st.session_state.faq_engine = FAQEngine(DEFAULT_DB)

if "llm_engine" not in st.session_state:
    st.session_state.llm_engine = OllamaLLM()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Salam / Hello!** I am your **AI FAQ Assistant**.\n\nAsk me anything in **English**, **Hinglish (Roman Urdu)**, or **Urdu script**! You can inquire about technology, AI, password security, customer support, or orders.",
            "score": None,
            "suggestions": []
        }
    ]

if "similarity_threshold" not in st.session_state:
    st.session_state.similarity_threshold = 0.30

if "selected_language" not in st.session_state:
    st.session_state.selected_language = "Auto Detect"

if "llm_mode" not in st.session_state:
    st.session_state.llm_mode = False

# ==============================================================================
# 4. SIDEBAR CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/bot.png", width=90)
    st.title("FAQ Assistant")
    st.caption("AI-Powered Multilingual Support")
    
    st.divider()
    
    # Language Selection
    st.subheader("🌐 Language Control")
    lang_choice = st.selectbox(
        "Response Language:",
        ["Auto Detect", "English", "Hinglish (Roman Urdu)", "Urdu (اردو)"],
        index=["Auto Detect", "English", "Hinglish (Roman Urdu)", "Urdu (اردو)"].index(st.session_state.selected_language),
        help="Select your preferred response language or let the AI auto-detect."
    )
    st.session_state.selected_language = lang_choice
    
    st.divider()
    
    # Dataset Selector
    st.subheader("📚 Knowledge Base")
    dataset_options = {
        "Overall / General Assistant": "general_faq.json",
        "E-Commerce Customer Support": "ecommerce_faq.json",
        "University Admissions": "university_faq.json"
    }
    
    selected_dataset_name = st.selectbox(
        "Active Domain Dataset:",
        list(dataset_options.keys()),
        index=0
    )
    
    selected_file_path = os.path.join(DATA_DIR, dataset_options[selected_dataset_name])
    
    if st.session_state.get("current_db_file") != selected_file_path:
        st.session_state.faq_engine = FAQEngine(selected_file_path)
        st.session_state.current_db_file = selected_file_path
        st.success(f"Loaded {len(st.session_state.faq_engine.faqs)} FAQs!")
    
    # Similarity Threshold Slider
    st.session_state.similarity_threshold = st.slider(
        "Matching Threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05,
        help="Controls how strict the NLP engine is when matching questions."
    )
    
    st.divider()
    
    # LLM Mode Toggle
    st.subheader("⚡ AI Engine Mode")
    llm_toggle = st.toggle("Enable Generative LLM Mode", value=st.session_state.llm_mode)
    st.session_state.llm_mode = llm_toggle
    
    if llm_toggle:
        st.info("💡 Connecting to self-hosted Ollama server")
        llm_url = st.text_input("Ollama / ngrok Server URL:", value=st.session_state.llm_engine.base_url or "https://your-tunnel.ngrok-free.dev")
        llm_model = st.text_input("Model Name:", value=st.session_state.llm_engine.model_name or "qwen2.5-coder:32b")
        
        if st.button("Save & Test LLM Connection", use_container_width=True):
            st.session_state.llm_engine.set_config(llm_url, llm_model)
            with st.spinner("Pinging Ollama server..."):
                ok, msg = st.session_state.llm_engine.check_connection()
                if ok:
                    st.success("✅ Ollama Connected!")
                else:
                    st.warning(f"⚠️ {msg}\n(Falling back to local FAQ engine)")
    else:
        st.caption("🟢 **Running in Offline TF-IDF Mode** (Fast & 100% reliable)")

    st.divider()
    
    # Author & Repo Info
    st.markdown("### 👩‍💻 Developer")
    st.markdown("**Iqra Mushtaq**")
    st.markdown("[🔗 GitHub Repository](https://github.com/Iqra-Mushtaq2052/FAQ-Chatbot)")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat history cleared. How can I help you today?",
                "score": None,
                "suggestions": []
            }
        ]
        st.rerun()

# ==============================================================================
# 5. MAIN CHAT AREA
# ==============================================================================
st.markdown('<div class="main-header">🤖 AI-Powered Multilingual FAQ Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Instant answers powered by TF-IDF Semantic Matching & Generative LLMs</div>', unsafe_allow_html=True)

# Tabs: Chatbot & FAQ Database Explorer
tab_chat, tab_database = st.tabs(["💬 Chat Assistant", "📋 FAQ Database Explorer"])

with tab_chat:
    # Quick Questions Chips (3 sample buttons)
    st.markdown("**💡 Quick Questions:**")
    col1, col2, col3 = st.columns(3)
    
    sample_queries = [
        "Salam! Kya hal hai? Aap kya kar sakte ho?",
        "What is Artificial Intelligence (AI)?",
        "How can I protect my account passwords?"
    ]
    
    quick_input = None
    if col1.button(f"❓ {sample_queries[0]}", use_container_width=True):
        quick_input = sample_queries[0]
    if col2.button(f"❓ {sample_queries[1]}", use_container_width=True):
        quick_input = sample_queries[1]
    if col3.button(f"❓ {sample_queries[2]}", use_container_width=True):
        quick_input = sample_queries[2]

    st.write("")

    # Display Conversation History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("score") is not None and msg["score"] > 0:
                st.markdown(f'<span class="badge-confidence">🎯 Match Confidence: {msg["score"]*100:.1f}%</span>', unsafe_allow_html=True)
            
            # If there are suggestions
            if msg.get("suggestions"):
                st.write("**Related Questions:**")
                for s_idx, s_text in msg["suggestions"]:
                    if st.button(f"👉 {s_text}", key=f"sugg_{s_idx}_{random.randint(1000,9999)}"):
                        quick_input = s_text

    # Handle User Input (via chat_input or quick button)
    user_input = st.chat_input("Ask a question in English, Hinglish, or Urdu...") or quick_input

    if user_input:
        # Check for explicit language switches in user input
        lower_q = user_input.lower()
        if any(p in lower_q for p in ["answer in english", "answer me in english", "in english please", "speak in english"]):
            st.session_state.selected_language = "English"
        elif any(p in lower_q for p in ["in hinglish", "hinglish me", "hinglish ma", "speak in hinglish", "roman urdu"]):
            st.session_state.selected_language = "Hinglish (Roman Urdu)"
        elif any(p in lower_q for p in ["in urdu", "urdu me", "speak in urdu", "اردو میں"]):
            st.session_state.selected_language = "Urdu (اردو)"

        # Append user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "score": None,
            "suggestions": []
        })

        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        # Generate Assistant Response
        with st.chat_message("assistant", avatar="🤖"):
            if st.session_state.llm_mode and st.session_state.llm_engine.base_url:
                # LLM Mode with live stream placeholder
                response_placeholder = st.empty()
                response_buffer = [""]
                
                def on_token(token):
                    response_buffer[0] += token
                    response_placeholder.markdown(response_buffer[0] + "▌")

                full_ans, success, err = st.session_state.llm_engine.generate_response(
                    user_input,
                    st.session_state.faq_engine.faqs,
                    on_token_callback=on_token,
                    target_language=st.session_state.selected_language
                )
                
                if success:
                    response_placeholder.markdown(full_ans)
                    st.caption(f"⚡ Powered by Ollama LLM • Language: {st.session_state.selected_language}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_ans,
                        "score": None,
                        "suggestions": []
                    })
                else:
                    # Fallback to FAQ engine
                    st.warning(f"⚠️ LLM Offline ({err}). Answering from FAQ engine:")
                    ans, score, suggestions = st.session_state.faq_engine.get_response(
                        user_input, st.session_state.similarity_threshold
                    )
                    st.markdown(ans)
                    if score > 0:
                        st.markdown(f'<span class="badge-confidence">🎯 Match Confidence: {score*100:.1f}%</span>', unsafe_allow_html=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ans,
                        "score": score,
                        "suggestions": suggestions
                    })
            else:
                # FAQ Engine Mode
                ans, score, suggestions = st.session_state.faq_engine.get_response(
                    user_input, st.session_state.similarity_threshold
                )
                st.markdown(ans)
                if score > 0:
                    st.markdown(f'<span class="badge-confidence">🎯 Match Confidence: {score*100:.1f}%</span>', unsafe_allow_html=True)
                
                if suggestions:
                    st.write("**Did you mean:**")
                    for s_idx, s_text in suggestions:
                        st.markdown(f"- ❓ {s_text}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ans,
                    "score": score,
                    "suggestions": suggestions
                })

# ==============================================================================
# 6. FAQ DATABASE EXPLORER TAB
# ==============================================================================
with tab_database:
    st.subheader(f"📋 Loaded FAQs ({len(st.session_state.faq_engine.faqs)} Records)")
    st.caption("Search through all pre-indexed questions and verified answers.")
    
    search_filter = st.text_input("🔍 Search Database:", placeholder="Type a keyword to filter questions or answers...")
    
    filtered_faqs = st.session_state.faq_engine.faqs
    if search_filter:
        filtered_faqs = [
            f for f in filtered_faqs 
            if search_filter.lower() in f['question'].lower() or search_filter.lower() in f['answer'].lower()
        ]
        st.write(f"Showing {len(filtered_faqs)} matching records:")

    for i, faq in enumerate(filtered_faqs, 1):
        with st.expander(f"**{i}. {faq['question']}**"):
            st.markdown(f"**Answer:**\n{faq['answer']}")
