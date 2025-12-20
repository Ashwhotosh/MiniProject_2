import streamlit as st
import os
import warnings
import logging

# Silence Terminal
os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore")
logging.getLogger('chromadb').setLevel(logging.ERROR)

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Libraries
from tools_library import (
    fetch_ipo_details, download_pdf_logic, build_vs_logic,
    get_all_ipo_names, get_concurrent_ipos
)
from brain import execute_brain
from report_engine import generate_deep_dive_report
from comparison_engine import execute_peer_comparison

load_dotenv()

# --- CONFIG & STYLING ---
st.set_page_config(page_title="IPO Smooth Operator", page_icon="🚀", layout="wide")

# Custom CSS for "Attractive & Unique" UI
st.markdown("""
<style>
    /* Gradient Main Background */
    .stApp {
        background: linear-gradient(to bottom right, #0e1117, #151922);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #11141d;
        border-right: 1px solid #2b313e;
    }

    /* Glassmorphism Containers */
    div[data-testid="stExpander"], div.stChatInput {
        background-color: rgba(38, 45, 61, 0.4);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #FF4B4B, #FF914D);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 10px 20px;
        color: #e0e0e0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "active_ipo" not in st.session_state: st.session_state.active_ipo = None
if "active_category" not in st.session_state: st.session_state.active_category = "Mainboard"
if "last_report" not in st.session_state: st.session_state.last_report = ""


@st.cache_data
def load_data(): return get_all_ipo_names()


ipo_data = load_data()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2534/2534204.png", width=50)
    st.title("Control Panel")

    st.markdown("### 1️⃣ Select Target")
    category = st.radio("Category:", ["Mainboard", "SME"], horizontal=True)

    available_ipos = ipo_data.get(category, [])
    if not available_ipos: available_ipos = ["No IPOs found"]

    selected_ipo = st.selectbox("Choose IPO:", available_ipos)

    st.markdown("---")

    st.markdown("### 2️⃣ System Actions")
    if st.button("🚀 Initialize System", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.active_ipo = selected_ipo
        st.session_state.active_category = category  # Store for peer logic
        st.session_state.vector_store = None
        st.session_state.last_report = ""

        with st.status("booting_core_systems...", expanded=True):
            st.write(f"Targeting: **{selected_ipo}**")
            details = fetch_ipo_details(selected_ipo)

            if "id" in details:
                st.write("✅ Metadata Acquired")
                st.write("📥 Fetching Official RHP...")
                pdf = download_pdf_logic(details)
                if pdf:
                    st.write("🧠 Training Vector Brain...")
                    st.session_state.vector_store = build_vs_logic(pdf)
                    st.success("System Online & Ready")
                else:
                    st.warning("⚠️ RHP Missing. Using Web Search Fallback.")
            else:
                st.error("❌ Critical Error: IPO ID Not Found.")

# --- MAIN PAGE ---
if not st.session_state.active_ipo:
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h1>🚀 IPO Smooth Operator</h1>
        <p style='color: #888;'>Select an IPO from the sidebar to begin your analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Header Stats
with st.container():
    st.markdown(f"## 🎯 Analysis: {st.session_state.active_ipo}")
    # We could fetch details again for the header, or store them in session state.
    # For now, let's keep it clean.

tab_chat, tab_report, tab_compare = st.tabs(["💬 AI Assistant", "📑 360° Report", "⚔️ Peer Comparison"])

# === TAB 1: CHAT ===
with tab_chat:
    st.markdown("##### 🤖 Ask me anything about this IPO")

    # Message Container
    chat_container = st.container()
    with chat_container:
        for m in st.session_state.messages:
            avatar = "👤" if isinstance(m, HumanMessage) else "🤖"
            with st.chat_message("user" if isinstance(m, HumanMessage) else "assistant", avatar=avatar):
                st.write(m.content)

    # Input Area
    if prompt := st.chat_input("Ex: What are the risk factors? What is the GMP?"):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            response_container = st.empty()
            status_container = st.status("Thinking...", expanded=True)
            final_ans = ""

            for chunk in execute_brain(prompt, st.session_state.active_ipo, st.session_state.vector_store):
                if "Executing:" in chunk or "Complete" in chunk or "Synthesizing" in chunk:
                    status_container.write(chunk)
                else:
                    final_ans = chunk

            status_container.update(label="Response Ready", state="complete", expanded=False)
            response_container.markdown(final_ans)
            st.session_state.messages.append(AIMessage(content=final_ans))

# === TAB 2: REPORT ===
with tab_report:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📊 Comprehensive Due Diligence")
        st.caption("Generates a deep-dive investment memo using Real-time Data + RHP Analysis.")
    with col2:
        if st.button("Generate Report", type="primary", use_container_width=True):
            with st.status("Compiling Report (This takes ~30s)...", expanded=True) as status:
                full_text = ""
                for chunk in generate_deep_dive_report(st.session_state.active_ipo, st.session_state.vector_store):
                    if "**Phase" in chunk:
                        st.write(chunk)
                    else:
                        full_text = chunk
                st.session_state.last_report = full_text
                status.update(label="Report Complete", state="complete", expanded=False)

    if st.session_state.last_report:
        st.markdown("---")
        st.markdown(st.session_state.last_report)

# === TAB 3: PEER COMPARISON ===
with tab_compare:
    st.markdown("### ⚔️ Battle of the IPOs")

    # 1. Filter Logic
    # Default to the active IPO's category, but allow user to switch
    col_filter, col_spacer = st.columns([1, 2])
    with col_filter:
        peer_category = st.radio(
            "Filter Peers By:",
            ["Mainboard", "SME", "All"],
            horizontal=True,
            index=0 if st.session_state.active_category == "Mainboard" else 1
        )

    # 2. Fetch Peers based on filter
    peers_list = get_concurrent_ipos(st.session_state.active_ipo, category_filter=peer_category)

    if not peers_list:
        st.info("No active peers found in this category (others might be listed already).")
    else:
        selected_peers = st.multiselect(
            "Select Opponents:",
            options=peers_list,
            placeholder="Select peers to compare metrics..."
        )

        if st.button("⚔️ Run Comparison", type="primary", disabled=len(selected_peers) == 0):
            with st.status("Gathering Intelligence...", expanded=True) as status:
                result_container = st.empty()
                full_analysis = ""
                # Pass the vector_store so we can dig into the RHP
                for chunk in execute_peer_comparison(st.session_state.active_ipo, selected_peers, st.session_state.vector_store):
                    if "Phase" in chunk or "Fetching" in chunk:
                        st.write(chunk)
                    else:
                        full_analysis = chunk
                status.update(label="Analysis Complete", state="complete", expanded=False)
                result_container.markdown(full_analysis)