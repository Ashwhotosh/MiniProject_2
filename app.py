import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Import Libraries
from tools_library import (
    fetch_ipo_details, download_pdf_logic, build_vs_logic,
    get_all_ipo_names, get_concurrent_ipos
)
from brain import execute_brain
from report_engine import generate_deep_dive_report
from comparison_engine import execute_peer_comparison  # <--- NEW IMPORT

load_dotenv()
st.set_page_config(page_title="IPO Smooth Operator", page_icon="😎", layout="wide")

# Init State
if "messages" not in st.session_state: st.session_state.messages = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "active_ipo" not in st.session_state: st.session_state.active_ipo = None
if "last_report" not in st.session_state: st.session_state.last_report = ""
if "concurrent_peers" not in st.session_state: st.session_state.concurrent_peers = []


# --- CACHED DATA LOADER ---
@st.cache_data
def load_data():
    return get_all_ipo_names()


ipo_data = load_data()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Control Panel")
    category = st.radio("Select Category:", ["Mainboard", "SME"], horizontal=True)
    available_ipos = ipo_data.get(category, [])
    if not available_ipos: available_ipos = ["No IPOs found"]

    selected_ipo = st.selectbox("Choose Target IPO:", available_ipos)

    st.divider()

    if st.button("🚀 Initialize System", type="primary"):
        st.session_state.messages = []
        st.session_state.active_ipo = selected_ipo
        st.session_state.vector_store = None
        st.session_state.last_report = ""
        # Fetch potential peers immediately upon init
        st.session_state.concurrent_peers = get_concurrent_ipos(selected_ipo)

        with st.status("Initializing System...", expanded=True):
            st.write(f"Target: {selected_ipo}")
            st.write("Fetching ID & Details...")
            details = fetch_ipo_details(selected_ipo)

            if "id" in details:
                st.write(f"ID Found: {details['id']}")
                st.write("Downloading RHP...")
                pdf = download_pdf_logic(details['id'])
                if pdf:
                    st.write("Building Vector Brain...")
                    st.session_state.vector_store = build_vs_logic(pdf)
                    st.success("System Online.")
                else:
                    st.warning("RHP Download Failed (Web Search Fallback Active).")
            else:
                st.error("IPO Not Found in Database.")

# --- MAIN UI ---
st.title("😎 IPO Smooth Operator")

if not st.session_state.active_ipo:
    st.info("👈 Please Select an IPO and Click Initialize.")
    st.stop()

st.caption(f"Active Session: **{st.session_state.active_ipo}**")

# --- TABS ---
tab_chat, tab_report, tab_compare = st.tabs(["💬 Chat", "📑 360° Report", "⚔️ Peer Comparison"])

# === TAB 1: CHAT ===
with tab_chat:
    for m in st.session_state.messages:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        with st.chat_message(role): st.write(m.content)

    if prompt := st.chat_input("Ask about Sentiment, Peers, GMP..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            response_container = st.empty()
            status_container = st.status("Brain Working...", expanded=True)
            final_ans = ""
            for chunk in execute_brain(prompt, st.session_state.active_ipo, st.session_state.vector_store):
                if "Executing:" in chunk or "Complete" in chunk or "Synthesizing" in chunk:
                    status_container.write(chunk)
                else:
                    final_ans = chunk
            status_container.update(label="Complete", state="complete", expanded=False)
            response_container.markdown(final_ans)
            st.session_state.messages.append(AIMessage(content=final_ans))

# === TAB 2: REPORTS ===
with tab_report:
    st.markdown("### 📊 Automated Due Diligence")
    if st.button("Generate Report", type="primary"):
        with st.status("Generating Report...", expanded=True) as status:
            for chunk in generate_deep_dive_report(st.session_state.active_ipo, st.session_state.vector_store):
                if "**Phase" in chunk:
                    st.write(chunk)
                else:
                    st.session_state.last_report = chunk
            status.update(label="Done", state="complete", expanded=False)

    if st.session_state.last_report:
        st.divider()
        st.markdown(st.session_state.last_report)

# === TAB 3: COMPARISON ===
with tab_compare:
    st.markdown("### ⚔️ Battle of the IPOs")
    st.write("Compare your target IPO with other currently active IPOs.")

    if not st.session_state.concurrent_peers:
        st.warning("No concurrent IPOs found (Everything else might be listed).")
    else:
        # Multiselect for Peers
        peers_to_compare = st.multiselect(
            "Select Peers to Compare:",
            options=st.session_state.concurrent_peers,
            placeholder="Choose one or more active IPOs"
        )

        if st.button("Run Comparison Analysis", type="primary", disabled=len(peers_to_compare) == 0):
            result_container = st.empty()
            full_analysis = ""

            with st.status("Running Comparison...", expanded=True) as status:
                # Call the Comparison Engine
                for chunk in execute_peer_comparison(st.session_state.active_ipo, peers_to_compare):
                    if "Phase" in chunk or "Fetching" in chunk:
                        st.write(chunk)
                    else:
                        full_analysis = chunk

                status.update(label="Comparison Complete", state="complete", expanded=False)

            # Show Result
            result_container.markdown(full_analysis)