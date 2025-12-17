import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Reuse the robust logic we already built
from tools_library import fetch_ipo_details, download_pdf_logic, build_vs_logic

load_dotenv()

st.set_page_config(page_title="RHP Document Chat", page_icon="📄", layout="centered")

# --- STATE MANAGEMENT ---
if "messages" not in st.session_state: st.session_state.messages = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "current_ipo" not in st.session_state: st.session_state.current_ipo = None

# --- SIDEBAR: SETUP ---
with st.sidebar:
    st.header("📄 Document Loader")
    ipo_input = st.text_input("Enter IPO Name:", value="ICICI Prudential Asset Management Company Limited")

    if st.button("Load Document", type="primary"):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.session_state.current_ipo = ipo_input

        with st.status("Processing Document...", expanded=True):
            st.write("1️⃣ Fetching Metadata...")
            details = fetch_ipo_details(ipo_input)

            if "id" in details:
                st.write(f"✅ Found ID: {details['id']}")

                st.write("2️⃣ Downloading PDF...")
                pdf_path = download_pdf_logic(details['id'])

                if pdf_path:
                    st.write("3️⃣ Building Vector Index...")
                    # This uses the persistent client logic we fixed earlier
                    st.session_state.vector_store = build_vs_logic(pdf_path)
                    st.success("Document Ready!")
                else:
                    st.error("❌ Failed to download PDF.")
            else:
                st.error("❌ IPO not found in database.")

# --- MAIN CHAT INTERFACE ---
st.title("📄 RHP Direct Chat")
if st.session_state.current_ipo:
    st.caption(f"Active Document: **{st.session_state.current_ipo}**")
else:
    st.info("👈 Please load a document from the sidebar to begin.")

# Display History
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# Input Handler
if prompt := st.chat_input("Ask about Peers, Risks, Financials..."):

    if not st.session_state.vector_store:
        st.error("Please load a document first.")
        st.stop()

    # 1. User Message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    # 2. AI Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Document..."):
            # --- RAG LOGIC ---
            llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model="llama-3.1-8b-instant",
                temperature=0.1
            )

            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})

            system_prompt = """
            You are a financial analyst assistant. 
            Answer the user's question strictly based on the context provided below.
            If the answer is not in the context, say "I cannot find this information in the RHP."

            Context:
            {context}
            """

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])

            chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt_template))

            # Execute
            response = chain.invoke({"input": prompt})
            answer = response['answer']

            # Display Answer
            st.markdown(answer)

            # Show Sources (Glass Box)
            with st.expander("View Source Context"):
                for i, doc in enumerate(response['context']):
                    st.markdown(f"**Chunk {i + 1} (Page {doc.metadata.get('page', 'Unknown')}):**")
                    st.caption(doc.page_content[:300] + "...")

            # Save History
            st.session_state.messages.append(AIMessage(content=answer))