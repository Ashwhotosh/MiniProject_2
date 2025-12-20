# 🚀 IPO Smooth Operator: Agentic AI Investment Analyst

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red?style=for-the-badge&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-v0.3-green?style=for-the-badge&logo=langchain)
![Groq](https://img.shields.io/badge/AI-Llama3.3-orange?style=for-the-badge)

**IPO Smooth Operator** is an advanced Multi-Agent AI system designed to automate the due diligence process for Initial Public Offerings (IPOs).

Unlike standard chatbots, this system acts as a **Financial Analyst Brain**. It intelligently routes queries, scrapes real-time market data (GMP), analyzes social sentiment (Reddit/News), and performs Deep Retrieval (RAG) on official RHP (Red Herring Prospectus) PDF documents to provide a 360° investment view.

---

## 🌟 Key Features

### 1. 🧠 Intelligent "Brain" Architecture
The system doesn't just guess; it plans.
- **Intent Recognition:** Uses a Planner Agent to break down complex user queries (e.g., *"What is the sentiment and list the risk factors?"*) into executable steps.
- **Parallel Execution:** Can fetch GMP, scan Reddit, and query the PDF simultaneously.
- **Loop Prevention:** Implements "Tool Stripping" logic to ensure the AI never gets stuck in recursive loops.

### 2. 📄 RHP Document RAG (Retrieval-Augmented Generation)
- **Auto-Scraping:** Automatically scrapes the IPO page to find the specific RHP/DRHP PDF link (prioritizing Final RHP over Drafts).
- **Vector Search:** Embeds the 400+ page document into a ChromaDB vector store using HuggingFace embeddings.
- **Raw Query Injection:** Bypasses summarization loss by injecting the user's exact questions into the vector search for maximum accuracy.

### 3. 📊 Automated 360° Due Diligence Reports
- **Chained Sectioning:** Generates a massive, structured Investment Memo by writing it chapter-by-chapter (Financials, Risks, Promoters, etc.).
- **Hybrid Data:** Combines hard data (Price Band, GMP) with soft data (Sentiment) and fundamental data (RHP).

### 4. ⚔️ Advanced Peer Comparison
- **Concurrent Analysis:** Compares the Target IPO against other *currently active* IPOs (filters out already listed ones).
- **Fundamental Extraction:** Reads the "Industry Comparison" section from the Target's RHP to extract P/E, EPS, and RoNW ratios of competitors.
- **Battle Matrix:** Ranks peers based on a weighted mix of Valuation (Fundamentals) vs. Demand (GMP/Sentiment).

---
## 📂 Project Structure

IPO_Project
├── .env                     # API Keys (Groq, Reddit)
├── app.py                   # Main Streamlit UI (Glassmorphism design)
├── brain.py                 # The Logic Controller (Planner)
├── tools_library.py         # The Workers (Scrapers, Vector DB, RAG)
├── report_engine.py         # Logic for generating 360° Reports
├── comparison_engine.py     # Logic for Peer Comparison Battles
├── requirements.txt         # Dependency list (Golden Set)
├── rhp_chat.py              # Standalone RHP Chat Debugger
└── pdfs/                    # Auto-downloaded RHP documents

---

## 🛠️ System Architecture

The project follows a **Plan-and-Execute** pattern rather than a simple Re-Act loop.

```mermaid
graph TD
    User["User Query"] --> Brain["💡 The Brain (Planner)"]
    
    subgraph Agents ["Tool Execution Layer"]
        Brain -->|Fetch Price/Dates| GMP["📉 GMP Agent"]
        Brain -->|Scan Socials| Senti["🗣️ Sentiment Agent"]
        Brain -->|Vector Search| RAG["📄 RHP Document Agent"]
        RAG -.->|Fallback| Web["🌐 Web Search Agent"]
    end
    
    GMP --> Synthesis
    Senti --> Synthesis
    RAG --> Synthesis
    Web --> Synthesis
    
    Synthesis["📝 LLM Synthesis"] --> Output["Final Answer"]


