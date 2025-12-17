# report_engine.py
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools_library import fetch_ipo_details, fetch_sentiment, query_rhp


def generate_deep_dive_report(ipo_name, vector_store):
    """
    Orchestrates the data gathering and synthesis for a full report.
    """
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile", temperature=0.3)

    # --- PHASE 1: DATA GATHERING (The Checklist) ---
    yield "📊 **Phase 1: Gathering Market Data...**"

    # 1. Market Data
    details = fetch_ipo_details(ipo_name)

    # 2. Sentiment Scan
    yield "🗣️ **Phase 2: Scanning Market Sentiment...**"
    sentiment = fetch_sentiment(ipo_name, source="all")

    # 3. RHP Interrogation (If available)
    rhp_insights = "RHP Document not loaded. Fundamental analysis is limited."
    if vector_store:
        yield "📑 **Phase 3: Analyzing RHP Document (This may take 30s)...**"

        # We programmatically ask specific questions to the vector store
        questions = [
            "Summarize the Company's Business Model and Revenue Stream.",
            "What are the top 3 Internal Risk Factors mentioned?",
            "Who are the listed Peers and Competitors?",
            "What are the Key Financial indicators (Revenue/Profit growth)?"
        ]

        rhp_results = []
        for q in questions:
            # We use the existing query_rhp tool logic
            ans = query_rhp(ipo_name, q, vector_store=vector_store)
            rhp_results.append(f"**Question:** {q}\n**Evidence:** {ans}\n")

        rhp_insights = "\n".join(rhp_results)

    # --- PHASE 4: SYNTHESIS (The Writer) ---
    yield "✍️ **Phase 4: Writing Final Investment Memo...**"

    system_prompt = """
    You are a Senior Investment Banker writing a Due Diligence Report for a client.
    Use the raw data provided below to write a structured, professional report.

    **Report Structure:**
    1. **Executive Summary:** One paragraph overview (include GMP/Status).
    2. **Market Sentiment:** Summarize the public mood (Bullish/Bearish) based on Reddit/News.
    3. **Fundamental Analysis:**
       - Business Model
       - Financial Health
       - Peer Comparison
    4. **Risk Assessment:** Key risks involved.
    5. **Final Verdict:** A balanced conclusion (Avoid direct financial advice, use phrases like "Looking attractive" or "Wait and Watch").

    **Formatting:** Use Markdown, Bold headers, and Bullet points.
    """

    user_data = f"""
    IPO Name: {ipo_name}

    --- MARKET DATA ---
    {str(details)}

    --- SENTIMENT DATA ---
    {sentiment}

    --- RHP DOCUMENT INSIGHTS ---
    {rhp_insights}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()

    # Generate the final markdown
    final_report = chain.invoke({"input": user_data})

    yield final_report