import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools_library import fetch_ipo_details, fetch_sentiment


def execute_peer_comparison(target_ipo, selected_peers):
    """
    Fetches data for multiple IPOs and generates a ranking.
    """
    yield "🔄 **Phase 1: Gathering Competitive Intelligence...**"

    # 1. Collect Data for TARGET
    all_data = {}

    yield f"📊 Fetching data for Target: **{target_ipo}**..."
    target_details = fetch_ipo_details(target_ipo)
    target_senti = fetch_sentiment(target_ipo, source="all")

    all_data[target_ipo] = {
        "Details": target_details,
        "Sentiment": target_senti,
        "Role": "TARGET"
    }

    # 2. Collect Data for PEERS
    for peer in selected_peers:
        yield f"🕵️ Fetching data for Peer: **{peer}**..."
        p_details = fetch_ipo_details(peer)
        p_senti = fetch_sentiment(peer, source="all")

        all_data[peer] = {
            "Details": p_details,
            "Sentiment": p_senti,
            "Role": "PEER"
        }

    # 3. LLM Analysis
    yield "⚖️ **Phase 2: Calculating Rankings...**"

    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile", temperature=0.2)

    system_prompt = """
    You are a Portfolio Manager. You have data for a Target IPO and its Peers.

    **Your Goal:**
    1. Compare them based on GMP (Grey Market Premium), Financials (Price Band), and Market Sentiment.
    2. Rank them from #1 (Best Buy) to Last (Avoid).
    3. Justify the ranking. Why is #1 better than the Target? Or is the Target the best?

    **Rules:**
    - High GMP % is good.
    - Positive Sentiment is good.
    - If GMP is missing (N/A), consider it risky.

    **Output Format:**
    Use Markdown. Create a Comparison Table first, then the Ranking list with justifications.
    """

    # Convert dictionary to string
    data_str = json.dumps(all_data, indent=2, default=str)

    # --- CRITICAL FIX ---
    # We use {json_data} as a placeholder. We do NOT put f"{data_str}" inside the string.
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Analyze and Rank these IPOs based on the data below:\n\n{json_data}")
    ])

    chain = prompt | llm | StrOutputParser()

    # We pass the data string safely into the dictionary here
    analysis = chain.invoke({"json_data": data_str})

    yield analysis