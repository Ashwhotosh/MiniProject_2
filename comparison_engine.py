import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools_library import fetch_ipo_details, fetch_sentiment, query_rhp


def execute_peer_comparison(target_ipo, selected_peers, vector_store):
    """
    Generates a vast, multi-dimensional comparison report.
    """
    yield "🔄 **Phase 1: Analyzing Target's Competitive Landscape (RHP)...**"

    # 1. Extract Fundamental Comparison from Target RHP
    # The RHP always has a section comparing the company to peers. We extract that.
    rhp_fundamentals = "Target RHP not loaded. Fundamental comparison limited."
    if vector_store:
        yield "📖 Reading 'Industry Comparison' section from RHP..."
        q = """
        Extract the 'Comparison with Listed Industry Peers' or 'Basis for Issue Price' table.
        List the Peer Companies mentioned and their key financial ratios:
        - P/E (Price to Earnings)
        - EPS (Earnings Per Share)
        - RoNW (Return on Net Worth)
        - NAV (Net Asset Value)
        """
        rhp_fundamentals = query_rhp(target_ipo, q, vector_store=vector_store)

    # 2. Gather Live Market Data for ALL (Target + Peers)
    yield "📊 **Phase 2: Gathering Live Market Intelligence...**"

    market_data = {}
    companies_to_analyze = [target_ipo] + selected_peers

    for company in companies_to_analyze:
        role = "TARGET" if company == target_ipo else "PEER"
        yield f"🕵️ Scouting: **{company}** ({role})..."

        details = fetch_ipo_details(company)
        sentiment = fetch_sentiment(company, source="all")

        market_data[company] = {
            "Role": role,
            "Market Details": details,
            "Sentiment Summary": sentiment
        }

    # 3. Synthesis
    yield "⚖️ **Phase 3: Calculating Valuation & Rankings...**"

    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile", temperature=0.1)

    system_prompt = """
    You are a Senior Sector Analyst. You have:
    1. **Fundamental Data** extracted from the Target's RHP (P/E, RoNW of peers).
    2. **Live Market Data** (GMP, Price Band, Sentiment) for the Target and selected Peers.

    **Task:** Write a comprehensive "Peer Battle Report".

    **Report Structure:**

    ### 1. Financial Valuation (Fundamentals)
    - Use the RHP data to compare P/E, EPS, and RoNW.
    - Create a Markdown Table comparing the Target vs Peers on these metrics.
    - Analyze: Is the Target overvalued or undervalued compared to peers?

    ### 2. Grey Market & Demand (Hype)
    - Compare the GMP (%) and Market Sentiment of all companies.
    - Who has the strongest market momentum right now?

    ### 3. Strength & Weakness Matrix
    - Target's Key Advantage vs Peers.
    - Peers' Key Advantage vs Target.

    ### 4. The Leaderboard (Rank 1 to Last)
    - Rank them based on a mix of Valuation (Cheaper is better) and GMP (Higher is better).
    - **Verdict:** Justify why #1 is the best buy.
    """

    # Prepare inputs safely
    market_json = json.dumps(market_data, indent=2, default=str)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        **Target RHP Analysis (Fundamentals):**
        {rhp_data}

        **Live Market Data (GMP & Sentiment):**
        {market_data}

        Generate the Detailed Comparison Report now.
        """)
    ])

    chain = prompt | llm | StrOutputParser()

    analysis = chain.invoke({
        "rhp_data": rhp_fundamentals,
        "market_data": market_json
    })

    yield analysis