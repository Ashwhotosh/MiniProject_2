import os
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools_library import fetch_ipo_details, fetch_sentiment, query_rhp


def generate_section(section_title, specific_questions, vector_store, ipo_name, llm):
    """
    Helper function to generate a single detailed chapter of the report.
    """
    if not vector_store:
        return f"## {section_title}\n*RHP Document not available for analysis.*\n"

    # 1. Gather Raw Data for this section
    raw_context = []
    for q in specific_questions:
        # We query the RHP for specific details (e.g. "What is the EPS?")
        ans = query_rhp(ipo_name, q, vector_store=vector_store)
        raw_context.append(f"Q: {q}\nA: {ans}")

    context_str = "\n\n".join(raw_context)

    # 2. Write the Section
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a Senior Equity Research Analyst writing a specific section of an IPO Due Diligence Report.

        **Rules:**
        1. Be EXTREMELY detailed. Do not summarize if data is available.
        2. Use Bullet points, Tables, and Bold text for readability.
        3. If financial numbers are available, present them in a Markdown Table.
        4. Focus strictly on the section topic provided.
        5. If data is missing in context, state "Not disclosed in the retrieved sections."
        """),
        ("human", f"""
        **Section Title:** {section_title}
        **IPO Name:** {ipo_name}

        **Raw Research Data:**
        {context_str}

        Write the content for this section now in Markdown format. 
        Start directly with the content (do not repeat the title).
        """)
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})


def generate_deep_dive_report(ipo_name, vector_store):
    """
    Orchestrates the creation of a massive, multi-chapter report.
    """
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile", temperature=0.2)

    yield "📊 **Initializing Deep Dive Analysis...**"

    # --- PHASE 1: EXTERNAL DATA ---
    market_data = fetch_ipo_details(ipo_name)
    sentiment_data = fetch_sentiment(ipo_name, source="all")

    full_report = [f"# 📑 Investment Research Report: {ipo_name}\n---\n"]

    # --- PHASE 2: GENERATE SECTIONS (The Loop) ---

    # Define the chapters and the specific questions to ask the PDF for each
    chapters = {
        "1. Executive Summary & Market Sentiment": {
            "questions": [],  # This uses external data, handled separately below
            "type": "intro"
        },
        "2. Company Overview & Business Model": {
            "questions": [
                "What is the core business model and history of the company?",
                "What products or services does the company offer?",
                "Who are the key clients and what is the revenue model?",
                "What is the industry overview and market size?"
            ],
            "type": "rhp"
        },
        "3. Financial Health (The Numbers)": {
            "questions": [
                "Provide the summary of financial statements (Balance Sheet, P&L) for the last 3 years.",
                "What is the Total Revenue, PAT (Profit After Tax), and EBITDA trends?",
                "What are the key ratios: EPS, RoNW, NAV per share?",
                "Details of Capital Structure and Debt/Borrowings."
            ],
            "type": "rhp"
        },
        "4. Objects of the Issue & Promoters": {
            "questions": [
                "What are the Objects of the Issue? How will the raised capital be used?",
                "Who are the Promoters and Management? Give their profiles.",
                "Details of Offer for Sale (OFS) vs Fresh Issue."
            ],
            "type": "rhp"
        },
        "5. Risk Factors & Litigation (Critical)": {
            "questions": [
                "List the top 5 internal risk factors mentioned in the RHP.",
                "Are there any outstanding criminal or civil litigations against the company or promoters?",
                "What are the regulatory and industry-specific risks?"
            ],
            "type": "rhp"
        },
        "6. Peer Comparison & Competitive Landscape": {
            "questions": [
                "Who are the listed peers and competitors mentioned?",
                "Compare the company with its competitors on financial metrics.",
                "What is the company's market positioning?"
            ],
            "type": "rhp"
        }
    }

    # Iterate through chapters
    for title, config in chapters.items():
        yield f"✍️ **Drafting Section: {title.split('.')[1].strip()}...**"

        if config["type"] == "intro":
            # Special handling for Intro using API/Sentiment data
            intro_prompt = f"""
            Write the **Executive Summary** and **Market Sentiment** section.

            **IPO Details:** {str(market_data)}
            **Sentiment Analysis:** {sentiment_data}

            Include:
            - Current GMP and Price Band.
            - Opening/Closing Dates.
            - Public Demand (Subscription status if available).
            - Summary of online sentiment (Bullish/Bearish).
            """
            response = llm.invoke(intro_prompt).content
            full_report.append(f"## {title}\n{response}\n")

        elif config["type"] == "rhp":
            # Deep retrieval for RHP sections
            section_content = generate_section(title, config["questions"], vector_store, ipo_name, llm)
            full_report.append(f"## {title}\n{section_content}\n")

        yield f"✅ {title.split('.')[1].strip()} Complete."

    # --- PHASE 3: FINAL VERDICT ---
    yield "⚖️ **Formulating Final Investment Verdict...**"

    verdict_prompt = f"""
    Based on the entire report generated so far, write a **Final Verdict**.

    **Report Context:**
    {"".join(full_report)}

    **Instructions:**
    1. Highlight the biggest Strength.
    2. Highlight the biggest Risk.
    3. Provide a conclusion: "Apply for Long Term", "Apply for Listing Gains", or "Avoid".
    4. Add a standard financial disclaimer.
    """
    verdict = llm.invoke(verdict_prompt).content
    full_report.append(f"## 7. Final Verdict\n{verdict}")

    # Return the full joined string
    final_markdown = "\n".join(full_report)
    yield final_markdown