import os
import json
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List, Literal
from tools_library import fetch_ipo_details, fetch_sentiment, query_rhp
from dotenv import load_dotenv

load_dotenv()


# 1. Define the Planner Structure
class ToolCall(BaseModel):
    tool_name: Literal["gmp_tool", "sentiment_tool", "rhp_tool"]
    arguments: str = Field(description="The specific question or argument for the tool")


class Plan(BaseModel):
    steps: List[ToolCall] = Field(description="List of tools to execute")


# 2. The Brain Logic
def execute_brain(user_query, ipo_name, vector_store):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        yield "❌ Error: GROQ_API_KEY not found in .env file."
        return

    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0)

    # --- STEP 1: PLANNING ---
    structured_llm = llm.with_structured_output(Plan)

    system_prompt = f"""
    You are an expert IPO Analyst Brain.
    User Query: "{user_query}"
    Current IPO: "{ipo_name}"

    Break the query into steps. Available Tools:
    1. 'gmp_tool': For Price, GMP, Dates, Listing, Status. (Arg: 'details')
    2. 'sentiment_tool': For Market Mood, Hype. (Arg: 'reddit', 'news', or 'all')
    3. 'rhp_tool': Strictly for information found in the PDF (Peers, Financials, Risks).
       - IMPORTANT: For the argument, copy the User's specific question about the document verbatim. Do not summarize it to a keyword.
    """

    try:
        plan = structured_llm.invoke(system_prompt)
    except Exception as e:
        yield f"Error in planning: {e}"
        return

    results = []

    # --- STEP 2: EXECUTION ---
    for step in plan.steps:
        yield f"⚙️ **Executing:** {step.tool_name}..."

        output = ""
        if step.tool_name == "gmp_tool":
            output = str(fetch_ipo_details(ipo_name))

        elif step.tool_name == "sentiment_tool":
            output = fetch_sentiment(ipo_name, source=step.arguments)

        elif step.tool_name == "rhp_tool":
            # --- FIX IS HERE: RAW QUERY INJECTION ---
            # If the user is just asking a document question (single step plan),
            # we use their EXACT query instead of the AI's summarized argument.
            # This ensures 'rhp_chat.py' level accuracy.
            search_query = step.arguments

            if len(plan.steps) == 1:
                search_query = user_query

            # Pass vector_store explicitly
            output = query_rhp(ipo_name, query=search_query, vector_store=vector_store)
            # ----------------------------------------

        results.append(f"--- RESULT FROM {step.tool_name.upper()} ---\n{output}\n")
        yield f"✅ {step.tool_name} Complete."

    # --- STEP 3: SYNTHESIS ---
    yield "🧠 **Synthesizing Final Answer...**"

    final_prompt = f"""
    User Query: {user_query}

    Data Collected:
    {"".join(results)}

    Answer the user's query professionally based ONLY on the data above.
    """

    final_response = llm.invoke(final_prompt).content
    yield final_response