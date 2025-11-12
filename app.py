import streamlit as st
import pandas as pd
import os
from typing import Annotated, Optional, Dict, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Feedback Analysis Agent",
    page_icon="📊",
    layout="wide"
)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State schema for the LangGraph agent."""
    messages: Annotated[list[AnyMessage], add_messages]
    classification: Optional[Dict[str, Optional[str]]]
    data_frame: Optional[pd.DataFrame]
    quantitive_analysis_result: Optional[str]
    semantic_analysis_result: Optional[str]


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_data(file_path="feedback.csv") -> pd.DataFrame | None:
    """
    Load feedback data from CSV file into a Pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"❌ Error: '{file_path}' not found in the project directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None


# ============================================================================
# STRUCTURED OUTPUT MODELS
# ============================================================================

class QueryClassification(BaseModel):
    """Structured output for query classification"""
    quantitve_analysis: Optional[str] = Field(
        None,
        description="The quantitative/filtering part of the analysis query (e.g., 'filter Level < 3', 'count by ServiceName')"
    )
    semantic_analysis: Optional[str] = Field(
        None,
        description="The semantic/qualitative part of the analysis query (e.g., 'main topics', 'sentiment themes')"
    )


# ============================================================================
# ANALYSIS METHODS
# ============================================================================

def pandas_analysis(df: pd.DataFrame, user_query: str) -> dict:
    """
    Perform pandas data analysis and filtering.
    Returns a dictionary containing the filtered DataFrame, agent result, and intermediate steps.
    """
    llm = init_chat_model("openai:gpt-4o", temperature=0)

    pandas_agent_executor = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="tool-calling",
        verbose=True,
        allow_dangerous_code=True,
        return_intermediate_steps=True,
    )

    try:
        query = f"""IMPORTANT INSTRUCTIONS:
- When filtering, counting, or analyzing data, you MUST return the COMPLETE result set.
- NEVER use .head() to limit the results unless explicitly requested by the user.
- NEVER truncate or sample the data unless specifically asked to do so.
- Always show the full filtered dataset or complete analysis results.
- Generate classifications in original QUERY language.

QUERY: {user_query}
"""   
        result = pandas_agent_executor.invoke({"input": query})

        # Try to extract the DataFrame from intermediate steps
        extracted_df = df
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if len(step) > 1 and isinstance(step[1], pd.DataFrame):
                    extracted_df = step[1]
                    break

        # Return both the DataFrame, agent result, and intermediate steps
        return {
            "dataframe": extracted_df,
            "agent_result": result.get("output", "")
        }

    except Exception as e:
        # Return original DataFrame on error with error info
        return {
            "dataframe": df,
            "agent_result": f"Error: {str(e)}"
        }


def semantic_analysis(df: pd.DataFrame, query: str) -> str:
    """
    Analyzes the 'Text' column of the DataFrame to find semantic topics.
    Filters to first 300 rows for efficiency.
    """
    if 'Text' not in df.columns:
        return "Error: 'Text' column not found in DataFrame."
    
    if len(df) == 0:
        return "The DataFrame is empty. Nothing to analyze."
    
    SAMPLE_SIZE = 300
    sample_df = df.head(SAMPLE_SIZE)  # Use first 300 rows instead of random sample
    sample_texts = sample_df['Text'].dropna().tolist()
    
    if not sample_texts:
        return "No feedback text found in the data to analyze."
    
    prompt_template = f"""
    QUERY: "{query}"
    FEEDBACK CONTEXT:
    ---
    {sample_texts}
    ---
    """
    
    try:
        llm = init_chat_model("openai:gpt-4o", temperature=0)
        response = llm.invoke([HumanMessage(content=prompt_template)])
        return response.content
    except Exception as e:
        return f"Error during topic analysis: {str(e)}"


def classify_node(state: AgentState) -> AgentState:
    """Classify and split user queries into quantitative and semantic analysis parts.

    Uses the full conversation history to classify into:
    - quantitive_analysis: For filtering, counting, grouping, statistical analysis
    - semantic_analysis: For topics, themes, sentiment analysis
    """
    # Check if there are any messages
    if not state["messages"]:
        return {
            "classification": {"quantitve_analysis": None, "semantic_analysis": None}
        }

    llm = init_chat_model("openai:gpt-4o", temperature=0)

    # Create a structured LLM with the QueryClassification model
    structured_llm = llm.with_structured_output(QueryClassification)

    system_prompt = """
    You are a query classifier for a feedback analysis system.
    The feedback dataset has the following columns:
    - ID: Unique identifier
    - ServiceName: Name of the service
    - Level: Feedback level/rating (numeric)
    - Text: Feedback text content
    - ReferenceNumber: Reference number for tracking

    Your task is to analyze the conversation history and classify the user's intent into two parts:
    1. quantitve_analysis: For quantitative operations like filtering, counting, grouping, statistical analysis
    2. semantic_analysis: For qualitative analysis of the Text field like finding topics, themes, sentiment

    Examples:
    - "What are the 5 main topics of feedbacks with level < 3?"
      quantitve_analysis: "filter where Level < 3"
      semantic_analysis: "find 5 main topics"

    - "How many feedbacks per ServiceName?"
      quantitve_analysis: "count feedbacks grouped by ServiceName"
      semantic_analysis: None

    - "What are people complaining about?"
      quantitve_analysis: None
      semantic_analysis: "identify complaint topics"

    If a part doesn't apply, return None for that field.

    Consider the full conversation history to understand context and follow-up questions.

    IMPORTANT:
    Generate classifications in the same language as the user's query.
    """

    try:
        # Prepare messages with system prompt followed by full conversation history
        messages_for_classification = [SystemMessage(content=system_prompt)] + state["messages"]

        result = structured_llm.invoke(messages_for_classification)

        return {
            "classification": {
                "quantitve_analysis": result.quantitve_analysis,
                "semantic_analysis": result.semantic_analysis
            }
        }
    except Exception as e:
        # Return None for both on error
        return {
            "classification": {"quantitve_analysis": None, "semantic_analysis": None}
        }


# ============================================================================
# LANGGRAPH NODES
# ============================================================================

def orchestrator_node(state: AgentState) -> Command[Literal["classify"]]:
    """Orchestrator node that decides whether to route to classify chain or respond directly.

    This node uses the LLM to naturally determine if the user's query requires feedback
    data analysis (route to classify) or if it's a general question that can be answered
    directly (respond and end).
    """
    llm = init_chat_model("openai:gpt-4o", temperature=0)

    # System prompt that helps the LLM decide when to route vs respond
    system_message = SystemMessage(
        content="""You are a feedback analysis agent provided with analysis tool. Your purpose is to analyze customer feedback data
and help users understand patterns, trends, and insights.

**IMPORTANT ROUTING INSTRUCTIONS:**

1. For GENERAL questions or greetings, respond directly:
   - Greetings: "hello", "hi", "how are you"
   - Questions about your capabilities: "what can you do?", "how do you work?"
   - General conversation that doesn't require data analysis

   For these, provide a friendly, helpful response directly.

2. For ANALYSIS requests about feedback data, respond with: "ROUTE_TO_CLASSIFY"
   - Any question asking to analyze, filter, count, or explore feedback
   - Questions about topics, themes, patterns, trends in feedback
   - Requests to understand customer sentiment or complaints
   - Any query that requires looking at the actual feedback data

   For these, respond with exactly: "ROUTE_TO_CLASSIFY" (no other text)

Examples:
- "Hello!" → Respond directly with a greeting
- "What can you help me with?" → Respond directly explaining your capabilities
- "What are the main complaints?" → Respond: "ROUTE_TO_CLASSIFY"
- "How many feedbacks per service?" → Respond: "ROUTE_TO_CLASSIFY"
- "Show me negative feedback themes" → Respond: "ROUTE_TO_CLASSIFY"
"""
    )

    # Prepare messages with system prompt
    messages = [system_message] + state["messages"]

    # Invoke the LLM to decide
    response = llm.invoke(messages)
    response_text = response.content.strip()

    # Check if LLM decided to route to classify chain
    if "ROUTE_TO_CLASSIFY" in response_text:
        # Route to classify chain for data analysis
        return Command(goto="classify")
    else:
        # Respond directly and end
        return Command(
            update={"messages": [response]},
            goto=END
        )


def quantitve_analysis_node(state: AgentState) -> AgentState:
    """Node that performs pandas data analysis and filtering."""
    classification = state.get("classification", {})
    pandas_query = classification.get("quantitve_analysis") if classification else None

    if not pandas_query:
        # No pandas analysis needed, pass through
        return {"data_frame": load_data(), "quantitive_analysis_result": None}

    # Load data
    df = load_data()
    if df is None:
        return {"data_frame": None, "quantitive_analysis_result": "Error: Could not load data."}

    # Perform pandas analysis
    analysis_result = pandas_analysis(df, pandas_query)

    return {
        "data_frame": analysis_result["dataframe"],
        "quantitive_analysis_result": analysis_result["agent_result"],
    }


def semantic_analysis_node(state: AgentState) -> AgentState:
    """Node that performs semantic topic analysis on feedback text."""
    classification = state.get("classification", {})
    semantic_query = classification.get("semantic_analysis") if classification else None

    if not semantic_query:
        # No semantic analysis needed
        return {"semantic_analysis_result": None}

    # Get filtered DataFrame from state
    data_frame = state.get("data_frame")
    if data_frame is None:
        data_frame = load_data()

    if data_frame is None:
        return {"semantic_analysis_result": "Error: Could not load data."}

    # Perform semantic analysis
    result = semantic_analysis(data_frame, semantic_query)

    return {
        "semantic_analysis_result": result
    }


def format_results_node(state: AgentState) -> AgentState:
    """Node that formats the final results from all analyses.

    Priority-based output: semantic results take precedence over quantitative results.
    """
    # Priority 1: Semantic results
    if state.get("semantic_analysis_result"):
        final_message = state["semantic_analysis_result"]
    # Priority 2: Quantitative (pandas) results
    elif state.get("quantitive_analysis_result"):
        final_message = state["quantitive_analysis_result"]
    # Fallback: Error message
    else:
        final_message = "I couldn't identify what type of analysis you need. Please rephrase your question."

    return {"messages": [AIMessage(content=final_message)]}


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_after_classify(state: AgentState) -> str:
    """Route after classify node based on classification results in state."""
    classification = state.get("classification", {})

    if not classification:
        return "format_results"

    has_pandas = classification.get("quantitve_analysis") is not None
    has_semantic = classification.get("semantic_analysis") is not None

    # If both are needed, start with pandas (quantitative first)
    if has_pandas:
        return "quantitve_analysis"
    elif has_semantic:
        return "semantic_analysis"
    else:
        return "format_results"


def route_after_quantitive_analysys(state: AgentState) -> str:
    """Route after pandas analysis - check if semantic analysis is needed."""
    classification = state.get("classification", {})

    if not classification:
        return "format_results"

    has_semantic = classification.get("semantic_analysis") is not None

    if has_semantic:
        return "semantic_analysis"
    else:
        return "format_results"


# ============================================================================
# LANGGRAPH SETUP
# ============================================================================

def build_graph(api_key):
    """Create and compile the Command-based orchestrator LangGraph StateGraph.

    The graph starts with an orchestrator node that uses Command objects to decide
    whether to respond directly (END) or route to the classification processing chain.
    """

    # Build the graph
    graph_builder = StateGraph(AgentState)

    # Add nodes
    graph_builder.add_node("orchestrator", orchestrator_node)
    graph_builder.add_node("classify", classify_node)
    graph_builder.add_node("quantitve_analysis", quantitve_analysis_node)
    graph_builder.add_node("semantic_analysis", semantic_analysis_node)
    graph_builder.add_node("format_results", format_results_node)

    # Add edges
    # Start with the orchestrator (Command-based, routes automatically)
    graph_builder.add_edge(START, "orchestrator")

    # After classify, route based on classification results
    graph_builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "quantitve_analysis": "quantitve_analysis",
            "semantic_analysis": "semantic_analysis",
            "format_results": "format_results"
        }
    )

    # After quantitative analysis, route to semantic or results
    graph_builder.add_conditional_edges(
        "quantitve_analysis",
        route_after_quantitive_analysys,
        {
            "semantic_analysis": "semantic_analysis",
            "format_results": "format_results"
        }
    )

    # After semantic analysis, go to format results
    graph_builder.add_edge("semantic_analysis", "format_results")

    # After formatting results, end
    graph_builder.add_edge("format_results", END)

    # No checkpointer needed - Streamlit handles conversation history
    return graph_builder.compile()


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.title("📊 Feedback Analysis Agent")
    st.markdown("Ask questions about your feedback data. The agent can filter data and then run analysis on the subset.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OpenAI API key not found! Please set it in your .env file.")
        st.stop()

    # Load initial data
    initial_df = load_data()
    if initial_df is None:
        st.stop()

    with st.expander("📋 Dataset Overview", expanded=True):
        st.write(f"**Total Records:** {len(initial_df)}")
        st.dataframe(initial_df.head())

    if 'graph' not in st.session_state:
        st.session_state.graph = build_graph(api_key)
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"): st.write(msg.content)
        elif isinstance(msg, AIMessage) and msg.content:
            with st.chat_message("assistant"): st.write(msg.content)

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"): st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Invoke the graph with the current messages
                    initial_state = {"messages": st.session_state.messages}
                    result = st.session_state.graph.invoke(initial_state)

                    assistant_message = result["messages"][-1]
                    st.write(assistant_message.content)
                    st.session_state.messages.append(assistant_message)

                except Exception as e:
                    error_msg = f"❌ Error: {e}\n\nPlease try rephrasing."
                    st.error(error_msg)
                    st.session_state.messages.append(AIMessage(content=error_msg))

    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            # Reset the conversation
            st.rerun()

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: gray;'>Powered by LangGraph & OpenAI</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()