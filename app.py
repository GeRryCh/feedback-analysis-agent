import streamlit as st
import pandas as pd
import os
from typing import Annotated, Optional, Dict
from typing_extensions import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

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
    pandas_analysis: Optional[str] = Field(
        None,
        description="The quantitative/filtering part of the query for pandas analysis (e.g., 'filter Level < 3', 'count by ServiceName')"
    )
    semantic_analysis: Optional[str] = Field(
        None,
        description="The semantic/qualitative part of the query for topic analysis (e.g., 'main topics', 'sentiment themes')"
    )


# ============================================================================
# ANALYSIS METHODS
# ============================================================================

def pandas_analysis(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Perform pandas data analysis and filtering.
    Returns the filtered DataFrame after applying create_pandas_dataframe_agent.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
    
    # Create a custom prompt that instructs the agent to return the DataFrame
    custom_prompt = f"""
    {query}
    
    IMPORTANT: After completing the analysis/filtering, you must return the resulting DataFrame object itself.
    If you filtered the data, return the filtered DataFrame.
    If you performed analysis without filtering, return the original DataFrame.
    Your final answer should be the DataFrame object, not a description of it.
    """
    
    pandas_agent_executor = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="tool-calling",
        verbose=True,
        allow_dangerous_code=True,
        return_intermediate_steps=True
    )
    
    try:
        result = pandas_agent_executor.invoke({"input": custom_prompt})
        
        # Try to extract the DataFrame from intermediate steps
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if len(step) > 1 and isinstance(step[1], pd.DataFrame):
                    return step[1]
        
        # If no DataFrame found in intermediate steps, return original
        return df
        
    except Exception as e:
        # Return original DataFrame on error (no Streamlit UI calls in tool)
        return df


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
    Based on the user's query: "{query}", analyze the following {len(sample_texts)} feedback comments.
    Identify the 3-5 most common topics relevant to the query. For each topic,
    provide a short title and a 1-sentence description.

    Feedback to analyze:
    ---
    {sample_texts}
    ---
    """
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
        response = llm.invoke([HumanMessage(content=prompt_template)])
        return response.content
    except Exception as e:
        return f"Error during topic analysis: {str(e)}"


def classify(query: str) -> Dict[str, Optional[str]]:
    """
    Classify and split the user query into quantitative and semantic parts.
    Returns a structured dictionary with keys for each analysis method.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
    
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
    
    Your task is to split user queries into two parts:
    1. pandas_analysis: For quantitative operations like filtering, counting, grouping, statistical analysis
    2. semantic_analysis: For qualitative analysis of the Text field like finding topics, themes, sentiment
    
    Examples:
    - "What are the 5 main topics of feedbacks with level < 3?"
      pandas_analysis: "filter where Level < 3"
      semantic_analysis: "find 5 main topics"
    
    - "How many feedbacks per ServiceName?"
      pandas_analysis: "count feedbacks grouped by ServiceName"
      semantic_analysis: None
    
    - "What are people complaining about?"
      pandas_analysis: None
      semantic_analysis: "identify complaint topics"
    
    If a part doesn't apply, return None for that field.
    """
    
    try:
        result = structured_llm.invoke([
            HumanMessage(content=f"System: {system_prompt}\n\nUser query: {query}")
        ])
        
        return {
            "pandas_analysis": result.pandas_analysis,
            "semantic_analysis": result.semantic_analysis
        }
    except Exception as e:
        # Return None for both on error (no Streamlit UI calls in tool)
        return {"pandas_analysis": None, "semantic_analysis": None}


@tool
def process_request(user_input: str) -> str:
    """Analyze feedback data based on user queries.

    This tool can handle various types of feedback analysis including:
    - Quantitative analysis: filtering, counting, grouping by columns (Level, ServiceName, etc.)
    - Qualitative analysis: identifying topics, themes, and sentiment from feedback text
    - Combined analysis: first filtering data, then analyzing the subset

    Examples of queries this tool handles:
    - "What are the 5 main topics of feedbacks with level < 3?"
    - "How many feedbacks per ServiceName?"
    - "What are people complaining about?"
    - "Show negative feedback and identify main issues"

    The tool automatically determines whether to use quantitative filtering,
    semantic analysis, or both based on the query content.

    Args:
        user_input: The user's question or request about the feedback data

    Returns:
        A comprehensive analysis result as a formatted string
    """
    # Load data
    df = load_data()
    if df is None:
        return "Error: Could not load data."
    
    # Classify the query
    classification = classify(user_input)
    
    results = []
    filtered_df = df
    
    # Process pandas analysis if present
    if classification.get("pandas_analysis"):
        filtered_df = pandas_analysis(df, classification["pandas_analysis"])

        # Generate a summary of the pandas operation
        if isinstance(filtered_df, pd.DataFrame):
            if len(filtered_df) != len(df):
                results.append(f"**Data filtered:** {len(filtered_df)} records (from {len(df)} total)")
            else:
                results.append(f"**Data analyzed:** {len(df)} records")

    # Process semantic analysis if present
    if classification.get("semantic_analysis"):
        semantic_result = semantic_analysis(filtered_df, classification["semantic_analysis"])
        results.append(f"**Semantic Analysis Results:**\n{semantic_result}")
    
    # If no analysis was needed
    if not classification.get("pandas_analysis") and not classification.get("semantic_analysis"):
        results.append("I couldn't identify what type of analysis you need. Please rephrase your question.")
    
    return "\n\n".join(results)


# ============================================================================
# LANGGRAPH SETUP
# ============================================================================

def build_graph(api_key):
    """Create and compile the custom LangGraph StateGraph with tool-calling."""

    # Create LLM with bound tools
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
    tools = [process_request]
    llm_with_tools = llm.bind_tools(tools)

    def agent(state: AgentState):
        """Agent node that decides whether to call tools or respond."""
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # Build the graph
    graph_builder = StateGraph(AgentState)

    # Add nodes
    graph_builder.add_node("agent", agent)
    graph_builder.add_node("tools", ToolNode(tools))

    # Add edges
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition  # Built-in helper that routes to "tools" if tool_calls exist, otherwise END
    )
    graph_builder.add_edge("tools", "agent")  # After tools, go back to agent

    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)


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

    with st.expander("📋 Dataset Overview", expanded=False):
        st.write(f"**Total Records:** {len(initial_df)}")
        st.dataframe(initial_df.head())

    if 'graph' not in st.session_state:
        st.session_state.graph = build_graph(api_key)
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    st.markdown("---")
    with st.expander("💡 Example Queries"):
        st.markdown("""
        **Simple Queries:**
        - "What are the 5 main topics of feedbacks with level < 3?"
        - "How many feedbacks per ServiceName?"
        - "What are people complaining about?"
        
        **Complex Queries:**
        - "Show me the negative feedback (Level < 3) and what are the main issues?"
        - "Filter feedbacks for ServiceName 'Support' and identify common themes"
        """)

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
                    # Process_request handles everything internally
                    initial_state = {"messages": st.session_state.messages}

                    config = {"configurable": {"thread_id": "streamlit-session"}}
                    result = st.session_state.graph.invoke(initial_state, config=config)

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