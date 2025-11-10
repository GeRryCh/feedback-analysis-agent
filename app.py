import streamlit as st
import pandas as pd
import os
import random
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.tools import Tool, tool
from langgraph.graph import StateGraph, START
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
def load_data(file_path="feedback.csv"):
    """
    Load feedback data from CSV file into a Pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded DataFrame or None if error occurs
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"❌ Error: '{file_path}' not found in the project directory.")
        st.info("Please ensure your feedback.csv file is in the root directory of this project.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None


# ============================================================================
# TOOL 1: PANDAS DATA ANALYZER
# ============================================================================

def create_pandas_analyzer_tool(df, api_key):
    """
    Create a tool that wraps the pandas dataframe agent.

    Args:
        df (pd.DataFrame): DataFrame to analyze
        api_key (str): OpenAI API key

    Returns:
        Tool: LangChain Tool wrapping the pandas agent
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=api_key
    )

    # Create the pandas dataframe agent
    pandas_agent_executor = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="tool-calling",
        verbose=True,
        allow_dangerous_code=True
    )

    def analyze_data(query: str) -> str:
        """Execute pandas agent for data analysis."""
        try:
            result = pandas_agent_executor.invoke({"input": query})
            if isinstance(result, dict) and 'output' in result:
                return result['output']
            return str(result)
        except Exception as e:
            return f"Error analyzing data: {str(e)}"

    return Tool(
        name="analyze_data",
        func=analyze_data,
        description=(
            "Use this tool for all statistical analysis, data queries, counting records, "
            "filtering data, calculating averages, analyzing distributions, finding patterns "
            "in structured data, and any questions about the feedback dataset's numbers or statistics. "
            "This tool has access to the complete feedback DataFrame with columns: "
            "ID, ServiceName, Level, Text, ReferenceNumber, RequestID, ProcessID, CreationDate."
        )
    )


# ============================================================================
# TOOL 2: SEMANTIC TOPIC EXTRACTION
# ============================================================================

def create_semantic_topics_tool(feedback_texts: list[str]):
    """
    Create a tool that analyzes semantic topics from pre-loaded feedback texts.

    Args:
        feedback_texts (list[str]): The feedback texts to analyze

    Returns:
        Tool: A LangChain tool for semantic topic extraction
    """

    @tool
    def extract_semantic_topics() -> str:
        """
        Analyzes feedback texts to find high-level SEMANTIC topics and categories.
        Use this tool *only* when the user asks for 'main topics', 'themes', or 'what people are talking about'.
        This tool uses an AI model to understand the *meaning* of the feedback.

        Returns:
            str: A formatted string of the 5 main categories found.
        """
        texts = feedback_texts  # Use the feedback_texts from the closure

        if not texts:
            return "No feedback text was provided to analyze."

        # --- 1. Sample the Data ---
        # We can't send 5000+ comments. Let's take a representative sample.
        SAMPLE_SIZE = 300
        if len(texts) > SAMPLE_SIZE:
            sample_texts = random.sample(texts, SAMPLE_SIZE)
        else:
            sample_texts = texts

        # Filter out any non-string or empty data
        sample_texts = [str(t) for t in sample_texts if isinstance(t, str) and len(t.strip()) > 10]

        if len(sample_texts) < 10:
            return f"Not enough text data to analyze (found only {len(sample_texts)} valid comments)."

        # --- 2. Create the AI Analyst Prompt ---
        # This is the most important part.
        prompt_template = f"""
        You are a professional data analyst. I have a list of {len(sample_texts)} user feedback comments.
        Your job is to read all of them and identify the 5 most common, high-level topics or categories.

        **Instructions:**
        1.  **Analyze Meaning:** Do NOT just count words. Understand the *intent* and *meaning* of the comments.
        2.  **Group Synonyms:** Group similar ideas. For example, 'cannot log in', 'forgot password', and 'sign-in issue' should all be one topic.
        3.  **Provide Categories:** Return 5 categories.
        4.  **Format:** For each category, provide a short, clear title (e.g., "Positive Feedback on Clarity") and a 1-sentence description of what it includes.
        5.  **Ignore common words:** Do not list "thank you", "very", "was", etc., as topics. Focus on *actionable themes*.

        Here is the feedback to analyze:
        ---
        {sample_texts}
        ---
        End of feedback.

        Please provide your 5-category analysis.
        """

        # --- 3. Call the LLM to perform the analysis ---
        try:
            # Initialize a new LLM instance for this specific task
            # We use gpt-4o for better reasoning on this complex task
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return "Error: OPENAI_API_KEY not found for topic analysis tool."

            llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)

            messages = [HumanMessage(content=prompt_template)]

            response = llm.invoke(messages)

            analysis_result = response.content

            return f"Here is an AI-powered analysis of the main feedback themes:\n\n{analysis_result}"

        except Exception as e:
            return f"An error occurred while analyzing topics: {str(e)}"

    return extract_semantic_topics
    

# ============================================================================
# LANGGRAPH SETUP
# ============================================================================

def create_graph(df, api_key):
    """
    Create and compile the LangGraph StateGraph with tools.

    Args:
        df (pd.DataFrame): DataFrame for analysis
        api_key (str): OpenAI API key

    Returns:
        Compiled LangGraph application
    """
    # Extract feedback texts for topic analysis
    feedback_texts = []
    if 'Text' in df.columns:
        feedback_texts = df['Text'].dropna().tolist()
    # Create tools
    pandas_tool = create_pandas_analyzer_tool(df, api_key)
    topics_tool = create_semantic_topics_tool(feedback_texts)  # Create tool with feedback texts pre-bound
    tools = [pandas_tool, topics_tool]

    # Create LLM with tools bound
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=api_key
    )
    llm_with_tools = llm.bind_tools(tools)

    # Define agent node
    def agent(state: AgentState):
        """Agent node that calls the LLM with tools."""
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # Build the graph
    graph_builder = StateGraph(AgentState)

    # Add nodes
    graph_builder.add_node("agent", agent)
    graph_builder.add_node("tools", ToolNode(tools))

    # Add edges
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", tools_condition)
    graph_builder.add_edge("tools", "agent")

    # Compile with memory
    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """Main application function."""

    # Header
    st.title("📊 Feedback Analysis Agent")
    st.markdown("""
    Ask questions about your feedback data in natural language. This AI agent can:
    - Analyze statistical data and run queries on your feedback
    - Extract topics and themes from feedback text
    """)

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OpenAI API key not found!")
        st.info("""
        Please set your OpenAI API key:
        1. Copy `.env.example` to `.env`
        2. Add your API key: `OPENAI_API_KEY=your_key_here`
        3. Restart the application
        """)
        st.stop()

    # Load data
    with st.spinner("Loading feedback data..."):
        df = load_data()

    if df is None:
        st.stop()

    # Display data info
    with st.expander("📋 Dataset Overview", expanded=False):
        st.write(f"**Total Records:** {len(df)}")
        st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
        st.write("**First 5 rows:**")
        st.dataframe(df.head(), width='stretch')
        st.write("**Basic Statistics:**")
        # Exclude ID columns from statistics (they're identifiers, not meaningful metrics)
        id_columns = ['ID', 'ReferenceNumber', 'RequestID', 'ProcessID']
        stats_df = df.drop(columns=[col for col in id_columns if col in df.columns], errors='ignore')
        st.dataframe(stats_df.describe(), width='stretch')

    st.session_state.graph = create_graph(df, api_key)
    if st.session_state.graph is None:
        st.error("Failed to initialize agent.")
        st.stop()

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    st.markdown("---")

    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **Data Analysis Questions:**
        - What is the average Level (rating) in the feedback?
        - How many records have a Level of 5?
        - What percentage of feedback has a Level >= 4?
        - Show me the distribution of ratings

        **Topic/Theme Questions:**
        - What are the main topics mentioned in the feedback?
        - Extract the key themes from user feedback
        - What keywords appear most frequently?
        """)

    # Display chat history
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    # Chat input
    if prompt := st.chat_input("Ask a question about the feedback data..."):
        # Add user message to chat history
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Invoke the graph
                    config = {"configurable": {"thread_id": "streamlit-session"}}
                    result = st.session_state.graph.invoke(
                        {"messages": st.session_state.messages},
                        config=config
                    )

                    # Extract the assistant's response
                    assistant_message = result["messages"][-1]

                    # Display and store the response
                    st.write(assistant_message.content)
                    st.session_state.messages.append(assistant_message)

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}\n\nPlease try rephrasing your question."
                    st.error(error_msg)
                    st.session_state.messages.append(AIMessage(content=error_msg))

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
        Powered by LangGraph 🦜🕸️ and OpenAI 🤖
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
