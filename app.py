import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Feedback Analysis Agent",
    page_icon="📊",
    layout="wide"
)


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


def create_feedback_agent(df, api_key):
    """
    Create a LangChain pandas dataframe agent using the pre-built agent.

    Args:
        df (pd.DataFrame): DataFrame to analyze
        api_key (str): OpenAI API key

    Returns:
        AgentExecutor: Configured LangChain agent executor
    """
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=api_key
        )

        # Create the pandas dataframe agent with built-in functionality
        agent_executor = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type="tool-calling",
            verbose=True,
            allow_dangerous_code=True  # Required flag for code execution
        )

        return agent_executor
    except Exception as e:
        st.error(f"❌ Error creating agent: {str(e)}")
        return None


def process_query(agent, query):
    """
    Process a user query using the agent.

    Args:
        agent: LangChain agent executor
        query (str): User's natural language question

    Returns:
        str: Agent's response
    """
    try:
        # Use the invoke method with the correct input format
        response = agent.invoke({
            "input": query
        })

        # Extract the output from the response
        if isinstance(response, dict) and 'output' in response:
            return response['output']

        return str(response)

    except Exception as e:
        return f"❌ Error processing query: {str(e)}\n\nPlease try rephrasing your question or make it more specific."


def main():
    """Main application function."""

    # Header
    st.title("📊 Feedback Analysis Agent")
    st.markdown("""
    Ask questions about your feedback data in natural language, and get intelligent answers powered by AI.
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
        st.write("**Data Types:**")
        # Convert dtypes to a DataFrame for proper display
        dtypes_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Data Type': df.dtypes.values.astype(str)
        })
        st.dataframe(dtypes_df, width='stretch')
        st.write("**Basic Statistics:**")
        st.dataframe(df.describe(), width='stretch')

    # Create agent (cached in session state)
    if 'agent' not in st.session_state:
        with st.spinner("Initializing AI agent..."):
            st.session_state.agent = create_feedback_agent(df, api_key)
            if st.session_state.agent is None:
                st.stop()

    # Query interface
    st.markdown("---")
    st.subheader("🤔 Ask a Question")

    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What is the average rating/score in the feedback?
        - How many reviews have a 5-star rating?
        - What are the main topics mentioned in feedback with scores lower than 3?
        - What percentage of feedback is positive (score >= 4)?
        - Show me the most common words in negative feedback
        - How has the feedback trend changed over time?
        - What is the distribution of ratings?
        """)

    # Query input
    user_query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the main topic of feedback with a score lower than 3?",
        key="user_query"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("🔍 Analyze", type="primary", width='stretch')
    with col2:
        if st.button("🗑️ Clear", width='stretch'):
            st.session_state.user_query = ""
            st.rerun()

    # Process query
    if submit_button and user_query:
        with st.spinner("🤖 Analyzing your question..."):
            answer = process_query(st.session_state.agent, user_query)

        st.markdown("---")
        st.subheader("💡 Answer")
        st.success(answer)
    elif submit_button:
        st.warning("⚠️ Please enter a question first.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
        Powered by LangChain 🦜 and OpenAI 🤖
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
