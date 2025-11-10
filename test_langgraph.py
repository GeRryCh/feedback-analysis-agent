"""
Quick test script to verify LangGraph setup works correctly.
"""
import os
from dotenv import load_dotenv
import pandas as pd
from app import create_graph, load_data

load_dotenv()

def test_graph():
    """Test that the graph can be created and invoked."""
    print("Loading data...")
    df = load_data()
    if df is None:
        print("❌ Failed to load data")
        return False
    print(f"✓ Loaded {len(df)} records")

    print("\nChecking API key...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return False
    print("✓ API key found")

    print("\nCreating LangGraph...")
    try:
        graph = create_graph(df, api_key)
        print("✓ Graph created successfully")
    except Exception as e:
        print(f"❌ Failed to create graph: {e}")
        return False

    print("\n" + "="*60)
    print("🎉 All tests passed! The refactored app is ready to use.")
    print("="*60)
    print("\nRun the app with: streamlit run app.py")
    return True

if __name__ == "__main__":
    test_graph()
