# 📊 Feedback Analysis Agent

An AI-powered application that analyzes user feedback using natural language queries. Built with Streamlit, LangChain, and OpenAI.

## 🌟 Features

- **Natural Language Interface**: Ask questions about your feedback data in plain English
- **Intelligent Analysis**: Powered by OpenAI's GPT-4 and LangChain
- **Pandas Integration**: Automatically translates questions to pandas operations
- **Interactive UI**: Clean, user-friendly Streamlit interface
- **Data Visualization**: View dataset overview and statistics

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.8 or higher** installed on your system
- **OpenAI API Key**: Sign up at [OpenAI Platform](https://platform.openai.com/) and create an API key
- **Your feedback data**: A CSV file named `feedback.csv` in the project root directory

## 🚀 Installation

Follow these steps to set up the application locally:

### 1. Clone or Download the Project

```bash
cd feedback-anlaysis-agent
```

### 2. Create a Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` in a text editor and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

### 5. Add Your Data

Place your `feedback.csv` file in the project root directory. The CSV should contain your feedback data with columns such as:
- Rating/Score
- Feedback text/comments
- Date (optional)
- Any other relevant fields

**Example CSV structure:**
```csv
id,rating,comment,date
1,5,Great product!,2024-01-15
2,2,Needs improvement,2024-01-16
3,4,Good overall,2024-01-17
```

## ▶️ Running the Application

1. Make sure your virtual environment is activated
2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. The application will open in your default web browser at `http://localhost:8501`

## 💡 Usage

### Asking Questions

Once the app is running, you can ask questions like:

- "What is the average rating in the feedback?"
- "How many reviews have a 5-star rating?"
- "What are the main topics mentioned in feedback with scores lower than 3?"
- "What percentage of feedback is positive (score >= 4)?"
- "Show me the distribution of ratings"
- "What are the most common complaints?"

### Understanding the Interface

- **Dataset Overview**: Click to expand and see your data summary, column names, and basic statistics
- **Ask a Question**: Type your question in natural language
- **Example Questions**: Click to see suggested queries you can ask
- **Analyze Button**: Submit your question to get an AI-powered answer
- **Clear Button**: Reset the input field

## 🏗️ Project Structure

```
feedback-anlaysis-agent/
├── app.py                 # Main application file
├── feedback.csv           # Your feedback data (you provide this)
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create from .env.example)
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🔧 Troubleshooting

### "feedback.csv not found"
- Ensure your CSV file is named exactly `feedback.csv` (case-sensitive)
- Place it in the same directory as `app.py`

### "OpenAI API key not found"
- Check that `.env` file exists in the project root
- Verify the API key is set correctly: `OPENAI_API_KEY=sk-...`
- Restart the Streamlit application after adding the key

### Module Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify you're using the correct Python version (3.8+)
- Check that your virtual environment is activated

### API Rate Limits or Errors
- Verify your OpenAI account has available credits
- Check your API key is valid at [OpenAI Platform](https://platform.openai.com/)
- Consider switching to `gpt-3.5-turbo` in `app.py` for lower costs

## 🛠️ Customization

### Changing the LLM Model

Edit `app.py`, line with `ChatOpenAI` initialization:

```python
llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # Change to gpt-3.5-turbo for lower cost
    temperature=0,
    openai_api_key=api_key
)
```

### Adjusting Agent Behavior

Modify the `create_agent()` function parameters:
- `max_iterations`: Number of reasoning steps (default: 5)
- `verbose`: Set to `False` to hide agent reasoning in console

## 📊 Sample Questions by Use Case

### Quantitative Analysis
- "What is the average rating?"
- "How many total feedback entries do we have?"
- "What percentage of reviews are 4 stars or above?"

### Qualitative Analysis
- "What are common themes in negative feedback?"
- "Summarize the main complaints"
- "What do customers like most about the product?"

### Trend Analysis
- "How has feedback changed over time?"
- "What's the rating distribution?"
- "Are there any patterns in the data?"

## 🔒 Security Note

This application uses `allow_dangerous_code=True` to execute AI-generated pandas code. For production use:
- Implement stricter code validation
- Consider sandboxed execution environments
- Review and audit all generated code
- Use in trusted environments only

## 📝 Dependencies

- **streamlit**: Web interface framework
- **pandas**: Data manipulation and analysis
- **langchain**: Agent orchestration framework
- **langchain-openai**: OpenAI integration for LangChain
- **langchain-experimental**: Pandas dataframe agent
- **python-dotenv**: Environment variable management
- **openai**: OpenAI API client

## 📄 License

This project is provided as-is for educational and analytical purposes.

## 🤝 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Verify all installation steps were followed correctly
3. Ensure your CSV data is properly formatted

## 🎯 Future Enhancements

Potential improvements:
- Support for multiple data sources
- Export analysis results
- Custom visualization generation
- Conversation history
- Multi-file analysis
- Advanced NLP preprocessing

---

**Built with** 🦜 LangChain | 🤖 OpenAI | 📊 Streamlit | 🐼 Pandas
