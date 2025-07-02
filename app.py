
import streamlit as st
import pandas as pd
import os
import sys
import subprocess
import tempfile
import traceback
import time
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import glob
import urllib.parse
from langchain_community.utilities import SQLDatabase
from langchain.tools import Tool
from crewai import LLM, Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import Field

    # Configure page
st.set_page_config(
    page_title="ERP Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set page background
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
    }
    
    .stSidebar {
        background-color: #f1f5f9;
    }
    
    .stTextArea textarea {
        background-color: #f8fafc !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        color: #1f2937 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    .stDataFrame {
        border: none !important;
    }
    
    .stMetric {
        background: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS for sophisticated styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .status-success {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
    }
    
    .status-processing {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    
    .status-error {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ef4444;
        margin: 1rem 0;
    }
    
    .query-input {
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        font-size: 1.1rem;
        background: #f9fafb;
        transition: all 0.3s ease;
    }
    
    .result-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
    }
    

    
    .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    
    .dataframe-container {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'crew_result' not in st.session_state:
    st.session_state.crew_result = None
if 'dataframe_result' not in st.session_state:
    st.session_state.dataframe_result = None
if 'visualization_paths' not in st.session_state:
    st.session_state.visualization_paths = []

# Database configuration
@st.cache_resource
def initialize_database():
    """Initialize database connection"""
    try:
       openai_api_key = os.getenv("OPENAI_API_KEY")
        
        password = urllib.parse.quote_plus("Ispl@2025")
        DATABASE_URI = f"mysql+pymysql://kavin:{password}@192.168.1.134:3306/_b7843b7b27adc018"
        
        db = SQLDatabase.from_uri(DATABASE_URI)
        return db
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

# SQL Tool class
class SQLTool(BaseTool):
    name: str = "sql_tool"
    description: str = "A tool for running SQL queries against the database. Use this to execute SELECT statements to retrieve data, INSERT to add records, UPDATE to modify existing data, DELETE to remove records, and other SQL operations. Provide the complete SQL query as input."
    
    db: any = Field(default_factory=lambda: initialize_database())
    
    def _run(self, query: str) -> str:
        """Execute the SQL query and return results"""
        try:
            return str(self.db.run(query))
        except Exception as e:
            return f"Error performing SQL query: {str(e)}"

# Initialize CrewAI components
@st.cache_resource
def setup_crew():
    """Setup CrewAI agents and tasks"""
    llm = LLM(model="openai/gpt-4.1-mini-2025-04-14",api_key=openai_api_key)
    
    agent_1 = Agent(
        role="ERP SQL Database Assistant",
        goal="Understand user queries in natural language and provide accurate data insights that match ERP report results by analyzing MariaDB schema and executing appropriate SQL queries with proper business logic and filtering",
        backstory="You are an experienced ERP data analyst who specializes in translating business questions into SQL queries for MariaDB-based ERP systems. You have deep knowledge of ERP database structures, understand how ERP reports apply business rules and filters, and can quickly identify table relationships, column meanings, and data patterns. You ensure your SQL results match exactly what users see in their ERP reports by applying the same filtering logic, date contexts, and business rules. You communicate findings clearly in natural language while being precise about the data you're presenting and always cross-reference against expected ERP report behavior.",
        tools=[SQLTool()],
        llm=llm
    )

    task_1 = Task(
        description="Analyze the user's natural language query {query} and generate SQL that produces results identical to what they would see in their ERP reports. First explore the database schema to understand available tables and columns, then consider what business filters and logic the ERP system applies by default (such as enabled/disabled status, active records, date ranges). Generate and execute appropriate SQL queries that include proper joins, filtering, and calculations to match ERP report standards. Present the results in a clear, conversational manner that aligns with how data appears in the ERP interface. Handle follow-up questions and maintain context throughout the conversation while ensuring consistency with ERP report behavior.",
        expected_output="A natural language response that directly answers the user's question with relevant data insights that match ERP report results, including any clarifications about the data source, business logic applied, limitations, or additional context that would be helpful for ERP users.",
        agent=agent_1
    )

    agent_2 = Agent(
        role="Data Conversion Specialist",
        goal="Produce only the raw contents of a valid Python script (dataframe.py) that converts SQL query results from ERP SQL Database Assistant into pandas DataFrames and exports them as CSV, with no extra text, no markdown code fences, and no surrounding quotes.",
        backstory="You are a data processing specialist who writes clean, production-ready Python scripts. Your scripts load SQL query results (assumed to be JSON/dictionary/list of dictionaries), convert them to pandas DataFrames with correct columns and data types, and export them as CSV files for analysis. You ensure the output is ONLY the code needed for dataframe.py, without any explanation or formatting.",
        llm=llm
    )

    task_2 = Task(
        description="Take the output from agent_1 (ERP SQL Database Assistant) and produce ONLY the raw contents of a valid dataframe.py file. The file should contain complete, executable Python code that loads the given SQL query result (as a dictionary or list of dictionaries), converts it to a pandas DataFrame with correct column names and data types, and exports it as a CSV file. Do not include any markdown formatting (like ```python), no text before or after, and no quotes at the start or end. Output ONLY the code itself, so it can be saved directly as dataframe.py without modification.",
        expected_output="The exact contents of a valid Python script (dataframe.py) containing only executable pandas code to convert SQL results into a DataFrame and export to CSV, with no extra text or formatting.",
        agent=agent_2,
        output_file="dataframe.py",
        context=[task_1]
    )

    agent_3 = Agent(
        role="Data Visualization Specialist",
        goal="Produce only the raw contents of a valid Python script (diagram.py) that creates seaborn visualizations from pandas DataFrames, with no extra text, no markdown code fences, and no surrounding quotes.",
        backstory="You are a data visualization expert who writes clear, production-ready Python scripts using seaborn and matplotlib. Your scripts analyze DataFrame columns to choose suitable plots (bar, line, scatter, heatmap, etc.), set proper titles, labels, legends, and save plots as high-quality images. You ensure the output is ONLY the code for diagram.py with no explanations or formatting.",
        llm=llm
    )

    task_3 = Task(
        description="Take the pandas DataFrame output from agent_2 and produce ONLY the raw contents of a valid diagram.py file. The file should contain complete, executable Python code that uses seaborn and matplotlib to create appropriate plots based on the DataFrame's columns. Include clear titles, axis labels, legends, and save the plots as high-quality images. Output ONLY the code itself with no additional text, no markdown formatting (like ```python), and no quotes at the start or end. The code must be ready to save and run as a .py file without modification.",
        expected_output="The exact contents of a valid Python script (diagram.py) containing only executable seaborn and matplotlib code to produce professional visualizations from the DataFrame and save them as images, with no extra text or formatting.",
        agent=agent_3,
        output_file="diagram.py",
        context=[task_2]
    )

    crew = Crew(
        agents=[agent_1, agent_2, agent_3],
        tasks=[task_1, task_2, task_3],
        process=Process.sequential
    )
    
    return crew

def execute_python_file(file_path, description):
    """Execute a Python file and capture results"""
    try:
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        # Execute the file
        with open(file_path, 'r') as f:
            code = f.read()
        
        exec(code, {'__name__': '__main__'})
        
        # Restore stdout
        sys.stdout = old_stdout
        
        output = captured_output.getvalue()
        return True, output, None
        
    except Exception as e:
        sys.stdout = old_stdout
        return False, None, str(e)

def display_dataframe_results():
    """Display DataFrame results if CSV exists"""
    csv_files = glob.glob("*.csv")
    if csv_files:
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                st.markdown("### 📊 Generated Dataset")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📈 Rows</h3>
                        <h2>{len(df)}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📋 Columns</h3>
                        <h2>{len(df.columns)}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>💾 File Size</h3>
                        <h2>{os.path.getsize(csv_file)//1024} KB</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True, height=400)
                st.markdown('</div>', unsafe_allow_html=True)
                
                return df
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    return None

def display_visualizations():
    """Display generated visualizations"""
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.svg']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(ext))
    
    if image_files:
        st.markdown("### 📈 Generated Visualizations")
        
        for i, img_path in enumerate(image_files):
            try:
                st.markdown(f'<div class="chart-container">', unsafe_allow_html=True)
                
                if img_path.endswith('.svg'):
                    with open(img_path, 'r') as f:
                        svg_content = f.read()
                    st.image(svg_content)
                else:
                    image = Image.open(img_path)
                    st.image(image, use_column_width=True, caption=f"Visualization {i+1}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error displaying image {img_path}: {str(e)}")

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 ERP Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Natural Language to SQL Analytics with Automated Visualizations</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <h3>🎯 Dashboard Features</h3>
            <ul>
                <li>Natural Language Queries</li>
                <li>Automated SQL Generation</li>
                <li>Data Processing & Export</li>
                <li>Smart Visualizations</li>
                <li>ERP-Specific Analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔧 System Status")
        db = initialize_database()
        if db:
            st.markdown('<div class="status-success">✅ Database Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ Database Connection Failed</div>', unsafe_allow_html=True)
    
    # Main content
    st.markdown("### 💬 Natural Language Query")
    
    # Query input
    query = st.text_area(
        "Enter your business question:",
        placeholder="e.g., Which employees have the highest number of leave applications this year?",
        height=100,
        key="query_input"
    )
    
    # Auto-process when query is entered
    if query and query.strip():
        if not st.session_state.analysis_complete or st.session_state.get('last_query') != query:
            st.session_state.last_query = query
            st.session_state.analysis_complete = False
            
            # Processing status
            st.markdown('<div class="status-processing">🔄 Processing your query...</div>', unsafe_allow_html=True)
            
            with st.spinner("Analyzing query and generating insights..."):
                try:
                    crew = setup_crew()
                    result = crew.kickoff({"query": query})
                    st.session_state.crew_result = result
                    
                    # Execute dataframe.py if it exists
                    if os.path.exists("dataframe.py"):
                        st.markdown('<div class="status-processing">📊 Generating dataset...</div>', unsafe_allow_html=True)
                        success, output, error = execute_python_file("dataframe.py", "Data Processing")
                        if success:
                            st.session_state.dataframe_result = output
                        else:
                            st.error(f"Error executing dataframe.py: {error}")
                    
                    # Execute diagram.py if it exists
                    if os.path.exists("diagram.py"):
                        st.markdown('<div class="status-processing">📈 Creating visualizations...</div>', unsafe_allow_html=True)
                        success, output, error = execute_python_file("diagram.py", "Visualization Generation")
                        if not success:
                            st.warning(f"Visualization generation had issues: {error}")
                    
                    st.session_state.analysis_complete = True
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.code(traceback.format_exc())
        
        # Display results if analysis is complete
        if st.session_state.analysis_complete:
            st.markdown('<div class="status-success">✅ Analysis Complete</div>', unsafe_allow_html=True)
            
            # Display DataFrame results
            df = display_dataframe_results()
            
            # Display visualizations
            display_visualizations()
    
    else:
        # Welcome message
        st.markdown("""
        <div class="result-container">
            <h3>👋 Welcome to ERP Analytics Dashboard</h3>
            <p>Enter a natural language query above to get started. The system will:</p>
            <ul>
                <li>🔍 Analyze your question</li>
                <li>🗄️ Generate appropriate SQL queries</li>
                <li>📊 Process and export data</li>
                <li>📈 Create relevant visualizations</li>
                <li>💡 Provide business insights</li>
            </ul>
            <p><strong>Example queries:</strong></p>
            <ul>
                <li>"Which employees have the highest number of leave applications this year?"</li>
                <li>"Show me sales performance by region for the last quarter"</li>
                <li>"What are the top 5 products by revenue this month?"</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
