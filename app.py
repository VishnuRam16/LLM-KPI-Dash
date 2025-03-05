import streamlit as st
import pandas as pd
import json
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ----------------------- Streamlit UI Styling -----------------------

st.set_page_config(page_title="AI-Powered Data Insights", page_icon="📊", layout="wide")

# Apply custom CSS for modern design
st.markdown("""
    <style>
        /* General Styling */
        body {
            color: white;
            background-color: #1c1c3c;
        }
        .stApp {
            background-color: #1c1c3c;
        }
        /* Title */
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #d8b9ff;
        }
        /* Subheader */
        .subheader {
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
            text-align: center;
        }
        /* Dataframe styling */
        .dataframe {
            background-color: #2a2a4a;
            border-radius: 10px;
            padding: 10px;
        }
        /* Button Styling */
        .stButton>button {
            background-color: #6a0dad;
            color: white;
            border-radius: 12px;
            padding: 10px 24px;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #501f85;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------- Smart Data Loading & Cleaning -----------------------

def load_data(file):
    """Load CSV or Excel file into a Pandas DataFrame and intelligently clean it."""
    if file is not None:
        try:
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
            df = clean_data(df)
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None

def clean_data(df):
    """Preprocess dataset while preserving all data types correctly."""
    df = df.copy()
    df.drop_duplicates(inplace=True)

    # Convert numeric columns
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.3:
                df[col] = df[col].astype("category")
        elif df[col].dtype == "bool":
            df[col] = df[col].astype(bool)
        else:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass

    # Convert date/time columns
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ["date", "time", "year", "day", "month"]):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass

    # Fill missing values
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=['category']).columns
    for col in cat_cols:
        df[col] = df[col].astype(str)
        df[col] = df[col].fillna("Unknown")
        df[col] = df[col].astype("category")

    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    datetime_cols = df.select_dtypes(include=['datetime']).columns
    for col in datetime_cols:
        df[col] = df[col].fillna(df[col].min())

    return df

# ----------------------- AI-Powered Insights -----------------------

def generate_insights(df):
    """Generate AI-powered insights dynamically based on dataset context."""
    dataset_summary = get_dataset_summary(df)

    prompt_template = """
    You are an expert data analyst. Analyze the dataset summary and generate key insights.

    **1. Identify Important Patterns & Trends:**  
    - Highlight key patterns in the dataset.  
    - Detect anomalies or unusual values.  
    - Identify trends in numerical and categorical data.  

    **2. Recommendations for Data Analysis:**  
    - Suggest what further analysis can be done.  
    - Recommend key business decisions based on insights.  

    **Dataset Summary (For AI Processing Only, Do Not Display):**
    {dataset_summary}

    **Provide insights in clear, human-readable format. DO NOT return JSON.**
    """

    llm = Ollama(model="mistral")
    prompt = PromptTemplate(template=prompt_template, input_variables=["dataset_summary"])
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    try:
        insights = llm_chain.run({"dataset_summary": dataset_summary})
        return insights
    except Exception as e:
        st.error(f"Error generating AI insights: {e}")
        return "⚠️ AI could not generate insights. Please check your data."

# ----------------------- AI-Driven Dataset Summary -----------------------

def get_dataset_summary(df):
    """Generate a structured dataset summary."""
    num_rows, num_cols = df.shape

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_columns = df.select_dtypes(include=['datetime']).columns.tolist()
    boolean_columns = df.select_dtypes(include=['bool']).columns.tolist()

    summary = f"""
    **Dataset Overview:**
    - Total Rows: {num_rows}
    - Total Columns: {num_cols}
    
    **Column Categories:**
    - Numeric Columns: {numeric_columns}
    - Categorical Columns: {categorical_columns}
    - Date/Time Columns: {datetime_columns}
    - Boolean Columns: {boolean_columns}

    **Sample Data (First 3 Rows):**
    {df.head(3).to_string(index=False)}
    """

    return summary

# ----------------------- Streamlit UI -----------------------

def main():
    st.markdown('<h1 class="title">🚀 AI-Powered Data Insights</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="subheader">📊 Upload Your Dataset & Get AI-Driven Insights</h2>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.markdown('<h2 class="subheader">📜 Data Preview</h2>', unsafe_allow_html=True)
            st.dataframe(df.head(10))  # Show first 10 rows

            st.markdown('<h2 class="subheader">🔍 AI-Generated Insights</h2>', unsafe_allow_html=True)
            insights = generate_insights(df)
            st.write(insights)

if __name__ == "__main__":
    main()