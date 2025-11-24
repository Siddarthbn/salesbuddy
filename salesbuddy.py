# Sales Buddy

import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import base64
import mimetypes
import boto3 # AWS SDK
from botocore.exceptions import ClientError
from io import BytesIO
import json

# ---------------------- AWS CONFIG --------------------------
# Set these variables to match your AWS setup
S3_BUCKET_NAME = "zodopt"
S3_FILE_KEY = "Leaddata/Leads by Status.xlsx" # Corrected S3 path

# --- AWS Secrets Manager Configuration ---
AWS_REGION = "ap-south-1" # Mumbai region
GEMINI_SECRET_NAME = "salesbuddy/secrets" # Corrected Secret path
GEMINI_SECRET_KEY = "GEMINI_API_KEY" # Key used within the JSON structure of the secret

# ---------------------- CONFIG -----------------------------
GEMINI_MODEL = "gemini-2.5-flash"

# Paths for local assets (used only for images/styling)
BACKGROUND_IMAGE_PATH = "background.jpg"
LOGO_IMAGE_PATH = "zodopt.png"

REQUIRED_COLS = [
    "Record Id", "Full Name", "Lead Source", "Company", "Lead Owner",
    "Street", "City", "State", "Country", "Zip Code",
    "First Name", "Last Name", "Annual Revenue", "Lead Status"
]

DISQUALIFYING_STATUSES = ["Disqualified", "Closed - Lost", "Junk Lead"]

# ---------------------- FUNCTIONS: AWS & DATA LOADING --------------------------

@st.cache_resource
def get_secret(secret_name, region_name, key_name):
    """
    Fetches a specific key's value from a secret stored in AWS Secrets Manager.
    """
    try:
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )
        
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )

        if 'SecretString' in get_secret_value_response:
            secret = json.loads(get_secret_value_response['SecretString'])
            return secret.get(key_name), None
        else:
            return None, "❌ Secret is not a JSON string (binary secret not supported for API Key)."
    
    except ClientError as e:
        error_map = {
            'ResourceNotFoundException': f"Secret **{secret_name}** not found.",
            'InvalidRequestException': "Invalid request to Secrets Manager.",
            'InvalidParameterException': "Invalid parameter in Secrets Manager request.",
        }
        return None, f"❌ Secrets Manager Error: {error_map.get(e.response['Error']['Code'], str(e))}"
    except Exception as e:
        return None, f"❌ Unexpected error in Secrets Manager: {e}"


@st.cache_data(ttl=600)
def load_data_from_s3(bucket_name, file_key, required_cols):
    """
    Downloads an Excel file from S3 and loads it into a Pandas DataFrame.
    """
    try:
        s3 = boto3.client('s3')
        # Download the file object from S3
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        excel_data = obj['Body'].read()
        
        # Read the Excel data directly into a DataFrame from memory
        df = pd.read_excel(BytesIO(excel_data))
        df.columns = df.columns.str.strip()

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"❌ Missing essential columns: **{', '.join(missing)}**"

        df_filtered = df[required_cols]
        # Use errors='coerce' to turn non-numeric values into NaN for safety
        df_filtered['Annual Revenue'] = pd.to_numeric(df_filtered['Annual Revenue'], errors='coerce')
        return df_filtered, None

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return None, f"❌ S3 File not found: s3://**{bucket_name}/{file_key}**"
        return None, f"❌ S3 Error: {e}"
    except Exception as e:
        return None, f"❌ Error reading/processing data: {e}"


def filter_data_context(df, query):
    """
    Filters the DataFrame based on smart keywords (e.g., location, 'hot leads') 
    to reduce the data context sent to the LLM.
    """
    df_working = df.copy()
    query_lower = query.lower()

    # Smart lead filtering: Exclude disqualified leads for "best/hot" queries
    key_phrases = ["best leads", "hot leads", "convertible", "potential", "possibility", "high value"]
    if any(p in query_lower for p in key_phrases):
        df_working = df_working[~df_working["Lead Status"].isin(DISQUALIFYING_STATUSES)]

    # Location extraction
    locations = ["bangalore", "bengaluru", "delhi", "new york", "london", "texas", "india"]
    loc_match = next((loc for loc in locations if loc in query_lower), None)

    if loc_match:
        mask = (
            df_working["City"].astype(str).str.lower().str.contains(loc_match, na=False) |
            df_working["State"].astype(str).str.lower().str.contains(loc_match, na=False) |
            df_working["Country"].astype(str).str.lower().str.contains(loc_match, na=False)
        )
        df_working = df_working[mask]
        if df_working.empty:
            # Important: return empty df with columns to maintain structure
            df_working = df.head(0) 

    return df_working.to_csv(index=False, sep="\t")


def ask_gemini(question, data_context, api_key):
    """
    Sends the question and filtered data context to the Gemini API.
    """
    try:
        # Configure API key for the current session/request
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)

        prompt = f"""
You are ZODOPT Sales Buddy. You strictly analyze ONLY the following tab-separated CRM lead data.
Do not guess or hallucinate any values outside the dataset.

--- DATASET ---
{data_context}

--- QUESTION ---
{question}

Provide structured bullet-point insights, using the data fields provided.
"""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"❌ Gemini API Error: {e}"


# ---------------------- BACKGROUND CSS ----------------------
def set_background(image_path):
    # (Function remains the same for background styling)
    try:
        if not os.path.exists(image_path):
            return

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"

        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode()

        st.markdown(f"""
            <style>
            [data-testid="stAppViewContainer"] {{ padding: 0 !important; margin: 0 !important; background-color: transparent !important; }}
            [data-testid="stHeader"] {{ background: rgba(0,0,0,0) !important; }}

            .stApp {{
                background-image: url("data:{mime_type};base64,{encoded_image}");
                background-size: cover !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                background-attachment: fixed !important;
            }}

            .main .block-container {{
                background: transparent !important;
                padding-top: 3rem;
                padding-left: 4rem;
                padding-right: 4rem;
            }}

            h1, h2, h3, p, label, .stMarkdown, .stSelectbox label, .stButton button {{
                color: #111 !important;
                text-shadow: 0.5px 0.5px 1px rgba(255,255,255,0.8);
            }}

            .chat-bubble-user {{
                background: rgba(230,247,255,0.8);
                padding: 10px;
                border-radius: 10px;
                margin: 5px 0;
                text-align: right;
            }}

            .chat-bubble-ai {{
                background: rgba(255,255,255,0.85);
                padding: 12px;
                border-radius: 10px;
                margin: 5px 0;
                border-left: 4px solid #32CD32;
            }}
            </style>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Background error: {e}")

# ---------------------- STREAMLIT APP ----------------------

def main():

    st.set_page_config(
        page_title="ZODOPT Sales Buddy",
        page_icon=LOGO_IMAGE_PATH,
        layout="wide"
    )

    set_background(BACKGROUND_IMAGE_PATH)

    # Header
    col1, col2 = st.columns([6,1])

    with col1:
        st.title("💰 ZODOPT Sales Buddy Agent")

    with col2:
        if os.path.exists(LOGO_IMAGE_PATH):
            st.image(LOGO_IMAGE_PATH, width=95)

    st.divider()

    # --- 1. Load API Key from Secrets Manager ---
    with st.spinner("Securing API Key from AWS Secrets Manager..."):
        gemini_api_key, secret_msg = get_secret(GEMINI_SECRET_NAME, AWS_REGION, GEMINI_SECRET_KEY)
    
    if gemini_api_key is None:
        st.error(secret_msg)
        st.stop()
    
    # --- 2. Load Data from S3 ---
    with st.spinner(f"Loading sales data from S3 bucket **{S3_BUCKET_NAME}**..."):
        df_filtered, load_msg = load_data_from_s3(S3_BUCKET_NAME, S3_FILE_KEY, REQUIRED_COLS)
    
    if df_filtered is None:
        st.error(load_msg)
        st.stop()
    
    st.success(f"Successfully loaded **{len(df_filtered):,}** leads from S3.")

    # ---------------------- Chat Section ----------------------
    st.write("### 💬 Chat with Sales Buddy")

    if "chat" not in st.session_state:
        st.session_state.chat = [
            {"role": "ai", "content": "Hello! I have loaded your CRM data. Ask me anything about your leads using the examples below!"}
        ]

    # Render chat
    for msg in st.session_state.chat:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-bubble-user'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-ai'>{msg['content']}</div>", unsafe_allow_html=True)

    # Chat input
    query = st.chat_input("Ask your CRM-related question...")

    if query:
        st.session_state.chat.append({"role": "user", "content": query})

        # Process the query using the loaded data and key
        with st.spinner("Analyzing..."):
            # Step 1: Filter data context based on query
            data_ctx = filter_data_context(df_filtered, query)
            
            # Step 2: Ask Gemini with the context
            reply = ask_gemini(query, data_ctx, gemini_api_key)

        st.session_state.chat.append({"role": "ai", "content": reply})
        st.rerun()

    # ---------------------- Sample Questions ----------------------
    st.divider()
    with st.expander("✨ Click for Sample Questions to Ask Your Sales Buddy", expanded=False):
        
        sample_questions = [
            # Lead Status and Pipeline Analysis (4)
            "How many leads are currently in the **'Negotiation/Review'** status?",
            "What is the **distribution of leads** across all lead statuses?",
            "Which **lead owner** has the highest number of **'Qualified'** leads?",
            "Show me a list of all leads that are currently **'Open'** but have an **Annual Revenue greater than 50,000**.",
            
            # Geographical and Locational Insights (4)
            "How many leads do we have in **Bangalore**?",
            "What is the average Annual Revenue of leads located in the **State of Texas**?",
            "Provide a breakdown of lead sources for leads in **New York**.",
            "List the top 5 companies from **India**.",

            # High-Value & Hot Lead Identification (4)
            "Who are our **best convertible leads** based on Annual Revenue?",
            "Analyze the **hot leads** and tell me which **Lead Source** is performing the best.",
            "Give me the **full names** and **companies** of all leads not marked as 'Disqualified' or 'Lost'.",
            "Which leads have the highest **potential** for conversion right now?",
            
            # Source and Company Analysis (4)
            "Which **Lead Source** has generated the most leads?",
            "What is the **average Annual Revenue** for leads sourced from **'Web Download'**?",
            "List the **top 10 companies** by lead count.",
            "Compare the lead status distribution between leads from **'Partner'** and **'Trade Show'** sources.",
            
            # Individual Lead Detail Lookups (4)
            "What is the current **Lead Status** and **Annual Revenue** for **John Doe** (or a specific Record Id)?",
            "Who is the **Lead Owner** for the company **Acme Corp**?",
            "Provide the **Street, City, and Zip Code** for the lead named **Jane Smith**.",
            "What is the total number of leads owned by **Sarah Connor** and what is their combined annual revenue?"
        ]

        # Display the sample questions in two columns
        cols = st.columns(2)
        for i, q in enumerate(sample_questions):
            col_index = i % 2
            cols[col_index].markdown(f"**-** {q}", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
