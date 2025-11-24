import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import base64
import mimetypes
import boto3 
from botocore.exceptions import ClientError
from io import BytesIO
import json
import tempfile 
import warnings # Re-adding warnings module for robust operation

# Suppress the openpyxl UserWarning that occurs on loading the Excel file
# This warning is common when reading files and doesn't affect functionality.
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl") 

# ---------------------- AWS CONFIG --------------------------
# Set these variables to match your AWS setup
S3_BUCKET_NAME = "zodopt"
S3_FILE_KEY = "Leaddata/Leads by Status.xlsx" 

# --- AWS Secrets Manager Configuration ---
AWS_REGION = "ap-south-1" 
GEMINI_SECRET_NAME = "salesbuddy/secrets" 
GEMINI_SECRET_KEY = "GEMINI_API_KEY" 

# ---------------------- CONFIG -----------------------------
GEMINI_MODEL = "gemini-2.5-flash" 

# Paths for local assets (used only for images/styling)
BACKGROUND_IMAGE_PATH = "background.jpg"
LOGO_IMAGE_PATH = "zodopt.png"

# --- TOKEN-SAVING CONFIGURATION ---
# Max leads to sample and send for analysis to prevent quota errors and save tokens
MAX_LEADS_TO_SEND = 150 
# Only send the most essential columns to save tokens.
CORE_ANALYSIS_COLS = ["Record Id", "Full Name", "Company", "Lead Status", "Annual Revenue", "City"]


REQUIRED_COLS = [
    "Record Id", "Full Name", "Lead Source", "Company", "Lead Owner",
    "Street", "City", "State", "Country", "Zip Code",
    "First Name", "Last Name", "Annual Revenue", "Lead Status"
]

DISQUALIFYING_STATUSES = ["Disqualified", "Closed - Lost", "Junk Lead"]

# ---------------------- FUNCTIONS: AWS & DATA LOADING --------------------------

@st.cache_resource
def get_secret(secret_name, region_name, key_name):
    """Fetches a specific key's value from a secret stored in AWS Secrets Manager."""
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
    """Downloads an Excel file from S3 and loads it into a Pandas DataFrame."""
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        excel_data = obj['Body'].read()
        
        df = pd.read_excel(BytesIO(excel_data))
        df.columns = df.columns.str.strip()

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"❌ Missing essential columns: **{', '.join(missing)}**"

        df_filtered = df[required_cols]
        # Ensure Annual Revenue is numeric, coercing errors to NaN, then filling with 0
        df_filtered['Annual Revenue'] = pd.to_numeric(df_filtered['Annual Revenue'], errors='coerce').fillna(0)
        return df_filtered, None

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return None, f"❌ S3 File not found: s3://**{bucket_name}/{file_key}**"
        return None, f"❌ S3 Error: {e}"
    except Exception as e:
        return None, f"❌ Error reading/processing data: {e}"


def filter_data_context(df, query):
    """
    Filters the DataFrame based on smart keywords, samples rows, and returns the filtered 
    data as a tab-separated CSV string AND the count of leads sent.
    """
    df_working = df.copy()
    query_lower = query.lower()
    
    # --- 1. Smart Filtering based on Query Keywords ---
    key_phrases = ["best leads", "hot leads", "convertible", "potential", "possibility", "high value"]
    if any(p in query_lower for p in key_phrases):
        df_working = df_working[~df_working["Lead Status"].isin(DISQUALIFYING_STATUSES)]

    # Filter by location keywords
    locations = ["bangalore", "bengaluru", "delhi", "new york", "london", "texas", "india"]
    loc_match = next((loc for loc in locations if loc in query_lower), None)

    if loc_match:
        mask = (
            df_working["City"].astype(str).str.lower().str.contains(loc_match, na=False) |
            df_working["State"].astype(str).str.lower().str.contains(loc_match, na=False) |
            df_working["Country"].astype(str).str.lower().str.contains(loc_match, na=False)
        )
        df_working = df_working[mask]
    
    # --- 2. Token Optimization: Column Reduction and Sampling ---
    if df_working.empty:
        return df.head(0).to_csv(index=False, sep="\t"), 0
        
    cols_to_send = [col for col in CORE_ANALYSIS_COLS if col in df_working.columns]
    df_working = df_working[cols_to_send]

    original_lead_count = len(df_working)

    if original_lead_count > MAX_LEADS_TO_SEND:
        # Prioritize leads by highest Annual Revenue before sampling
        df_working = df_working.sort_values(by='Annual Revenue', ascending=False)
        df_working = df_working.head(MAX_LEADS_TO_SEND)
    
    count_sent = len(df_working)
    
    # Return the data as a tab-separated string AND the count of leads sent
    return df_working.to_csv(index=False, sep="\t"), count_sent


# ---------------------- GEMINI INTERACTION --------------------------

def ask_gemini(question, data_context, api_key, count_sent):
    """
    Sends the question and filtered data context (as a string) to the Gemini API,
    including the count of records for context.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL) 

        # --- REFINED PROMPT STRUCTURE (The "questions training the model part") ---
        prompt = f"""
You are ZODOPT Sales Buddy, a highly accurate and senior sales analyst. You strictly analyze ONLY the following tab-separated CRM lead data.
The dataset contains a subset of records for analysis, totaling {count_sent} leads.
Your analysis MUST be limited exclusively to the provided subset. Do not guess, speculate, or hallucinate any values outside the dataset.
Ensure all numerical calculations (like sums, averages, counts, distributions) are mathematically accurate based on the data provided.

--- DATASET (Tab-Separated) ---
{data_context}

--- QUESTION ---
{question}

Provide clear, structured, and actionable bullet-point insights based ONLY on the data provided.
If the answer requires comparing groups or summarizing distributions, use Markdown tables for clarity (e.g., Lead Status | Count | Avg Revenue).
"""
        # --- END REFINED PROMPT ---

        response = model.generate_content(prompt)
        
        # Enhanced error handling for stability
        if response.text:
            return response.text
        
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason.name
            
            if finish_reason == "SAFETY":
                safety_ratings = response.candidates[0].safety_ratings
                reasons = [f"{r.category.name}: {r.probability.name}" for r in safety_ratings]
                return f"❌ **Analysis Blocked (Safety):** The prompt or data violated safety policies. Safety Reasons: {', '.join(reasons)}"
            
            if finish_reason != "STOP":
                 return f"⚠️ **Model Generation Failed:** The model stopped for reason: {finish_reason}. Try rephrasing your question or reducing complexity."

        return "⚠️ **Generation Error:** The model returned an empty response. Please try again or rephrase your question."

    except Exception as e:
        return f"❌ Gemini API Error: An exception occurred during the API call: {e}"


# ---------------------- STYLING AND UI FUNCTIONS ----------------------
def set_background(image_path):
    """Sets the Streamlit app background using CSS."""
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
            /* Streamlit layout adjustments */
            [data-testid="stAppViewContainer"] {{ padding: 0 !important; margin: 0 !important; background-color: transparent !important; }}
            [data-testid="stHeader"] {{ background: rgba(0,0,0,0) !important; }}

            /* Full-page background image */
            .stApp {{
                background-image: url("data:{mime_type};base64,{encoded_image}");
                background-size: cover !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                background-attachment: fixed !important;
            }}

            /* Content container styling for better readability */
            .main .block-container {{
                background: transparent !important;
                padding-top: 3rem;
                padding-left: 4rem;
                padding-right: 4rem;
            }}

            /* Text styling for visibility against background */
            h1, h2, h3, p, label, .stMarkdown, .stSelectbox label, .stButton button {{
                color: #111 !important;
                text-shadow: 0.5px 0.5px 1px rgba(255,255,255,0.8);
            }}

            /* Custom chat bubble styling */
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
    
    total_leads = len(df_filtered)
    st.success(f"Successfully loaded **{total_leads:,}** leads from S3.")

    # ---------------------- Chat Section ----------------------
    st.write("### 💬 Chat with Sales Buddy")

    if "chat" not in st.session_state:
        st.session_state.chat = [
            {"role": "ai", "content": f"Hello! I have loaded your CRM data containing {total_leads:,} leads. Ask me anything about your leads using the examples below!"}
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
        # Step 1: Filter data context based on query (returns CSV string and count now)
        data_ctx, count_sent = filter_data_context(df_filtered, query) 
        
        with st.spinner(f"Analyzing {count_sent:,} relevant lead records..."):
            
            # Step 2: Ask Gemini with the data string, API key, and count
            reply = ask_gemini(query, data_ctx, gemini_api_key, count_sent)

        st.session_state.chat.append({"role": "ai", "content": reply})
        st.rerun()

    # ---------------------- Sample Questions (The "questions training the model part" for the user) ----------------------
    st.divider()
    with st.expander("✨ Click for Sample Questions to Ask Your Sales Buddy", expanded=False):
        
        sample_questions = [
            "What is the **total Annual Revenue** associated with leads currently in the **'Qualified'** status?",
            "Show the **distribution of leads** across all lead statuses in a table.",
            "Which **Lead Owner** has the highest count of **'Negotiation/Review'** leads?",
            "List the **top 5 companies** by lead count that are *not* marked as 'Disqualified'.",
            
            "How many leads are in **New York** and what is their **average Annual Revenue**?",
            "List the **Full Names** and **Companies** of the top 3 high-value leads in **India**.",
            "Provide a breakdown of **Lead Sources** for leads located in the **State of Texas**.",
            "Compare the lead status distribution between leads from **'Partner'** and **'Trade Show'** sources.",

            "Who are our **hot convertible leads** (Full Name and Company) based on Annual Revenue?",
            "What is the **average Annual Revenue** for leads that have a high **potential** for conversion right now?",
            "What is the current **Lead Status** and **Annual Revenue** for the lead with **Record Id 12345** (use a sample ID from your data)?",
            "What is the total number of leads owned by **Jane Doe** and what is their combined annual revenue?"
        ]

        # Display the sample questions in two columns
        cols = st.columns(2)
        for i, q in enumerate(sample_questions):
            col_index = i % 2
            cols[col_index].markdown(f"**-** {q}", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
