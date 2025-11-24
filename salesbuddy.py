import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import base64
import mimetypes
import boto3
import io
import json

# ---------------------- AWS CONFIG (UPDATED) --------------------------
# Set these variables to match your AWS setup
S3_BUCKET_NAME = "zodopt"
S3_FILE_KEY = "Leaddata/Leads by Status.xlsx" # Corrected S3 path

# --- AWS Secrets Manager Configuration ---
AWS_REGION = "ap-south-1" # Mumbai region
GEMINI_SECRET_NAME = "salesbuddy/secrets" # Corrected Secret path
GEMINI_SECRET_KEY = "GEMINI_API_KEY" # Key used within the JSON structure of the secret

# Paths for local assets (used only for images/styling)
BACKGROUND_IMAGE_PATH = "background.jpg"
LOGO_IMAGE_PATH = "zodopt.png"

# --- TOKEN-SAVING CONFIGURATION ---
# To prevent 429 quota errors, limit the dataset size sent to the LLM
MAX_LEADS_TO_SEND = 150  # Limit to 150 leads for LLM analysis. Adjust based on quota.
# Only send the most essential columns to save tokens.
CORE_ANALYSIS_COLS = ["Record Id", "Full Name", "Company", "Lead Status", "Annual Revenue", "City"]


REQUIRED_COLS = [
    "Record Id", "Full Name", "Lead Source", "Company", "Lead Owner",
    "Street", "City", "State", "Country", "Zip Code",
    "First Name", "Last Name", "Annual Revenue", "Lead Status"
]

DISQUALIFYING_STATUSES = ["Disqualified", "Closed - Lost", "Junk Lead"]

# ---------------------- SECRETS MANAGER FUNCTION --------------------------

@st.cache_resource
def get_secret_value(secret_name, secret_key, region_name):
    """
    Retrieves the secret from AWS Secrets Manager using the specified region.
    """
    try:
        client = boto3.client(
            service_name='secretsmanager',
            region_name=region_name
        )

        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except Exception as e:
        st.error(f"❌ Secrets Manager Error: Failed to retrieve secret '{secret_name}' in region **{region_name}**. Ensure the IAM Role has 'secretsmanager:GetSecretValue' permission. Details: {e}")
        st.stop()
        return None

    if 'SecretString' in get_secret_value_response:
        secret = get_secret_value_response['SecretString']

        # Parse the JSON string to get the specific key's value
        try:
            secret_dict = json.loads(secret)
            if secret_key in secret_dict:
                return secret_dict[secret_key]
            else:
                st.error(f"❌ Secrets Manager Error: Key '{secret_key}' not found in the secret JSON for '{secret_name}'.")
                st.stop()
        except json.JSONDecodeError:
            st.error(f"❌ Secrets Manager Error: Secret string for '{secret_name}' is not valid JSON.")
            st.stop()
    return None

# ---------------------- FUNCTIONS --------------------------

@st.cache_data(ttl=600)
def load_sales_data_from_s3(bucket_name, file_key, required_cols):
    """Loads the Excel file directly from S3 into a Pandas DataFrame."""
    try:
        s3 = boto3.client('s3', region_name=AWS_REGION)

        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        excel_data = response['Body'].read()
        df = pd.read_excel(io.BytesIO(excel_data))

        df.columns = df.columns.str.strip()

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"❌ Missing essential columns: {', '.join(missing)}. Check your file structure."

        df_filtered = df[required_cols]
        df_filtered['Annual Revenue'] = pd.to_numeric(df_filtered['Annual Revenue'], errors='coerce').fillna(0)

        return df_filtered, None

    except Exception as e:
        return None, f"❌ S3/Boto3 Error: Failed to load data from s3://{bucket_name}/{file_key} in **{AWS_REGION}**. Details: {e}"


def filter_data_context(df, query):
    df_working = df.copy()
    query_lower = query.lower()

    # --- 1. Smart Filtering ---
    key_phrases = ["best leads", "hot leads", "convertible", "potential", "possibility"]
    if any(p in query_lower for p in key_phrases):
        df_working = df_working[~df_working["Lead Status"].isin(DISQUALIFYING_STATUSES)]

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
        return df.head(0).to_csv(index=False, sep="\t")


    # --- 2. Token Optimization: Column Reduction ---
    cols_to_send = [col for col in CORE_ANALYSIS_COLS if col in df_working.columns]
    df_working = df_working[cols_to_send]
    
    
    # --- 3. Token Optimization: Row Sampling ---
    if len(df_working) > MAX_LEADS_TO_SEND:
        # Prioritize high-value leads by sorting before sampling
        df_working = df_working.sort_values(by='Annual Revenue', ascending=False)
        df_working = df_working.head(MAX_LEADS_TO_SEND)

    return df_working.to_csv(index=False, sep="\t")


def ask_gemini(question, data_context):
    try:
        # Fetch the key securely
        gemini_api_key = get_secret_value(GEMINI_SECRET_NAME, GEMINI_SECRET_KEY, AWS_REGION)

        if not gemini_api_key:
            return "❌ API Key Error: Gemini API key could not be retrieved from AWS Secrets Manager."

        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
You are ZODOPT Sales Buddy. You strictly analyze ONLY the following tab-separated CRM lead data.
The data provided is a filtered or sampled subset of the full database. 
Your analysis must be limited to the provided subset.

--- DATASET (Sampled/Filtered) ---
{data_context}

--- QUESTION ---
{question}

Provide clear, structured, bullet-point insights based ONLY on the data provided.
"""
        
        response = model.generate_content(prompt)
        
        # --- ENHANCED RESPONSE CHECK ---
        if response.text:
            return response.text
        
        # Handle cases where the model generates a response object but no text (e.g., safety block)
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason.name
            
            if finish_reason == "SAFETY":
                safety_ratings = response.candidates[0].safety_ratings
                reasons = [f"{r.category.name}: {r.probability.name}" for r in safety_ratings]
                return f"❌ Analysis Blocked: The model failed to generate a response due to safety policies. Safety Reasons: {', '.join(reasons)}"
            
            if finish_reason != "STOP":
                 return f"⚠️ Model Generation Failed: The model stopped for reason: {finish_reason}. Try rephrasing your question."
        
        # Fallback for unknown empty response
        return "⚠️ Generation Error: The model returned an empty response. Please try again or rephrase your question."

    except Exception as e:
        # Catches API key errors, connection issues, or unhandled 429s/403s.
        return f"❌ Gemini API Error: {e}"


# ---------------------- BACKGROUND CSS ----------------------
def set_background(image_path):
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

            h1, h2, h3, p, label {{
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

    # Load data from S3
    df_filtered, load_msg = load_sales_data_from_s3(S3_BUCKET_NAME, S3_FILE_KEY, REQUIRED_COLS)
    if df_filtered is None:
        st.error(load_msg)
        st.stop()

    # Chat section
    st.write("### 💬 Chat with Sales Buddy")

    if "chat" not in st.session_state:
        # Refined professional initial message
        initial_message = f"Welcome! I have successfully loaded **{len(df_filtered)}** CRM leads. How can I assist you with lead analysis today?"
        
        st.session_state.chat = [
            {"role": "ai", "content": initial_message}
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

        with st.spinner("Analyzing..."):
            data_ctx = filter_data_context(df_filtered, query)
            reply = ask_gemini(query, data_ctx)

        st.session_state.chat.append({"role": "ai", "content": reply})
        st.rerun()


if __name__ == "__main__":
    main()
