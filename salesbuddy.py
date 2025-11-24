import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import base64
import mimetypes
import boto3
import io

# ---------------------- AWS S3 CONFIG --------------------------
# Set these variables to match your AWS setup
S3_BUCKET_NAME = "zodopt"
S3_FILE_KEY = "Leads by Status.xlsx"               
AWS_REGION = "ap-south-1"                      

# Paths for local assets (used only for images/styling)
BACKGROUND_IMAGE_PATH = "background.jpg"
LOGO_IMAGE_PATH = "zodopt.png"

GEMINI_API_KEY = "AIzaSyBgKTlULVARw37Ec0WCor0YFC3cHXq64Mc" # Use environment variables in production

REQUIRED_COLS = [
    "Record Id", "Full Name", "Lead Source", "Company", "Lead Owner",
    "Street", "City", "State", "Country", "Zip Code",
    "First Name", "Last Name", "Annual Revenue", "Lead Status"
]

DISQUALIFYING_STATUSES = ["Disqualified", "Closed - Lost", "Junk Lead"]

# ---------------------- FUNCTIONS --------------------------

@st.cache_data(ttl=600)
def load_sales_data_from_s3(bucket_name, file_key, required_cols):
    """Loads the Excel file directly from S3 into a Pandas DataFrame."""
    try:
        # boto3 automatically picks up credentials from the EC2 Instance Profile (IAM Role)
        s3 = boto3.client('s3', region_name=AWS_REGION)
        
        # Download the file object from S3
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        
        # Read the Excel data into a BytesIO object (in-memory file)
        excel_data = response['Body'].read()
        df = pd.read_excel(io.BytesIO(excel_data))
        
        df.columns = df.columns.str.strip()

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"❌ Missing essential columns: {', '.join(missing)}. Check your file structure."

        df_filtered = df[required_cols]
        # Use errors='coerce' to turn non-numeric values into NaN, then fill with 0
        df_filtered['Annual Revenue'] = pd.to_numeric(df_filtered['Annual Revenue'], errors='coerce').fillna(0)
        
        return df_filtered, None

    except Exception as e:
        # Catch S3/Boto3 errors (e.g., AccessDenied, NoSuchKey)
        return None, f"❌ S3/Boto3 Error: Failed to load data from s3://{bucket_name}/{file_key}. Details: {e}"


def filter_data_context(df, query):
    df_working = df.copy()
    query_lower = query.lower()

    # Smart lead filtering
    key_phrases = ["best leads", "hot leads", "convertible", "potential", "possibility"]
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
            df_working = df.head(0)

    return df_working.to_csv(index=False, sep="\t")


def ask_gemini(question, data_context):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash") # Hardcoding the model as requested

        prompt = f"""
You are ZODOPT Sales Buddy. You strictly analyze ONLY the following tab-separated CRM lead data.
Do not guess or hallucinate any values outside the dataset.

--- DATASET ---
{data_context}

--- QUESTION ---
{question}

Provide structured bullet-point insights.
"""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
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
        st.session_state.chat = [
            {"role": "ai", "content": f"Hello! I've loaded {len(df_filtered)} leads from S3. Ask me anything about your CRM leads."}
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
