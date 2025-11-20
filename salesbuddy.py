import streamlit as st
import pandas as pd
import os
import google.generativeai as genai

# ---------------------- CONFIG -----------------------------
GEMINI_MODEL = "gemini-2.5-flash"

# Paths for Excel, background and logo
SALES_DATA_PATH = "salesbuddy.xlsx"
BACKGROUND_IMAGE_PATH = "background.jpg"
LOGO_IMAGE_PATH = "zodopt.png"

GEMINI_API_KEY = "AIzaSyBgKTlULVARw37Ec0WCor0YFC3cHXq64Mc"

REQUIRED_COLS = [
    "Record Id", "Full Name", "Lead Source", "Company", "Lead Owner",
    "Street", "City", "State", "Country", "Zip Code",
    "First Name", "Last Name", "Annual Revenue", "Lead Status" 
]

DISQUALIFYING_STATUSES = ["Disqualified", "Closed - Lost", "Junk Lead"]

# ---------------------- FUNCTIONS --------------------------

@st.cache_data(ttl=600)
def load_sales_data(file_path, required_cols):
    if not os.path.exists(file_path):
        return None, f"❌ File not found at: {file_path}"
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"❌ Missing essential columns: {', '.join(missing)}" 
        df_filtered = df[required_cols]
        df_filtered['Annual Revenue'] = pd.to_numeric(df_filtered['Annual Revenue'], errors='coerce')
        return df_filtered, "Sales data loaded successfully! Ready for AI analysis."
    except Exception as e:
        return None, f"❌ Error reading Excel: {e}"

def filter_data_context(df, query):
    df_working = df.copy()
    query_lower = query.lower()
    
    potential_phrases = ["possibility to be turned into sales", "best leads", "potential sales", "hot leads", "convertible", "most valuable"]
    if any(phrase in query_lower for phrase in potential_phrases):
        df_working = df_working[~df_working['Lead Status'].isin(DISQUALIFYING_STATUSES)]
        st.caption("Status filter applied: Excluding permanently disqualified leads.")
        
    locations = ["bangalore", "bengaluru", "new york", "london", "california", "india", "texas", "washington", "oregon", "canada"]
    location_match = next((loc for loc in locations if loc in query_lower), None)

    if location_match:
        location_mask = (
            df_working['City'].astype(str).str.lower().str.contains(location_match, na=False) |
            df_working['State'].astype(str).str.lower().str.contains(location_match, na=False) |
            df_working['Country'].astype(str).str.lower().str.contains(location_match, na=False)
        )
        df_working = df_working[location_mask]
        
        if not df_working.empty:
            st.caption(f"Location filter applied: Analyzing **{len(df_working)}** leads matching **'{location_match.title()}'**.")
        else:
            st.warning(f"No leads found for '{location_match.title()}' after status filtering.")
            df_working = df.head(0)
            
    return df_working.to_csv(index=False, sep="\t")

def get_training_examples():
    return """ 
--- TRAINING EXAMPLES (RESTRICTED COLUMNS) ---
# Include your 30 examples here
--- END TRAINING EXAMPLES ---
"""

def ask_gemini(question, data_context):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        available_cols = ", ".join(REQUIRED_COLS)
        training_examples = get_training_examples()
        
        system_prompt = (
            "You are 'ZODOPT Sales Buddy', an expert CRM & sales analyst. "
            "Use ONLY the provided dataset to answer questions. "
            f"The available columns are: {available_cols}. "
            "Refer to the TRAINING EXAMPLES above for analysis and output formatting guidance. "
            "Do not invent data. Provide concise, accurate answers, formatted as structured bullet points or lists. "
            "If the dataset is empty, state clearly 'No relevant leads found in the current context.'"
        )

        full_query = f"""
{system_prompt}

{training_examples}

--- SALES DATA (Tab-Separated Context) ---
{data_context}

--- USER QUESTION ---
{question}
"""
        response = model.generate_content(full_query)
        return response.text
    except Exception as e:
        return f"❌ Gemini API Error: {e}"

# ---------------------- STREAMLIT UI ------------------------

def set_background_and_logo():
    if os.path.exists(BACKGROUND_IMAGE_PATH):
        with open(BACKGROUND_IMAGE_PATH, "rb") as f:
            bg_data = f.read()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/jpg;base64,{bg_data.encode("base64")});
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    if os.path.exists(LOGO_IMAGE_PATH):
        st.image(LOGO_IMAGE_PATH, width=120, use_column_width=False, output_format="PNG")

def main():
    st.set_page_config(page_title="ZODOPT Sales Buddy", layout="wide")
    
    # Custom CSS for header layout
    st.markdown(
        """
        <style>
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Header with title and logo on right
    logo_col, title_col = st.columns([6,1])
    with title_col:
        if os.path.exists(LOGO_IMAGE_PATH):
            st.image(LOGO_IMAGE_PATH, width=100)
    with logo_col:
        st.title("💰 ZODOPT Sales Buddy Agent")
        st.subheader("AI-Powered CRM & Sales Insights")

    st.divider()

    df_filtered, load_message = load_sales_data(SALES_DATA_PATH, REQUIRED_COLS)
    if df_filtered is None:
        st.error(load_message)
        st.stop()
    else:
        st.success(load_message)

    st.divider()
    st.write("### 💬 Chat with Sales Buddy")

    if "chat" not in st.session_state:
        st.session_state.chat = [
            {"role": "ai", "content": "Hello! I'm ZODOPT Sales Buddy. Ask me about Lead Source, Annual Revenue, Lead Owner, Lead Status, or Location."}
        ]

    for msg in st.session_state.chat:
        if msg["role"] == "user":
            st.markdown(f"<div style='background:#E6F7FF;padding:10px;border-radius:8px;text-align:right'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#FFFFFF;padding:10px;border-radius:8px;border-left:4px solid #32CD32'>{msg['content']}</div>", unsafe_allow_html=True)

    user_input = st.chat_input("Ask something about your CRM leads...")

    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})
        with st.spinner("Analyzing leads..."):
            context_text = filter_data_context(df_filtered, user_input)
            response = ask_gemini(user_input, context_text)
        st.session_state.chat.append({"role": "ai", "content": response})
        st.rerun()

if __name__ == "__main__":
    main()
