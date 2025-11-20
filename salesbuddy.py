import streamlit as st
import pandas as pd
import os
import google.generativeai as genai

# ---------------------- CONFIG -----------------------------

GEMINI_MODEL = "gemini-2.5-flash"
# Ensure this path is correct for your system
SALES_DATA_PATH = r"C:\Users\DELL\Desktop\salesbuddy.xlsx" 

# ⚠️ REPLACE WITH YOUR REAL KEY
# Note: For production, store this securely (e.g., Streamlit Secrets)
GEMINI_API_KEY = "AIzaSyBgKTlULVARw37Ec0WCor0YFC3cHXq64Mc" 

# Define the columns that are essential for analysis (ONLY AVAILABLE COLUMNS).
REQUIRED_COLS = [
    "Record Id", "Full Name", "Lead Source", "Company", "Lead Owner",
    "Street", "City", "State", "Country", "Zip Code",
    "First Name", "Last Name", "Annual Revenue", "Lead Status" 
]

# Define statuses that represent leads that CANNOT be converted (must be filtered out)
DISQUALIFYING_STATUSES = ["Disqualified", "Closed - Lost", "Junk Lead"]


# ---------------------- FUNCTIONS --------------------------

@st.cache_data(ttl=600) # Cache the data load for 10 minutes
def load_sales_data(file_path, required_cols):
    """
    Reads the Excel file and filters it to only the required columns,
    and performs necessary data cleaning.
    """
    if not os.path.exists(file_path):
        return None, f"❌ File not found at: {file_path}"

    try:
        df = pd.read_excel(file_path)

        # 1. Standardize and Clean Column Names
        df.columns = df.columns.str.strip() 

        # 2. Check for missing required columns
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"❌ Missing essential columns: {', '.join(missing)}" 

        # 3. Filter the DataFrame to ONLY the required columns.
        df_filtered = df[required_cols]

        # 4. Data Type Conversion and Cleaning (CRITICAL for analysis)
        # Convert Annual Revenue to numeric (handling errors)
        df_filtered['Annual Revenue'] = pd.to_numeric(df_filtered['Annual Revenue'], errors='coerce')

        return df_filtered, "Sales data loaded successfully! Ready for AI analysis."

    except Exception as e:
        return None, f"❌ Error reading Excel: {e}"


def filter_data_context(df, query):
    """
    Implements business logic to filter data based on query intent 
    (sales potential status) and any specified location.
    """
    df_working = df.copy()
    query_lower = query.lower()
    
    # --- 1. Identify Sales Potential Filter (Status Check) ---
    potential_phrases = ["possibility to be turned into sales", "best leads", "potential sales", "hot leads", "convertible", "most valuable"]
    should_filter_status = any(phrase in query_lower for phrase in potential_phrases)

    if should_filter_status and "Lead Status" in df_working.columns:
        # Filter out rows where 'Lead Status' is in the disqualifying list
        df_working = df_working[~df_working['Lead Status'].isin(DISQUALIFYING_STATUSES)]
        st.caption("Status filter applied: Excluding permanently disqualified leads.")
        
    # --- 2. Identify Location Filter (Location Check) ---
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
            df_working = df.head(0) # Return empty DF context
            
    df_context = df_working
    
    # Build LLM-context (using the highly filtered data)
    context = df_context.to_csv(index=False, sep="\t")
    return context


def get_training_examples():
    """
    Returns the comprehensive list of few-shot training examples,
    restricted to the 14 available columns.
    (Total: 30 Examples)
    """
    return """
--- TRAINING EXAMPLES (RESTRICTED COLUMNS) ---

# Core Performance & Aggregation
Example 1:
User Question: What are the top 3 Lead Sources by the number of leads?
Expected Response Structure:
- Lead Source A: 50 leads
- Lead Source B: 35 leads
- Lead Source C: 20 leads

Example 2:
User Question: Who is the Lead Owner with the highest total Annual Revenue?
Expected Response Structure:
- Lead Owner: [Name of Owner]
- Total Annual Revenue: [Calculated Revenue Sum]

Example 3 (Filtering & Listing):
User Question: List all leads in 'California' that have a Lead Status of 'Contacted'.
Expected Response Structure:
- Full Name: Alice Johnson (Company: XYZ Corp)
- Full Name: Bob Smith (Company: ABC Inc)

Example 4 (Using Status & Location Filtering):
User Question: What is the average Annual Revenue for convertible leads in 'New York'?
Expected Response Structure:
- Total Convertible Leads in New York: [Number]
- Average Annual Revenue: [Calculated Average Value]

Example 5 (Lead Source Performance):
User Question: What is the total Annual Revenue generated by the 'Web' Lead Source compared to 'Phone Inquiry'?
Expected Response Structure:
- Web Lead Revenue: [Calculated Sum]
- Phone Inquiry Revenue: [Calculated Sum]
- Difference: Web source generated [X] more revenue. 

Example 6 (Data Completeness Check):
User Question: List the leads that are missing a value in the 'Annual Revenue' column.
Expected Response Structure:
- Lead 1: [Full Name], Company: [Company]
- Lead 2: [Full Name], Company: [Company]

Example 7 (Conversion Rate Query):
User Question: What is the conversion rate (Contacted to total leads) for the 'Alice Johnson' Lead Owner?
Expected Response Structure:
- Total Leads Handled: [Total Count]
- Leads Contacted: [Contacted Count]
- Conversion Rate: [Calculated Percentage]%

Example 8 (Simple Aggregation):
User Question: What is the total number of leads in the state of 'Texas'?
Expected Response Structure:
- State: Texas
- Total Leads: [Count]

Example 9 (Owner Performance):
User Question: List the top 5 Lead Owners based on the number of 'Converted' leads.
Expected Response Structure:
- Owner 1: [Name], Converted Leads: [Count]
- Owner 2: [Name], Converted Leads: [Count]
- Owner 3: [Name], Converted Leads: [Count]
- Owner 4: [Name], Converted Leads: [Count]
- Owner 5: [Name], Converted Leads: [Count]

Example 10 (Filtering by Revenue Range):
User Question: List all leads with an Annual Revenue greater than $500,000.
Expected Response Structure:
- Full Name: Chris Lee (Revenue: $[Value])
- Full Name: Dana Fox (Revenue: $[Value])

Example 11 (Location Aggregation):
User Question: What is the distribution of Lead Statuses in the 'City' of Boston?
Expected Response Structure:
- New: [Count] leads
- Contacted: [Count] leads
- Converted: [Count] leads
- Unqualified: [Count] leads

Example 12 (Lead Source Performance by Revenue):
User Question: Which Lead Source has the highest average Annual Revenue?
Expected Response Structure:
- Highest Average Revenue Source: [Source Name]
- Average Annual Revenue: [Calculated Average]

Example 13 (Multiple Filtering Criteria):
User Question: List the convertible leads in 'Florida' with an Annual Revenue greater than $100,000.
Expected Response Structure:
- Name: Eve Green (Revenue: $[Count])
- Name: Frank Hill (Revenue: $[Count])

Example 14 (Average Revenue by Lead Status):
User Question: What is the average Annual Revenue for leads with a 'Contacted' Lead Status?
Expected Response Structure:
- Status: Contacted
- Average Annual Revenue: [Calculated Average]

Example 15 (Owner Conversion Rate):
User Question: What is the conversion rate (Converted to total leads) for the 'Jessica Alba' Lead Owner?
Expected Response Structure:
- Total Leads Handled: [Total Count]
- Leads Converted: [Converted Count]
- Conversion Rate: [Calculated Percentage]%

Example 16 (Top Company by Revenue):
User Question: Which Company has the highest total Annual Revenue?
Expected Response Structure:
- Top Company: [Company Name]
- Total Revenue: [Calculated Sum]

Example 17 (Filtering by Status and Owner):
User Question: List all leads belonging to 'Robert Downey' that have a Lead Status of 'New'.
Expected Response Structure:
- Lead Name: Tony Stark (Company: SI)
- Lead Name: Pepper Potts (Company: SI)

Example 18 (Trend Comparison - Revenue by Source):
User Question: Compare the total Annual Revenue of leads from 'Partner' source versus 'Referral' source.
Expected Response Structure:
- Partner Revenue: [Sum]
- Referral Revenue: [Sum]
- Result: Partner source generated [X] more/less revenue.

Example 19 (Listing Leads with Specific Status):
User Question: List the Full Name, Company, and Annual Revenue of all leads with a status of 'Converted'.
Expected Response Structure:
- Full Name: Clark Kent (Company: Daily Planet, Revenue: [Value])
- Full Name: Bruce Wayne (Company: Wayne Enterprises, Revenue: [Value])

Example 20 (Overall Conversion Rate):
User Question: What is the overall conversion rate (Converted to total leads) for the entire dataset?
Expected Response Structure:
- Total Leads: [Total Count]
- Total Converted: [Converted Count]
- Overall Conversion Rate: [Calculated Percentage]%

# NEW FOCUSED EXAMPLES (21-30)

Example 21:
User Question: List the top 3 Lead Owners by the number of leads in a 'New' status.
Expected Response Structure:
- Owner A: [Count] new leads
- Owner B: [Count] new leads
- Owner C: [Count] new leads

Example 22:
User Question: What percentage of total leads are currently in the 'Contacted' status?
Expected Response Structure:
- Total Leads: [Total Count]
- Leads Contacted: [Contacted Count]
- Percentage Contacted: [Calculated Percentage]%

Example 23:
User Question: Which Lead Owner has the lowest average Annual Revenue from their converted leads?
Expected Response Structure:
- Lowest Average Revenue Owner: [Name]
- Average Converted Revenue: $[Calculated Average]

Example 24:
User Question: List the Full Name and Lead Status for all leads from 'Apple Inc.'.
Expected Response Structure:
- Full Name: Tim Cook (Status: [Status])
- Full Name: Jeff Williams (Status: [Status])

Example 25:
User Question: Calculate the total Annual Revenue for all leads in 'New York' that have a status of 'New'.
Expected Response Structure:
- Total New Leads in NY: [Count]
- Total Annual Revenue: $[Calculated Sum]

Example 26:
User Question: Compare the total lead volume owned by 'John Smith' versus 'Jane Doe'.
Expected Response Structure:
- John Smith Leads: [Count]
- Jane Doe Leads: [Count]
- Result: [Name] owns [X] more/fewer leads.

Example 27:
User Question: List the leads with the top 5 highest Annual Revenue values.
Expected Response Structure:
- Lead 1: [Full Name] ($ [Value])
- Lead 2: [Full Name] ($ [Value])
- Lead 3: [Full Name] ($ [Value])
- Lead 4: [Full Name] ($ [Value])
- Lead 5: [Full Name] ($ [Value])

Example 28:
User Question: Which Lead Source has the highest lead volume AND a conversion rate (Converted to total leads) above 50%?
Expected Response Structure:
- Top Source: [Source Name]
- Total Volume: [Count]
- Conversion Rate: [Calculated Percentage]%

Example 29:
User Question: List the states where the average Annual Revenue is greater than $250,000.
Expected Response Structure:
- State A: $[Average Revenue]
- State B: $[Average Revenue]

Example 30:
User Question: List all leads in 'Washington' that are still in the 'New' or 'Contacted' status.
Expected Response Structure:
- Full Name: Gary King (Status: New)
- Full Name: Simon Pegg (Status: Contacted)
--- END TRAINING EXAMPLES ---
"""


def ask_gemini(question, data_context):
    """
    Sends query + filtered dataset to Gemini with the comprehensive few-shot examples.
    """
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)

        model = genai.GenerativeModel(GEMINI_MODEL)

        available_cols = ", ".join(REQUIRED_COLS)
        
        # Integrate the restricted training examples
        training_examples = get_training_examples()
        
        system_prompt = (
            "You are 'ZODOPT Sales Buddy', an expert CRM & sales analyst. "
            "Use ONLY the provided dataset (which may be pre-filtered for relevancy) to answer questions. "
            f"The available columns are: {available_cols}. "
            "Refer to the TRAINING EXAMPLES above for analysis and output formatting guidance. "
            "Do not invent data. **Provide concise, accurate answers, formatted STRICTLY as structured bullet points or lists.** "
            "If the dataset is empty, state clearly 'No relevant leads found in the current context.'"
        )

        # Construct the final prompt payload
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

def main():
    st.set_page_config(page_title="ZODOPT Sales Buddy", layout="wide")

    st.title("💰 ZODOPT Sales Buddy Agent")
    st.subheader("AI-Powered CRM & Sales Insights")

    st.divider()

    # Load sales data
    df_filtered, load_message = load_sales_data(SALES_DATA_PATH, REQUIRED_COLS)

    if df_filtered is None:
        st.error(load_message)
        st.stop()
    else:
        st.success(load_message)

    st.divider()
    st.write("### 💬 Chat with Sales Buddy")

    # Chat History
    if "chat" not in st.session_state:
        st.session_state.chat = [
            {"role": "ai", "content": "Hello! I'm ZODOPT Sales Buddy. I am now configured with 30 focused examples for your available columns. Ask me questions about **Lead Source, Annual Revenue, Lead Owner, Lead Status, or Location**."}
        ]

    # Display messages
    for msg in st.session_state.chat:
        if msg["role"] == "user":
            # User message styling (light blue background, right-aligned)
            st.markdown(
                f"<div style='background:#E6F7FF;padding:10px;border-radius:8px;text-align:right'>{msg['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            # AI message styling (white background, green left border)
            st.markdown(
                f"<div style='background:#FFFFFF;padding:10px;border-radius:8px;border-left:4px solid #32CD32'>{msg['content']}</div>",
                unsafe_allow_html=True
            )

    # Chat Input
    user_input = st.chat_input("Ask something about your CRM leads...")

    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})

        with st.spinner("Analyzing leads..."):
            
            # Conditionally filter the DataFrame context based on the user's question.
            context_text = filter_data_context(df_filtered, user_input)
            
            response = ask_gemini(user_input, context_text)

        st.session_state.chat.append({"role": "ai", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()
