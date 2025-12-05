import streamlit as st
import pandas as pd
import time
import boto3
import json
import random
from io import BytesIO
import google.generativeai as genai


# ================== CONFIG =====================
S3_BUCKET_NAME = "zodopt"
S3_FILE_KEY = "Leaddata/Leads by Status.xlsx"
AWS_REGION = "ap-south-1"
GEMINI_SECRET_NAME = "salesbuddy/secrets"
GEMINI_SECRET_KEY = "GEMINI_API_KEY"
GEMINI_MODEL = "gemini-2.5-flash"

REQUIRED_COLS = [
    "Record Id", "Full Name", "Lead Source", "Company", "Lead Owner",
    "Street", "City", "State", "Country", "Zip Code",
    "First Name", "Last Name", "Annual Revenue", "Lead Status"
]

ACTION_CHIPS = [
    "Qualification",
    "Needs Analysis",
    "Proposal/Price Quote",
    "Negotiation/Review",
    "Closed Won",
    "Closed Lost"
]


# ================== STYLING =====================

CSS = """
<style>

* { font-family:"Inter", sans-serif; }

.stApp {
    background:#f7f8fa;
}

.block-container {
    padding:0 !important;
    max-width:900px;
}

/* ======= GRADIENT HEADER ======= */
.gradient-header {
    margin:20px auto 10px auto;
    width:100%;
    background: linear-gradient(90deg, #0066ff 0%, #7b00ff 100%);
    border-radius:22px;
    padding:26px 34px;
    display:flex;
    justify-content:space-between;
    align-items:center;
    color:white;
}

.header-title {
    font-size:34px;
    font-weight:700;
}

.logo-text {
    font-size:30px;
    font-weight:700;
}

.logo-text span:nth-child(1){ color:#ff3d0a; }
.logo-text span:nth-child(2){ color:#00c48c; }

/* ======= API Credits ======= */
.credits-line {
    font-size:13px;
    color:#666;
    margin-left:4px;
}

/* ======= Lead Header ======= */
.lead-section {
    background:white;
    border-radius:12px;
    padding:18px 22px;
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-top:12px;
    border:1px solid #eee;
}

.lead-left { display:flex; gap:12px; align-items:center; }

.lead-avatar {
    width:44px;
    height:44px;
    font-weight:700;
    border-radius:50%;
    background:#8647e8;
    color:white;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:20px;
}

.lead-name {
    font-size:18px;
    font-weight:700;
}

.lead-score {
    font-size:12px;
    color:#777;
}

/* ======= Chips ======= */
.chip-bar {
    margin-top:12px;
    display:flex;
    gap:10px;
    overflow-x:auto;
    scrollbar-width:none;
}

.chip-bar::-webkit-scrollbar { display:none; }

.chip-btn {
    font-size:13px;
    padding:7px 18px;
    background:#f1f1f1;
    border-radius:18px;
    border:1px solid #ddd;
    cursor:pointer;
    white-space:nowrap;
}

.chip-active {
    background:#efe5ff;
    color:#8035ff;
    border:1px solid #8035ff;
}

/* ======= Chat ======= */
.chat-container {
    padding: 18px 6px 90px 6px;
}

.msg-user {
    background:#18a05c;
    color:white;
    padding:10px 16px;
    border-radius:18px 18px 0 18px;
    max-width:65%;
    margin-left:auto;
    margin-bottom:14px;
}

.msg-ai {
    background:white;
    border:1px solid #eee;
    padding:10px 16px;
    border-radius:18px 18px 18px 0;
    max-width:65%;
    margin-bottom:14px;
}

.time-u {
    font-size:10px;
    text-align:right;
    margin-top:4px;
    opacity:0.8;
}

.time-ai {
    font-size:10px;
    color:#777;
    margin-top:4px;
}

/* ======= Input Bar ======= */
.input-bar {
    position:fixed;
    bottom:0;
    left:0;
    right:0;
    max-width:900px;
    margin:auto;
    background:white;
    border-top:1px solid #eee;
    padding:12px 14px;
}

.input-row {
    display:flex;
    gap:10px;
    align-items:center;
}

.send-btn {
    background:#18a05c;
    color:white;
    height:42px;
    width:42px;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    cursor:pointer;
    font-size:18px;
}

</style>
"""


# ================== HELPERS =====================

def get_remaining_api_credits():
    return random.randint(2500, 5000)

@st.cache_resource
def get_secret():
    try:
        client = boto3.client('secretsmanager', region_name=AWS_REGION)
        val = client.get_secret_value(SecretId=GEMINI_SECRET_NAME)
        return json.loads(val['SecretString'])[GEMINI_SECRET_KEY]
    except:
        return None

@st.cache_data
def load_data():
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=S3_FILE_KEY)
    df = pd.read_excel(BytesIO(obj['Body'].read()))
    df.columns = df.columns.str.strip()
    return df[REQUIRED_COLS]


def ask_gemini(query, key):
    genai.configure(api_key=key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    r = model.generate_content(query)
    return r.text


# ================== APP RENDER =====================

def render(user_data):
    st.markdown(CSS, unsafe_allow_html=True)

    api_key = get_secret()
    df = load_data()
    credits = get_remaining_api_credits()

    if "chat" not in st.session_state:
        st.session_state.chat = []

    if "lead" not in st.session_state:
        st.session_state.lead = {
            "name":"Acme Corporation",
            "score":"0%",
            "status":"Qualification"
        }

    lead = st.session_state.lead


    # ========= Gradient Header ==========
    st.markdown(f"""
    <div class="gradient-header">
        <div class="header-title">SalesBuddy</div>
        <div class="logo-text"><span>zod</span><span>opt</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='credits-line'>Total API Credits Left: {credits:,}</div>", unsafe_allow_html=True)


    # ========= Lead Section ==========
    st.markdown(f"""
    <div class="lead-section">
        <div class="lead-left">
            <div class="lead-avatar">A</div>
            <div>
                <div class="lead-name">{lead['name']}</div>
                <div class="lead-score">Score: {lead['score']}</div>
            </div>
        </div>
        <div style="font-size:22px;color:#666;">⋮</div>
    </div>
    """, unsafe_allow_html=True)


    # ========= Status Tabs ==========
    st.markdown("<div class='chip-bar'>", unsafe_allow_html=True)
    for chip in ACTION_CHIPS:
        active = "chip-active" if chip == lead['status'] else ""
        if st.button(chip, key=chip):
            st.session_state.lead['status'] = chip
        st.markdown(
            f"<style>[key='{chip}'] button{{padding:7px 18px;border-radius:18px;}}[key='{chip}'] button:hover{{opacity:0.85;}}</style>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)


    # ========= Chat ==========
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state.chat:
        if msg["role"]=="user":
            st.markdown(
                f"<div class='msg-user'>{msg['content']}<div class='time-u'>{msg['timestamp']}</div></div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='msg-ai'>{msg['content']}<div class='time-ai'>{msg['timestamp']}</div></div>",
                unsafe_allow_html=True)

    # ========= Input ==========
    st.markdown("<div class='input-bar'>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([10,1])
        with col1:
            query = st.text_input("", placeholder="Type a message...")
        with col2:
            send = st.form_submit_button("▶")
        if send and query:
            st.session_state.chat.append({"role":"user","content":query,"timestamp":time.strftime("%I:%M %p")})
    st.markdown("</div>", unsafe_allow_html=True)
