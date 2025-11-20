import streamlit as st

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="ZodOpt",
    page_icon="⚡",
    layout="centered"
)

# --------------------------
# USER DATABASE (Dummy)
# Replace with DB later
# --------------------------
USERS = {
    "admin": "admin123",
    "siddarth": "pass123",
}

# --------------------------
# Login Function
# --------------------------
def login_page():

    st.title("⚡ ZodOpt Login")
    st.write("Welcome to the ZodOpt Platform. Please log in to continue.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

# --------------------------
# Home Page
# --------------------------
def home_page():
    st.title("🚀 ZodOpt Platform")
    st.write(f"Welcome, **{st.session_state['username']}!**")

    st.markdown("""
    ### 🔍 What is ZodOpt?
    ZodOpt is an intelligent optimization & automation platform powered by AI.

    ### ✨ Features
    - Real-time insights  
    - Automation workflows  
    - Powerful AI tools  
    - Developer-friendly APIs  
    """)

    if st.button("Logout"):
        st.session_state.clear()
        st.experimental_rerun()

# --------------------------
# Main App Logic
# --------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    home_page()
else:
    login_page()

