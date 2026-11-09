"""Web UI with two tabs: Stock analysis and Stock screener.

Run:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Indian Share Tips", page_icon="📈", layout="wide")
st.navigation([st.Page("page_analysis.py", title="Stock analysis", icon="📈", default=True),
               st.Page("page_screener.py", title="Stock screener", icon="🔎")], position="top").run()
