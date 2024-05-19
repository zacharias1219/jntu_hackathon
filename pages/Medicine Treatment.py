import streamlit as st
from utilities.icon import page_icon

st.set_page_config(
    page_title="Medicine Treatment",
    layout="wide",
    page_icon="💊",
    initial_sidebar_state="collapsed",
)

st.title("💊 Medicine Treatment")