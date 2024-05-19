import streamlit as st
from utilities.icon import page_icon

st.set_page_config(
    page_title="Remote Patient Monitoring",
    layout="wide",
    page_icon="🤒",
    initial_sidebar_state="collapsed",
)

from langchain_community.llms import Ollama 
import pandas as pd
from pandasai import SmartDataframe

llm = Ollama(model="mixtral")

st.title("Data Analysis with PandasAI")

uploader_file = st.file_uploader("Upload a CSV file", type= ["csv"])

if uploader_file is not None:
    data = pd.read_csv(uploader_file)
    st.write(data.head(3))
    df = SmartDataframe(data, config={"llm": llm})
    prompt = st.text_area("Enter your prompt:")

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                st.write(df.chat(prompt))
        else:
            st.warning("Please enter a prompt!")