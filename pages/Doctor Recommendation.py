import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import time
import bs4
from utilities.icon import page_icon
from dotenv import load_dotenv

st.set_page_config(
    page_title="Doctor Recommendation",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="collapsed",
)

load_dotenv()  #

groq_api_key = os.environ['GROQ_API_KEY']


if "vector" not in st.session_state:

    st.session_state.embeddings = OllamaEmbeddings()

    st.session_state.loader=WebBaseLoader(web_paths=("https://www.practo.com/hyderabad/treatment-for-rheumatic-arthritis",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("u-border-general--bottom")

                     )))
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.documents = st.session_state.text_splitter.split_documents( st.session_state.docs)
    st.session_state.vector = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)

st.title("Find doctors")

llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name='mixtral-8x7b-32768'
    )

prompt = ChatPromptTemplate.from_template("""
You are a doctor recommender and you need to recommend doctors to people suffering with Rheumatoid Arthritis from NIMS Hospital.
Refer from the webpage and recommend the correct doctor to the patient based on their condition, symptoms and experience. 
I will tip you $200 if the user finds the answer helpful. 
<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input your prompt here")


# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print(f"Response time: {time.process_time() - start}")

    st.write(response["answer"])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            # print(doc)
            # st.write(f"Source Document # {i+1} : {doc.metadata['source'].split('/')[-1]}")
            st.write(doc.page_content)
            st.write("--------------------------------")