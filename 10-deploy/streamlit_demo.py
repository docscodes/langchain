from langchain_openai import ChatOpenAI
import streamlit as st
from langchain.globals import set_debug

st.title("Ask Anything")

with st.sidebar:
    st.title("Provide Your API Key")
    OPENAI_API_KEY = st.text_input(
        "Enter your OpenAI API key", type="password")

if not OPENAI_API_KEY:
    st.info("Please provide your OpenAI API key")
    st.stop()

llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

question = st.text_input("Enter the question:")

if question:
    response = llm.invoke(question)
    st.write(response.content)
