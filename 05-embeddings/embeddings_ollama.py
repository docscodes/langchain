import os
from langchain_ollama import OllamaEmbeddings

llm=OllamaEmbeddings(model="llama3.2:3b")

text = input("Enter the text")
response = llm.embed_query(text)
print(response)