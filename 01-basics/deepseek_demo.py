from langchain_community.chat_models import ChatOllama

llm=ChatOllama(model="deepseek-r1:1.5b")

question = input("Enter the question")
response = llm.invoke(question)
print(response.content)