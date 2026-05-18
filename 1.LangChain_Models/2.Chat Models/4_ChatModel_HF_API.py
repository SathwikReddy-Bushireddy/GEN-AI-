from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
# import os

load_dotenv()
# print(os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
llm=HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.5", # which model is used
    task="text-generation", # which task is being performed
    max_new_tokens=100, # max no.of words to be used
    temperature=0.7
)
model=ChatHuggingFace(llm=llm)
result = model.invoke("What is the team of Hyderabad city in IPL")
print(result.content)
