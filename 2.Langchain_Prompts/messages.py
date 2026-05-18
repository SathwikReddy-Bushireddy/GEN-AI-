from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.5",
    task="text-generation",

)
model = ChatHuggingFace(llm=llm)
messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='tell me about langchain')
]
res=model.invoke(messages)
messages.append(AIMessage(content=res.content))

print(messages)