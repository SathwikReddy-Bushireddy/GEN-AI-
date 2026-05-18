from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model='gpt-4',temperature=0,max_completion_tokens=10) 
#temperature defines the creativity level vaires from 0-2, it affects the randomness of the model
#max_completion_tokens it gives how many tokens can be used (tokens==words)
result = model.invoke("Write a 5 line poem on India")
print(result.content)