from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()
hf_llm  = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    temperature=0
)
model = ChatHuggingFace(llm=hf_llm)
parser=StrOutputParser()

prompt=PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']   
)

chain=RunnableSequence(prompt,model,parser)
chain.invoke({'topic':'AI'})