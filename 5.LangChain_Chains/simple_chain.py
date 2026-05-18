from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

hf_llm  = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.5",
    task="text-generation",
    temperature=0
)
model = ChatHuggingFace(llm=hf_llm)

prompt = PromptTemplate(
    template="Generate 5 interesting facrts about {topic}",
    input_variables=['topic']
)

# prompt=template.invoke({'topic':'hyderabad'})
parser=StrOutputParser()
chain=prompt | model | parser
res=chain.invoke({'topic':'hyderabad'})
print(res)

# visualize chain
chain.get_graph().print_ascii()