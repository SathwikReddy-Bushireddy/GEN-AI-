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

prompt1=PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template="Generate a 5 pointer summary on the following text \n {text}",
    input_variables=['text']
)

parser=StrOutputParser()
chain = prompt1 | model | parser | prompt2 | model | parser
res = chain.invoke({'topic':'unemployment in india'})
print(res)

# visualize chain
chain.get_graph().print_ascii()