from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

hf_llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.5",
    task="text-generation",
    temperature=0
)
model=ChatHuggingFace(llm=hf_llm)

# 1st-prompt -> detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

# 2nd-report -> summary
template2=PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'black hole'})
res1=model.invoke(prompt1) 

prompt2=template2.invoke({'text':res1.content})
res2=model.invoke(prompt2)

parser=StrOutputParser()
chain = template1 | model | parser | template2 | model | parser 
res=chain.invoke({'topic':'black hole'})
print(res)