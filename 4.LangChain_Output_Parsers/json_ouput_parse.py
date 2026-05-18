from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

hf_llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.5",
    task="text-generation",
    temperature=0
)
model=ChatHuggingFace(llm=hf_llm)

parser=JsonOutputParser()
template=PromptTemplate(
    template="Give me the name,age,city of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt=template.format()
# res=model.invoke(prompt)
# res1=parser.parse(res.content)
# print(res1)
# print(type(res1))


# using chains
chain = template | model | parser
res=chain.invoke({})
print(res)