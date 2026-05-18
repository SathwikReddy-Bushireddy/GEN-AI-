from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

load_dotenv()

hf_llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.5",
    task="text-generation",
    temperature=0
)
model=ChatHuggingFace(llm=hf_llm)

class Person(BaseModel):
    name:str=Field(description='Name of the person')
    age:int=Field(gt=18, description='Age of the person')
    city:str=Field(description='Name of the city the person belongs to')

parser=PydanticOutputParser(pydantic_object=Person)
template=PromptTemplate(
    template='Generate name,age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

promt=template.invoke({'place':'indian'})
res=model.invoke(promt)
res1=parser.parse(res.content)
print(res1)