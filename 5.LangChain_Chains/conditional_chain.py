from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
# RunnableParallel-> hepls in executing multiple chains parallely
# RunnableBranch-> helps in executing chains by if else
# RunnableLambda-> it converts a lambda function into runnable , then we can use that function as a chain
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

hf_llm  = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    temperature=0
)
model = ChatHuggingFace(llm=hf_llm)
parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal['positive','Negative'] = Field(description="Give the sentiment of the feedback")

parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2=PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)
prompt3=PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2 | model | parser),
    (lambda x:x.sentiment=='negative',prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not find the sentiment")
)

chain=classifier_chain | branch_chain

res = chain.invoke({'feedback':'This car gives a wonderful driving experience'})
print(res)

# visualize the chain
chain.get_graph().print_ascii()