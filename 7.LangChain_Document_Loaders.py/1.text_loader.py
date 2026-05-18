from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

hf_llm  = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    temperature=0
)
model = ChatHuggingFace(llm=hf_llm)
parser=StrOutputParser()

prompt = PromptTemplate(
    template='Write a summary about the following poem \n {poem}',
    input_variables=['poem']
)


loader = TextLoader('7.LangChain_Document_Loaders.py/Cricket.txt',encoding='utf-8')
doc=loader.load()
print(doc)
print(type(doc))
print(len(doc))
print(doc[0])
print(doc[0].page_content)
print(doc[0].metadata)

chain = prompt | model | parser
print(chain.invoke({'poem':doc[0].page_content}))