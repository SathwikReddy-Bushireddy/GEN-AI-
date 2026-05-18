from langchain_community.document_loaders import WebBaseLoader
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
    template='Answer the following question \n {question} from the following text \n {text}',
    input_variables=['question','text']
)

url='https://www.geeksforgeeks.org/artificial-intelligence/what-is-generative-ai/'
loader = WebBaseLoader(url)
docs=loader.load()

chain=prompt | model | parser
ans = chain.invoke({'question':'what are the applications of GEN AI ','text':docs[0]})
print(ans)