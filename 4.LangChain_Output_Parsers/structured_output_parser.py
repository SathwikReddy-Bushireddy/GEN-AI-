from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser,ResponseSchemac # StructuredOutputParser might be removed from langchain
load_dotenv()

hf_llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.5",
    task="text-generation",
    temperature=0
)
model=ChatHuggingFace(llm=hf_llm)

schema=[
    ResponseSchema(name='fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='Fact 3 about the topic')
]
parser = StructuredOutputParser.from_response_schemas(schema)
template=PromptTemplate(
    template='Give 3 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt=template.invoke({'topic':'black hole'})
res=model.invoke(prompt)
res1 = parser.parse(res.content)
print(res1)