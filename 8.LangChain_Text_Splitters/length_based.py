from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# text="""At the core of human advancement lies an insatiable desire to understand the unknown. Throughout history, every leap in civilization—from the mastery of fire to the exploration of distant galaxies—has been driven by our innate curiosity and the courage to ask fundamental questions. We are pattern-seeking creatures, inherently wired to find meaning in a chaotic universe. This drive pushes us to look beyond our immediate surroundings, challenge established paradigms, and build upon the collective wisdom of those who came before us."""
# splitter=CharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=0,
#     separator=''
# )
# result = splitter.split_text(text)

loader = PyPDFLoader('dl-curriculum.pdf')
docs=loader.load()
splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)
result =splitter.split_documents(docs)
print(result[0])