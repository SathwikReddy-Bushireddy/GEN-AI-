from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('7.LangChain_Document_Loaders.py/dl-curriculum.pdf')
docs=loader.load()
print(docs)
print(len(docs))