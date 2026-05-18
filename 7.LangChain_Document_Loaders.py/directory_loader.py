from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader=DirectoryLoader(
    path='7.LangChain_Document_Loaders.py/books',
    glob="*.pdf",
    loader_cls=PyPDFLoader 
)
docs = loader.load()
print(docs[0].page_content)
