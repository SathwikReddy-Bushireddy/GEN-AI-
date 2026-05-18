from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='7.LangChain_Document_Loaders.py/Social_Network_Ads.csv')
data=loader.load()
print(data[0])