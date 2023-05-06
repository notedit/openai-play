from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import openai


os.environ["OPENAI_API_BASE"] = "https://avchat-api.woa.com/v1"
os.environ["OPENAI_API_KEY"] = "leolxliu"

openai.debug = True
openai.log = "debug"

# OPENAI_API_BASE = os.environ.get('OPENAI_API_BASE')
# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# loader = TextLoader('imall.txt', encoding='utf-8')


loader = PyPDFLoader('data/647_41102_cn.pdf')

documents = loader.load()
# print(documents)
text_splitter = CharacterTextSplitter(
    separator='\n', chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'imdb2'

embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=docs, embedding=embedding, persist_directory=persist_directory)
vectordb.persist()
vectordb = None

# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)


query = "IM的数据中心有哪些"
docs = vectordb.similarity_search(query, k=5)
print(docs[0].page_content)
