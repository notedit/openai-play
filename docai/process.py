

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


pdfs = [
    #"data/647_75992_cn.pdf"
]


docs = []
for pdf in pdfs:
    loader = PyPDFLoader(pdf)
    doc = loader.load()
    docs.append(doc)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)

chunked_docs = []
for doc in docs:
    texts = text_splitter.split_documents(doc)
    print(texts)
    chunked_docs.append(texts)
    print(f"chunked_docs length: {len(texts)}")


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

persist_directory = "./db"

vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)


for chunks in chunked_docs:
    print(f"chunks length: {len(chunks)} \n")
    vectordb.add_texts([chunk.page_content for chunk in chunks])
