

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import pinecone
import os
import pprint

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
index_name = "trtc-docs"

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vectorstore = Pinecone.from_existing_index(
    index_name=index_name, embedding=embeddings)


llm = OpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm)


query = "usersig 如何计算？"
docs = vectorstore.similarity_search(query, include_metadata=True, k=2)


pprint.pprint(len(docs))
print('\n')
result = chain.run(input_documents=docs, question=query)
pprint.pprint(result)
