import os
from typing import Any
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone
# from const import INDEX_NAME
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

def run_llm(query:str)->Any:
    embeddings = OpenAIEmbeddings()
    dosearch = PineconeVectorStore.from_existing_index(index_name="doc-reader",embedding=embeddings)
    chat = ChatOpenAI(verbose=True,temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=dosearch.as_retriever(),
        return_source_documents=True
        )
    return qa({"query":query})

if __name__ == "__main__":
    run_llm(query="what is Google GenerativeAI?")