from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from const import INDEX_NAME

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

def ingest_docs() -> None:
    loader = ReadTheDocsLoader(path="langchain-docs",encoding='utf-8')
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20,separators=["\n\n","\n"," ",""])
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain_docs","https:/")
        doc.metadata.update({"source":new_url})

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings()
    PineconeVectorStore.from_documents(documents,embeddings,index_name=INDEX_NAME)
    print("******Added to pinecone vector data store*****")

if __name__ == "__main__":
    ingest_docs()