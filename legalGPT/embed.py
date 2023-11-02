from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

import os
from dotenv import load_dotenv

load_dotenv()

DATASET = "legalGPT/dataset/"
FAISS_INDEX = "legalGPT/vectorstore/"


def embed_all():
    """
    Embed all the documents (pdf) in the dataset folder
    """

    loader = DirectoryLoader(DATASET, loader_cls=PyPDFLoader)

    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    data_chunks = splitter.split_documents(data)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    vectorstore = FAISS.from_documents(data_chunks, embeddings)

    vectorstore.save_local(FAISS_INDEX)


if __name__ == "__main__":
    embed_all()
