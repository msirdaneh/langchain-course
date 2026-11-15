import os
from dotenv import load_dotenv
# Cmd + Shift + P to set the python interperter
from langchain_community.document_loaders import TextLoader #deprected
from langchain_core import embeddings
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    #print(os.environ["PINECONE_API_KEY"])
    
    # loader = UnstructuredLoader(file_path="./mediumblog1.txt",
    # chunking_strategy="basic",
    # max_characters=1000000,
    # encoding="UTF-8"
    # )
    loader = TextLoader(
        file_path="./mediumblog1.txt",
        encoding="utf-8"
    )
    print("Loading document...")
    document = loader.load()
    print(f"Loaded {len(document)} documents")
    
    print("Splitting...")
    text_splitter = CharacterTextSplitter(
        chunk_size= 1000,
        chunk_overlap = 0
    )
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_type=os.environ["OPENAI_API_KEY"])

    print("Ingesting...")
    PineconeVectorStore.from_documents(
       documents= texts, 
        embedding= embeddings,
        index_name = os.environ["INDEX_NAME"]
    )

    print("Finish")

