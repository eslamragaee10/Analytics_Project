from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.embeddings.get_embedding import BGEM3Embeddings
from src.ingestion.loader import load_documents
import os
import shutil

CHROMA_DB_PATH = "chroma_db"

def populate_database():
    answer = input("Do you need to clear the database : (y/n)\nif yes it will be cleared and populate the new one\nyour answer: ")
    if answer.lower() == "y":
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)
            print("Old database cleared.")

    documents = load_documents("data/")
    if not documents:
        print("No database found")
        return

    print(f"Loaded {len(documents)} pages")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    if not chunks:
        print("No chunks to embed. Please check your data folder.")
        return

    embedder = BGEM3Embeddings(model_name="bge-m3")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=CHROMA_DB_PATH
    )

    print(f"Database built successfully with {len(chunks)} chunks.")

if __name__ == "__main__":
    populate_database()
