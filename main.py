from src.ingestion.loader import load_documents
from src.ingestion.chunking import chunk_documents
from src.embeddings.get_embedding import BGEM3Embeddings
from ollama import chat

docs = load_documents()

chunks = chunk_documents(documents=docs)

texts = []
for chunk in chunks:
    texts.append(chunk.page_content)

chunn = texts[0:2]

embed = BGEM3Embeddings()
vector = embed.embed_documents(chunn)
print("Number of vectors:", len(vector))

question = input("Type your question: ")

answer = chat(
    model="mistral",
    messages=[{"role": "user", "content": question}]
)

print("The Answer:", answer['message']['content'])
