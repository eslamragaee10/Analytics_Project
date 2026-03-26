from src.ingestion.loader import load_documents
from src.ingestion.chunking import chunk_documents
from src.embeddings.get_embedding import BGEM3Embeddings


# docs = load_documents()
# chunks = chunk_documents(documents=docs)
embed = BGEM3Embeddings()
vector = embed.embed_query("chunks")
print(vector)
