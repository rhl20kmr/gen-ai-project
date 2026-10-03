import os
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

project_dir = os.path.dirname(os.path.abspath(__file__))
vector_db_path = os.path.join(project_dir, "vector_db")

print("=== RETRIEVING CHUNKS ===\n")
chroma_client = chromadb.PersistentClient(path=vector_db_path)
chroma_collection = chroma_client.get_collection("documents_collection")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
)

retriever = index.as_retriever(similarity_top_k=3)
retrieved_nodes = retriever.retrieve("What does the document say about adaptive radiation?")

for i, node in enumerate(retrieved_nodes):
    print(f"\n[Chunk {i+1}]")
    print(f"File: {node.node.metadata.get('file_name')}")
    print(f"Score: {node.score}")
    print(f"Text preview:\n{node.node.text[:600]}")
    print("-" * 80)
