import os

import chromadb
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore


project_dir = os.path.dirname(os.path.abspath(__file__))
vector_db_path = os.path.join(project_dir, "vector_db")
collection_name = "documents_collection"
load_dotenv(os.path.join(project_dir, ".env"))
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-distilroberta-v1")
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key or groq_api_key == "your_groq_api_key_here":
	raise SystemExit(
		"Set GROQ_API_KEY in the project .env file before starting retrieval."
	)
Settings.llm = Groq(model="openai/gpt-oss-120b")



chroma_client = chromadb.PersistentClient(path=vector_db_path)
try:
	chroma_collection = chroma_client.get_or_create_collection(collection_name)
except Exception as error:
	raise SystemExit(
		f"Could not open Chroma collection '{collection_name}': {error}"
	) from error

vector_count = chroma_collection.count()
if vector_count == 0:
	raise SystemExit(
		f"Chroma collection '{collection_name}' is empty. Run the ingestion script first."
	)

print(f"Loaded {vector_count} vectors from Chroma.")
#connect to the vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# Load index from Chroma
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    embed_model=embed_model
)

query_engine = index.as_query_engine(similarity_top_k=3)

# Ask your question
#input_query = input("Enter your question: ")
#response = query_engine.query("What does the document say about adaptive radiation?")
#response = query_engine.query(input_query)
while True:
    input_query = input("Enter your question (or 'exit' to quit): ")
    if input_query.lower() == "exit":
        break
    response = query_engine.query(input_query)
    print("Response from query engine:", response)

