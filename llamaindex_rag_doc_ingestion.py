import os

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

import nltk
import ssl

"""
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

#nltk.download("punkt_tab")
#nltk.download("stopwords")
"""
# configuration
project_dir = os.path.dirname(os.path.abspath(__file__))
docs_dir_path = os.path.join(project_dir, "docs_dir")
vector_db_path = os.path.join(project_dir, "vector_db")
collection_name="documents_collection"

# define embedding function - using fastest lightweight model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-distilroberta-v1")

# directory loader
loader = SimpleDirectoryReader(input_dir=docs_dir_path)
# load the documents
documents = loader.load_data()
#print('fulldocuments loaded:', documents)
print('Number of documents loaded:', len(documents))

# Define persistent DB location
db = chromadb.PersistentClient(path=vector_db_path)
# Create or retrieve the vector collection
chroma_collection = db.get_or_create_collection(name=collection_name)

print('Chroma collection created or retrieved:', chroma_collection)

# Create parser with chunking strategy - smaller chunks for speed
parser = SimpleNodeParser.from_defaults(chunk_size=256, chunk_overlap=10)
# Convert documents to chunks (nodes)
nodes = parser.get_nodes_from_documents(documents)
print('Number of nodes created from documents:', len(nodes))
#print('nodes created from documents:', nodes)
# Create vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
print('Vector store created with Chroma collection:', vector_store)

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)
print('Storage context created with vector store:', storage_context)

print("Vectors stored in Chroma:", chroma_collection.count())
# Create the vector store index
index = VectorStoreIndex(
    nodes, 
    storage_context=storage_context, 
    vector_store=vector_store, 
    embed_model=embed_model
)
print("Index created successfully!")
print("Total vectors in Chroma after indexing:", chroma_collection.count())


