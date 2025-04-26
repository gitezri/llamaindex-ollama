from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
#from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
#Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
from llama_index.embeddings.ollama import OllamaEmbedding

ollama_embed = OllamaEmbedding( model_name="nomic-embed-text", base_url="http://ai.localhost:11434", ) # Point at your local Ollama server

Settings.embed_model = ollama_embed # Use Ollama embeddings globally

load_dotenv()

documents = SimpleDirectoryReader("pdf/").load_data()

index = VectorStoreIndex.from_documents(documents) 

query_engine = index.as_query_engine()

response = query_engine.query("What are the design goals and give minimum details about it please.")

print(response)
