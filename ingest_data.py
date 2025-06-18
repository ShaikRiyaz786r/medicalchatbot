import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# --- INITIALIZATION ---
print("Initializing clients...")
try:
    genai.configure(api_key="AIzaSyDeZsXSMnMPi0A4jqu1YvXh9lABPkwufEA")
    pc = Pinecone(api_key='pcsk_32k22V_77oFKp7mL369qkiGoQpdN5PhdiixMoZg94f1tsXhkFkQVZ4vQg8RRo1mEV3u9RD')
except (KeyError, TypeError) as e:
    print("FATAL ERROR: Make sure GOOGLE_API_KEY and PINECONE_API_KEY are in your .env file.")
    exit()

embedding_model = "models/text-embedding-004"
index_name = "medical-chatbot3"
data_path = "data/"


# --- DATA LOADING & PROCESSING ---
def process_and_upload():
    """
    Loads PDFs, splits them into chunks, creates embeddings, and upserts to Pinecone.
    """
    # 1. Load PDFs
    print(f"Loading documents from {data_path}...")
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        print("No PDF documents found. Exiting.")
        return

    print(f"Loaded {len(documents)} pages.")

    # 2. Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)
    print(f"Created {len(text_chunks)} text chunks.")

    # 3. Create embeddings for chunks
    print("Creating embeddings for text chunks...")
    page_contents = [chunk.page_content for chunk in text_chunks]
    response = genai.embed_content(
        model=embedding_model,
        content=page_contents,
        task_type="RETRIEVAL_DOCUMENT"
    )
    doc_embeddings = response['embedding']
    print("Embeddings created successfully.")

    # 4. Connect to or Create Pinecone Index
    embedding_dimension = len(doc_embeddings[0])
    print(f"Connecting to '{index_name}' index...")
    if index_name not in pc.list_indexes().names():
        print(f"Index '{index_name}' does not exist. Creating it...")
        pc.create_index(
            name=index_name,
            dimension=embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print("Index created.")
    index = pc.Index(index_name)

    # 5. Prepare and Batch Upsert Data
    print("Preparing data for upload...")
    vectors_to_upsert = []
    for i, (chunk, vec) in enumerate(zip(text_chunks, doc_embeddings)):
        vectors_to_upsert.append({
            "id": str(i),
            "values": vec,
            "metadata": {"text": chunk.page_content, "source": chunk.metadata.get('source', 'Unknown'), "page": chunk.metadata.get('page', 'Unknown')}
        })

    batch_size = 100
    print(f"Upserting {len(vectors_to_upsert)} vectors in batches of {batch_size}...")
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i : i + batch_size]
        print(f"Upserting batch {i//batch_size + 1}...")
        index.upsert(vectors=batch)
    
    print("Upload complete.")
    stats = index.describe_index_stats()
    print(f"\nIndex stats: {stats}")

if __name__ == '__main__':
    process_and_upload()