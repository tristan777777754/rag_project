import chromadb

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "pdf_chunks"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def store_chunks(chunks, embeddings):
    """
    Store chunk text, embeddings, and page metadata in ChromaDB.
    """

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [{"page": chunk["page"]} for chunk in chunks]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


def search_chunks(query_embedding: list[float], top_k: int = 5):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results


def get_collection_count():
    return collection.count()