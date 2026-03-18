from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-small-en")


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    embedding = embedding_model.encode(query, convert_to_numpy=True)
    return embedding.tolist()