from app.ingestion.pdf_loader import load_pdf
from app.processing.text_splitter import split_pages_into_chunks_semantic, filter_noisy_chunks
from app.embeddings.embedder import embed_texts
from app.vector_store.chroma_store import store_chunks, get_collection_count
from app.retrieval.retriever import retrieve_relevant_chunks, reset_retriever
from app.llm.kimi_client import ask_kimi_with_context


def ingest_pdf(pdf_path: str = "data/raw/quant1.pdf"):
    pages = load_pdf(pdf_path)
    chunks = split_pages_into_chunks_semantic(pages)
    filtered_chunks = filter_noisy_chunks(chunks)

    text_list = [chunk["text"] for chunk in filtered_chunks]
    embeddings = embed_texts(text_list)

    store_chunks(filtered_chunks, embeddings)
    
    # Reset BM25 retriever so it will be rebuilt with new chunks
    reset_retriever()

    print(f"PDF ingested successfully from {pdf_path}!")
    print("Total chunks stored:", get_collection_count())


def chat():
    print("\nRAG PDF Chatbot")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        retrieved_chunks = retrieve_relevant_chunks(question, top_k=5)

        print("\nRetrieved chunks:\n")
        for i, chunk in enumerate(retrieved_chunks, start=1):
            print(f"--- Result {i} (Page {chunk['page']}) ---")
            print(chunk["text"][:400])
            print()

        context_texts = [chunk["text"] for chunk in retrieved_chunks]
        source_pages = sorted(set(str(chunk["page"]) for chunk in retrieved_chunks))

        answer = ask_kimi_with_context(question, context_texts)

        print("\nFinal Answer:\n")
        print(answer)
        print("\nSources: pages", ", ".join(source_pages))
        print("\n" + "=" * 60 + "\n")


def main():
    ingest_pdf()
    chat()


if __name__ == "__main__":
    main()