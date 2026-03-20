from app.ingestion.pdf_loader import load_pdf
from app.processing.text_splitter import split_pages_into_chunks_semantic


pdf_path = "data/raw/quant1.pdf"

# 1. load pdf
pages = load_pdf("data/raw/quant1.pdf")

print(f"Total pages: {len(pages)}")

# 2. split into chunks
chunks = split_pages_into_chunks_semantic(pages)

print(f"Total chunks: {len(chunks)}")


print("\n=== FIRST 30 CHUNKS ===")
for chunk in chunks[:30]:
    print(f"Page {chunk['page']} | Section: {chunk['section']}")

print("\n=== PAGE SECTION PREVIEW ===")
seen_pages = set()

for chunk in chunks:
    page = chunk["page"]
    if page in seen_pages:
        continue
    seen_pages.add(page)

    preview = chunk["text"][:300].replace("\n", " ")
    print(f"\n--- Page {page} | Section: {chunk['section']} ---")
    print(preview)

    if page >= 10:
        break