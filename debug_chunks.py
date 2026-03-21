import sys
from app.ingestion.pdf_loader import load_pdf
from app.processing.text_splitter import split_pages_into_chunks_semantic


pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/quant3.pdf"

# 1. load pdf
pages = load_pdf(pdf_path)

print(f"Total pages: {len(pages)}")

# 2. split into chunks
chunks = split_pages_into_chunks_semantic(pages)

print(f"Total chunks: {len(chunks)}")


print("\n=== FIRST 30 CHUNKS ===")
for chunk in chunks[:30]:
    print(f"Page {chunk['page']} | Section: {chunk['section']} | Topic: {chunk.get('section_topic')} | Subtopic: {chunk.get('subsection_topic')}")

print("\n=== PAGE SECTION PREVIEW ===")
seen_pages = set()

for chunk in chunks:
    page = chunk["page"]
    if page in seen_pages:
        continue
    seen_pages.add(page)

    preview = chunk["text"][:300].replace("\n", " ")
    print(f"\n--- Page {page} | Section: {chunk['section']} | Topic: {chunk.get('section_topic')} | Subtopic: {chunk.get('subsection_topic')} ---")
    print(preview)

    if page >= 10:
        break
