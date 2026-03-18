def split_pages_into_chunks(pages, chunk_size=800, overlap=100):
    """
    Split each page into chunks while keeping page numbers.
    """

    chunks = []

    for page in pages:
        text = page["text"]
        page_num = page["page"]

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": page_num
                })

            start += chunk_size - overlap

    return chunks