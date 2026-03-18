import fitz


def load_pdf(file_path: str):
    """
    Load PDF and return a list of pages with cleaned text in reading order.
    """
    doc = fitz.open(file_path)
    pages = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")

        # sort blocks by vertical position first, then horizontal position
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

        page_text_parts = []

        for block in blocks:
            x0, y0, x1, y1, text, *_ = block

            cleaned = text.strip()
            if not cleaned:
                continue

            # skip tiny noisy blocks
            if len(cleaned) < 20:
                continue

            page_text_parts.append(cleaned)

        page_text = "\n".join(page_text_parts)

        if page_text.strip():
            pages.append({
                "page": page_num + 1,
                "text": page_text
            })

    doc.close()
    return pages