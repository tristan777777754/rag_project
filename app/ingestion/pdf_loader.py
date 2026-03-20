import re
import fitz


def _looks_like_short_header(text: str) -> bool:
    """Keep genuine short section headers while still dropping tiny noise."""
    cleaned = text.strip()
    if not cleaned:
        return False
    normalized = re.sub(r"\s+", " ", cleaned).strip()
    if len(normalized) > 60:
        return False
    if re.search(r"[.!?]", normalized):
        return False
    words = normalized.split()
    if not (1 <= len(words) <= 8):
        return False
    if any(ch.isdigit() for ch in normalized):
        return False
    return normalized[:1].isalpha()


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

            # Skip tiny noisy blocks, but preserve short section headers such as
            # "Summary", "Methods", "Conclusion", or "References".
            if len(cleaned) < 20 and not _looks_like_short_header(cleaned):
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