import re
import fitz
from statistics import median


def _looks_like_short_header(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False
    normalized = re.sub(r"\s+", " ", cleaned).strip()
    if len(normalized) > 80:
        return False
    if re.search(r"[!?]", normalized):
        return False
    words = normalized.split()
    if not (1 <= len(words) <= 10):
        return False
    if re.match(r"^(\d+(\.\d+)*|[A-Za-z])\s*[\.|-]?\s+[A-Za-z]", normalized):
        return True
    if re.search(r"[.:;]", normalized):
        return False
    return any(re.search(r"[A-Za-z]", w) for w in words)


def _extract_line_meta(page):
    data = page.get_text("dict")
    lines_meta = []
    sizes = []

    for block in data.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            bbox = line.get("bbox", [0, 0, 0, 0])
            max_size = max((span.get("size", 0) for span in spans), default=0)
            max_flags = max((span.get("flags", 0) for span in spans), default=0)
            is_upper = sum(1 for c in text if c.isalpha() and c.isupper()) > sum(1 for c in text if c.isalpha() and c.islower())
            item = {
                "text": text,
                "y0": bbox[1],
                "x0": bbox[0],
                "y1": bbox[3],
                "size": max_size,
                "flags": max_flags,
                "is_upper": is_upper,
            }
            lines_meta.append(item)
            sizes.append(max_size)

    lines_meta.sort(key=lambda x: (x["y0"], x["x0"]))
    body_size = median(sizes) if sizes else 0
    return lines_meta, body_size


def load_pdf(file_path: str):
    """Load PDF and preserve both text and lightweight layout metadata."""
    doc = fitz.open(file_path)
    pages = []

    for page_num, page in enumerate(doc):
        blocks = sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0]))
        page_text_parts = []

        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            cleaned = text.strip()
            if not cleaned:
                continue
            if len(cleaned) < 20 and not _looks_like_short_header(cleaned):
                continue
            page_text_parts.append(cleaned)

        page_text = "\n".join(page_text_parts)
        lines_meta, body_size = _extract_line_meta(page)

        if page_text.strip():
            pages.append({
                "page": page_num + 1,
                "text": page_text,
                "lines_meta": lines_meta,
                "body_font_size": body_size,
            })

    doc.close()
    return pages
