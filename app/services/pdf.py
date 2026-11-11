from io import BytesIO

from pypdf import PdfReader


def extract_pages(content: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(BytesIO(content))
    if reader.is_encrypted:
        try:
            if not reader.decrypt(""):
                raise ValueError("Encrypted PDFs are not supported")
        except Exception as exc:
            raise ValueError("Encrypted PDFs are not supported") from exc
    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        if text:
            pages.append((page_number, text))
    return pages


def chunk_pages(
    pages: list[tuple[int, str]], chunk_size: int, overlap: int
) -> list[tuple[int, str]]:
    chunks: list[tuple[int, str]] = []
    step = chunk_size - overlap
    for page_number, text in pages:
        for start in range(0, len(text), step):
            chunk = text[start : start + chunk_size].strip()
            if chunk:
                chunks.append((page_number, chunk))
    return chunks
