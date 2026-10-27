from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader


def parse_pdf(file_path: str) -> list[dict]:
    path = Path(file_path)
    reader = PdfReader(str(path))
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            records.append({
                "text": text.strip(),
                "source": path.name,
                "page": page_number,
                "file_type": "pdf",
            })

    return records


def parse_html(file_path: str) -> list[dict]:

    path = Path(file_path)
    html = path.read_text(encoding="utf=8")
    soup = BeautifulSoup(html, "html.parser")

    for element in soup(["script", "style", "nav", "footer"]):
        element.decompose()

    text = soup.get_text(separator=" ", strip=True)

    return [{
        "text": text,
        "source": path.name,
        "page": None,
        "file_type": "html",
    }] if text else []