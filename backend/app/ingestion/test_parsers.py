
from pathlib import Path
from app.ingestion.parsers import parse_html, parse_pdf


def test_html_parser():
    path = Path("../data/raw/sample.html")
    records = parse_html(str(path))

    assert len(records) == 1
    assert records[0]["source"] == "sample.html"
    assert records[0]["file_type"] == "html"
    assert "20 days of annual leave" in records[0]["text"]

    print("HTML parser: PASS")
    print(records[0])


if __name__ == "__main__":
    test_html_parser()
