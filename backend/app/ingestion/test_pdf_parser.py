
from pathlib import Path

from app.ingestion.parsers import parse_pdf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "sample_policy.pdf"


def test_pdf_parser():
    records = parse_pdf(str(PDF_PATH))

    assert len(records) == 2, (
        f"Expected 2 page records, got {len(records)}"
    )

    assert records[0]["source"] == "sample_policy.pdf"
    assert records[0]["file_type"] == "pdf"
    assert records[0]["page"] == 1
    assert "20 days of annual leave" in records[0]["text"]

    assert records[1]["page"] == 2
    assert "submit leave requests in advance" in records[1]["text"]

    print("PDF parser: PASS")
    for record in records:
        print(
            f"Page {record['page']}: "
            f"{record['text']}"
        )


if __name__ == "__main__":
    test_pdf_parser()
