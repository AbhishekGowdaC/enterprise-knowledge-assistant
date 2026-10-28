
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "sample_policy.pdf"


def main():
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(PDF_PATH), pagesize=letter)

    pdf.setTitle("Employee Policy Test")
    pdf.drawString(72, 740, "Employee Leave Policy")
    pdf.drawString(
        72, 710,
        "Full-time employees receive 20 days of annual leave per year."
    )
    pdf.drawString(
        72, 680,
        "Leave requests must be approved by the employee's manager."
    )

    pdf.showPage()
    pdf.drawString(72, 740, "Additional Policy")
    pdf.drawString(
        72, 710,
        "Employees should submit leave requests in advance."
    )
    pdf.save()

    print(f"Created test PDF: {PDF_PATH}")


if __name__ == "__main__":
    main()
