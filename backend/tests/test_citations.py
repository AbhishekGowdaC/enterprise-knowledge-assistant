
import sys
from pathlib import Path

# Allow imports from the backend directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.generation.answer import validate_citations


def test_valid_citation():
    context = [
        "[Source: employee_handbook (1).pdf, chunk: 4]\n"
        "Employees receive 10 paid sick leave days per year."
    ]

    answer = (
        "Employees receive 10 paid sick leave days per year.\n\n"
        "[Source: employee_handbook (1).pdf, chunk: 4]"
    )

    valid, message = validate_citations(answer, context)
    assert valid is True, message


def test_fabricated_citation():
    context = [
        "[Source: employee_handbook (1).pdf, chunk: 4]\n"
        "Employees receive 10 paid sick leave days per year."
    ]

    answer = (
        "Employees receive 10 paid sick leave days per year.\n\n"
        "[Source: employee_handbook (1).pdf, chunk: 99]"
    )

    valid, message = validate_citations(answer, context)
    assert valid is False
    assert "not found" in message.lower()


def test_missing_citation():
    context = [
        "[Source: employee_handbook (1).pdf, chunk: 4]\n"
        "Employees receive 10 paid sick leave days per year."
    ]

    answer = "Employees receive 10 paid sick leave days per year."

    valid, message = validate_citations(answer, context)
    assert valid is False
    assert "no valid source citations" in message.lower()


if __name__ == "__main__":
    test_valid_citation()
    test_fabricated_citation()
    test_missing_citation()
    print("All 3 citation tests passed!")
