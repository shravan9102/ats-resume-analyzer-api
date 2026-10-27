import io
import pytest
from docx import Document
from reportlab.pdfgen import canvas

from backend.services.resume_parser import parse_resume_file, validate_file
from backend.utils.file_utils import FileParsingError, FileValidationError


def make_pdf(text="Jane Doe\nPython developer with FastAPI experience"):
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 800
    for line in text.split("\n"):
        c.drawString(72, y, line); y -= 20
    c.save()
    return buf.getvalue()


def make_docx(text="Jane Doe. Python developer."):
    d = Document(); d.add_paragraph(text)
    buf = io.BytesIO(); d.save(buf)
    return buf.getvalue()


def test_pdf_ok():
    text, meta = parse_resume_file(make_pdf(), "r.pdf")
    assert "Python developer" in text and meta["file_type"] == "pdf"


def test_docx_ok():
    text, meta = parse_resume_file(make_docx(), "r.docx")
    assert "Python developer" in text and meta["file_type"] == "docx"


def test_txt_ok():
    text, meta = parse_resume_file("Jane\nSkills: Python".encode(), "r.txt")
    assert "Skills" in text and meta["file_type"] == "txt"


def test_empty_file_gives_clear_error():
    ok, msg, ftype = validate_file(b"", "r.pdf")   # used to raise a 2-tuple unpack error
    assert not ok and "empty" in msg.lower() and ftype is None
    with pytest.raises(FileValidationError):
        parse_resume_file(b"", "r.pdf")


def test_garbage_rejected():
    with pytest.raises(FileValidationError):
        parse_resume_file(b"\x00\x01\x02binary", "r.pdf")


def test_corrupt_pdf_gives_parsing_error():
    with pytest.raises(FileParsingError):
        parse_resume_file(b"%PDF-1.4\nnot really a pdf", "r.pdf")


def test_blank_pdf_no_text():
    buf = io.BytesIO(); c = canvas.Canvas(buf); c.showPage(); c.save()
    with pytest.raises(FileParsingError):
        parse_resume_file(buf.getvalue(), "blank.pdf")
