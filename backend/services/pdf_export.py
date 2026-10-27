import io
import logging

try:
    from weasyprint import HTML, CSS
    WEASYPRINT_INSTALLED = True
    WEASYPRINT_ERROR = ""
except (ImportError, OSError) as _exc:
    # OSError: native libraries (pango/gobject) missing, e.g. on Windows.
    WEASYPRINT_INSTALLED = False
    WEASYPRINT_ERROR = str(_exc)

logger = logging.getLogger('ats_resume_scorer')

def generate_combined_pdf(html_docs: dict[str, str]) -> bytes:
    if not WEASYPRINT_INSTALLED:
        raise ImportError(
            "PDF generation is unavailable: WeasyPrint or its system libraries "
            f"(pango/gobject) could not be loaded. {WEASYPRINT_ERROR}"
        )
        
    documents = []
    
    # Render all 3 HTML strings to WeasyPrint Document objects
    for name, html_str in html_docs.items():
        doc = HTML(string=html_str).render()
        documents.append(doc)
    
    # Merge them into the first document
    first_doc = documents[0]
    for other_doc in documents[1:]:
        for page in other_doc.pages:
            first_doc.pages.append(page)
            
    # Write combined PDF bytes
    pdf_bytes = first_doc.write_pdf()
    return pdf_bytes