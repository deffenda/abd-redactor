import fitz
import json
from pathlib import Path

def test_metrics_overlay(input_pdf, output_pdf):
    # Dummy metrics
    total_hits = 42
    pages_with_redactions = 3
    redactions_per_page = {"1": 10, "2": 20, "3": 12}
    metrics_text = f"PII Redaction Metrics:\n"
    metrics_text += f"Total redactions: {total_hits}\n"
    metrics_text += f"Pages with redactions: {pages_with_redactions}\n"
    metrics_text += f"Redactions per page: {json.dumps(redactions_per_page)}\n"

    with fitz.open(input_pdf) as doc:
        if doc.page_count > 0:
            last_page = doc[-1]
            rect = fitz.Rect(36, 36, 400, 120 + 12 * len(redactions_per_page))
            last_page.insert_textbox(rect, metrics_text, fontsize=10, color=(0, 0, 1))
        doc.save(output_pdf, garbage=4, deflate=True, clean=True)

if __name__ == "__main__":
    test_metrics_overlay("/Users/deffenda/Downloads/health_report.pdf", "/Users/deffenda/Desktop/metrics_overlay_test.pdf")
    print("Overlay test complete. Check metrics_overlay_test.pdf on your Desktop.")
