from weasyprint import HTML
import sys

def convert_html_to_pdf(html_file, pdf_file):
    print(f"Converting {html_file} to {pdf_file}...")
    try:
        HTML(html_file).write_pdf(pdf_file)
        print("Successfully generated PDF.")
    except Exception as e:
        print(f"Error generating PDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    convert_html_to_pdf('report.html', 'report.pdf')
