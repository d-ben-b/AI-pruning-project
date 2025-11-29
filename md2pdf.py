import markdown
import sys
import os
from weasyprint import HTML

def convert_md_to_pdf(input_file, output_file=None):
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.pdf'

    print(f"Converting {input_file} to {output_file}...")

    try:
        # Read Markdown
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Convert to HTML
        html_content = markdown.markdown(text, extensions=['tables', 'fenced_code'])

        # Add CSS
        css = """
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }
            img {
                max-width: 100%;
                height: auto;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            code {
                background-color: #f8f9fa;
                padding: 2px 4px;
                border-radius: 4px;
            }
        </style>
        """
        
        full_html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Document</title>{css}</head><body>{html_content}</body></html>"

        # Convert to PDF
        HTML(string=full_html, base_url=os.getcwd()).write_pdf(output_file)
        print(f"Successfully generated {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python md2pdf.py <input_file.md> [output_file.pdf]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    convert_md_to_pdf(input_file, output_file)
