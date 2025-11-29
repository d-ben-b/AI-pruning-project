import markdown
import sys
import os
import re
from weasyprint import HTML

def convert_md_to_pdf(input_file, output_file=None):
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.pdf'

    # Get the directory of the input file to use as base_url for images
    base_dir = os.path.dirname(os.path.abspath(input_file))
    
    print(f"Converting {input_file} to {output_file}...")
    print(f"Base directory for resources: {base_dir}")

    try:
        # Read Markdown
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # 1. Insert TOC after the first H1 header
        # Find the first line starting with # (but not ##)
        if '[TOC]' not in text:
            print("Inserting TOC after the first H1 header...")
            # Regex to find the first H1 and append [TOC] after it
            # Look for ^# followed by space and text, then newline
            text = re.sub(r'(^#\s+.*$)', r'\1\n\n[TOC]\n', text, count=1, flags=re.MULTILINE)

        # Convert to HTML with TOC extension
        html_content = markdown.markdown(text, extensions=['tables', 'fenced_code', 'toc'])

        # 2. Group "Bold Title" + "Image" to keep them together
        # Strategy: Find paragraphs containing <strong>...</strong> followed by <img ...>
        # and wrap them in a div with class="keep-together"
        print("Grouping titles and images...")
        # Regex explanation:
        # <p> matches start of paragraph
        # \s* matches optional whitespace
        # <strong>.*?</strong> matches the bold title
        # \s* matches whitespace/newlines
        # <img matches the image tag
        # .*? matches content of img tag
        # </p> matches end of paragraph
        html_content = re.sub(
            r'(<p>\s*<strong>.*?</strong>\s*<img.*?</p>)',
            r'<div class="keep-together">\1</div>',
            html_content,
            flags=re.DOTALL
        )

        # Add CSS
        css = """
        <style>
            @page {
                size: A4;
                margin: 2.5cm;
                @bottom-center {
                    content: counter(page);
                    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                    font-size: 10pt;
                    color: #7f8c8d;
                }
            }
            body {
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                font-size: 11pt;
                max-width: none;
                margin: 0;
                padding: 0;
            }
            
            /* Keep Together Class */
            .keep-together {
                page-break-inside: avoid;
                break-inside: avoid;
                margin-bottom: 20px;
                display: block;
            }

            /* Table of Contents Styling */
            .toc {
                background-color: #f9f9f9;
                border: 1px solid #e1e4e8;
                border-radius: 6px;
                padding: 20px;
                margin-bottom: 40px;
                page-break-after: always;
            }
            .toc ul {
                list-style-type: none;
                padding-left: 20px;
            }
            .toc > ul {
                padding-left: 0;
            }
            .toc a {
                text-decoration: none;
                color: #003366;
            }
            .toc a:hover {
                text-decoration: underline;
            }

            /* Headings */
            h1, h2, h3, h4, h5, h6 {
                color: #003366;
                font-weight: 600;
                margin-top: 1.5em;
                margin-bottom: 0.5em;
                page-break-after: avoid;
            }
            h1 {
                font-size: 24pt;
                border-bottom: 2px solid #003366;
                padding-bottom: 0.3em;
                margin-top: 0;
                page-break-before: always;
            }
            body > h1:first-child {
                page-break-before: avoid;
            }
            h2 {
                font-size: 18pt;
                border-bottom: 1px solid #eee;
                padding-bottom: 0.2em;
            }
            h3 {
                font-size: 14pt;
            }
            
            p {
                margin-bottom: 1em;
                text-align: justify;
                orphans: 3;
                widows: 3;
            }
            a {
                color: #003366;
                text-decoration: none;
            }
            
            /* Images */
            img {
                max-width: 100%;
                height: auto;
                display: block;
                margin: 10px auto; /* Reduced margin since we have wrapper */
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            /* Tables */
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                font-size: 10pt;
                page-break-inside: avoid;
            }
            th, td {
                border: 1px solid #e1e4e8;
                padding: 10px;
                text-align: left;
            }
            th {
                background-color: #f6f8fa;
                color: #003366;
                font-weight: 600;
            }
            tr:nth-child(even) {
                background-color: #fcfcfc;
            }
            
            /* Code Blocks */
            code {
                background-color: #f6f8fa;
                padding: 2px 4px;
                border-radius: 4px;
                font-family: "Consolas", "Monaco", monospace;
                font-size: 0.9em;
                color: #e83e8c;
            }
            pre {
                background-color: #f6f8fa;
                padding: 15px;
                border-radius: 6px;
                overflow-x: auto;
                border: 1px solid #e1e4e8;
                page-break-inside: avoid;
            }
            pre code {
                background-color: transparent;
                padding: 0;
                color: #24292e;
            }
            
            blockquote {
                border-left: 4px solid #003366;
                margin: 0;
                padding-left: 15px;
                color: #6a737d;
                background-color: #f9f9f9;
                padding: 10px 15px;
                page-break-inside: avoid;
            }
        </style>
        """
        
        full_html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Document</title>{css}</head><body>{html_content}</body></html>"

        # Convert to PDF
        HTML(string=full_html, base_url=base_dir).write_pdf(output_file)
        print(f"Successfully generated {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        input_file = input("Enter the input file: ")
    else:
        input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    convert_md_to_pdf(input_file, output_file)
