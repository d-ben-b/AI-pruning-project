import markdown
import os

# Read the markdown file
with open('report.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Convert to HTML
html = markdown.markdown(text, extensions=['tables', 'fenced_code'])

# Add some basic CSS for better printing
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
    h1, h2, h3 {
        color: #2c3e50;
    }
    code {
        background-color: #f8f9fa;
        padding: 2px 4px;
        border-radius: 4px;
    }
    @media print {
        body {
            max-width: 100%;
            padding: 0;
        }
    }
</style>
"""

# Write the HTML file
with open('report.html', 'w', encoding='utf-8') as f:
    f.write(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Midterm Project Report</title>{css}</head><body>")
    f.write(html)
    f.write("</body></html>")

print("Successfully generated report.html")
