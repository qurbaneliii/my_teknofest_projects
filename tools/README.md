# File Analysis Tools

This folder contains a small CLI to extract text from PDFs and other common file types in this repo, and to build a searchable index.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tools/extract_text.py --paths agroscan sonic --out analysis
```

Options:
- `--max-pages <N>`: limit how many PDF pages are parsed
- `--glob "**/*.pdf"`: process only certain file patterns

Outputs:
- `analysis/index.json`: summary of processed files with short previews
- `analysis/<file>.txt`: extracted text for each file when available
- `analysis/<file>.json`: per-file metadata

Notes:
- Some scanned PDFs contain only images; these require OCR to extract text. For that, install Tesseract and `pytesseract` (optional) and extend the script if needed.
```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr
pip install pytesseract pillow
```
```