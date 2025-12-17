#!/usr/bin/env python3
"""Text extraction utilities with CLI for PDFs and common text files."""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

__all__ = [
    "extract_from_file",
    "discover_files",
    "summarize_text",
    "extract_batch",
    "main",
]


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")


def read_ipynb(path: Path) -> str:
    try:
        import json as _json
        nb = _json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return ""
    parts = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            parts.append("\n".join(cell.get("source", [])))
        elif cell.get("cell_type") == "code":
            parts.append("\n".join(cell.get("source", [])))
    return "\n\n".join(parts)


def read_pdf_pymupdf(path: Path, max_pages: int | None) -> str:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return ""
    text_parts = []
    try:
        with fitz.open(path) as doc:
            n_pages = len(doc)
            last = n_pages if max_pages is None else min(n_pages, max_pages)
            for i in range(last):
                page = doc.load_page(i)
                text_parts.append(page.get_text("text"))
    except Exception:
        return ""
    return "\n".join(text_parts)


def read_pdf_fallback_pdfminer(path: Path, max_pages: int | None) -> str:
    try:
        from pdfminer.high_level import extract_text
    except Exception:
        return ""
    try:
        return extract_text(str(path), maxpages=max_pages)
    except Exception:
        return ""


def extract_from_file(path: Path, max_pages: int | None) -> dict:
    ext = path.suffix.lower()
    text = ""
    meta = {"path": str(path), "type": ext.lstrip("."), "size": path.stat().st_size}

    if ext in {".md", ".txt", ".py", ".html", ".htm", ".json", ".csv"}:
        text = read_text_file(path)
    elif ext == ".ipynb":
        text = read_ipynb(path)
    elif ext == ".pdf":
        text = read_pdf_pymupdf(path, max_pages=max_pages)
        if not text:
            text = read_pdf_fallback_pdfminer(path, max_pages=max_pages)
            if not text:
                meta["note"] = "No extractable text found (scanned/needs OCR)."
    else:
        meta["skipped"] = True
    meta["chars"] = len(text)
    meta["empty"] = len(text.strip()) == 0
    return {"meta": meta, "text": text}


def discover_files(paths: Iterable[Path]) -> list[Path]:
    results: list[Path] = []
    for p in paths:
        if p.is_file():
            results.append(p)
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    results.append(f)
    return results


def summarize_text(text: str, max_chars: int = 1200) -> str:
    t = " ".join(text.split())
    return t[:max_chars]


def extract_batch(
    paths: List[Path],
    out_dir: Path,
    glob: str | None = None,
    max_pages: int | None = None,
) -> Tuple[int, Path]:
    """Extract a batch of files and return count and index path."""

    out_dir.mkdir(parents=True, exist_ok=True)
    all_files = discover_files(paths)
    if glob:
        all_files = [f for f in all_files if f.match(glob)]

    processed = []
    for f in all_files:
        if f.name.startswith("."):
            continue
        res = extract_from_file(f, max_pages)
        meta = res["meta"]
        text = res["text"]

        rel = f.relative_to(paths[0] if len(paths) == 1 else Path.cwd())
        safe_rel = str(rel).replace("/", "__").replace("\\", "__")
        base_name = safe_rel.rsplit(".", 1)[0] if "." in safe_rel else safe_rel
        target_txt = out_dir / f"{base_name}.txt"
        target_meta = out_dir / f"{base_name}.json"

        if text:
            target_txt.parent.mkdir(parents=True, exist_ok=True)
            target_txt.write_text(text, encoding="utf-8")
        target_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        preview = summarize_text(text)
        processed.append(
            {
                "path": meta["path"],
                "type": meta["type"],
                "chars": meta["chars"],
                "empty": meta["empty"],
                "note": meta.get("note"),
                "preview": preview,
            }
        )

    index = {"total": len(processed), "generated_at": str(Path.cwd()), "items": processed}
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(processed), index_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract text from PDFs and common text files.")
    parser.add_argument("--paths", nargs="*", default=["."], help="Files or folders to scan")
    parser.add_argument("--out", default="analysis", help="Output folder for extracted text")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit PDF pages to parse")
    parser.add_argument("--glob", default=None, help="Optional glob to filter files (e.g., '**/*.pdf')")
    args = parser.parse_args()

    base_paths = [Path(p).resolve() for p in args.paths]
    out_dir = Path(args.out).resolve()

    count, index_path = extract_batch(base_paths, out_dir, glob=args.glob, max_pages=args.max_pages)
    print(f"Processed {count} files. Index: {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
