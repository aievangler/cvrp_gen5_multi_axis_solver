#!/usr/bin/env python3
"""
Minimal PDF generator for the Gen5_v5 handoff markdown.

Why custom: avoid adding heavy deps (reportlab/weasyprint). This writes a simple
multi-page PDF with monospaced text.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


def _escape_pdf_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _md_to_plain_lines(md: str) -> list[str]:
    lines: list[str] = []
    for raw in md.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("#"):
            # Normalize headings to plain.
            stripped = line.lstrip("#").strip()
            if stripped:
                lines.append(stripped.upper())
                lines.append("")
            continue
        lines.append(line)
    return lines


def _wrap_lines(lines: list[str], width: int) -> list[str]:
    out: list[str] = []
    for line in lines:
        if not line.strip():
            out.append("")
            continue
        indent = len(line) - len(line.lstrip(" "))
        prefix = " " * indent
        wrapped = textwrap.wrap(
            line.strip(),
            width=width,
            subsequent_indent=prefix,
            break_long_words=False,
            break_on_hyphens=False,
        )
        out.extend(wrapped if wrapped else [""])
    return out


def write_simple_pdf(text: str, out_path: Path) -> None:
    # Letter size in points.
    page_w, page_h = 612, 792
    margin_x, margin_y = 40, 40
    font_size = 9
    leading = 11

    max_lines_per_page = int((page_h - 2 * margin_y) // leading) - 1
    start_x = margin_x
    start_y = page_h - margin_y - font_size

    plain = _md_to_plain_lines(text)
    wrapped = _wrap_lines(plain, width=96)

    pages: list[list[str]] = []
    for i in range(0, len(wrapped), max_lines_per_page):
        pages.append(wrapped[i : i + max_lines_per_page])

    objects: list[bytes] = []

    def add_obj(obj_bytes: bytes) -> int:
        objects.append(obj_bytes)
        return len(objects)

    # 1) Catalog
    catalog_id = add_obj(b"<< /Type /Catalog /Pages 2 0 R >>")

    # 2) Pages object placeholder (Kids filled later)
    pages_id = add_obj(b"<< /Type /Pages /Kids [] /Count 0 >>")

    # 3) Font (monospace)
    font_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    page_obj_ids: list[int] = []
    content_obj_ids: list[int] = []

    for page_index, page_lines in enumerate(pages, start=1):
        # Content stream
        content_lines: list[str] = []
        content_lines.append("BT")
        content_lines.append(f"/F1 {font_size} Tf")
        content_lines.append(f"{start_x} {start_y} Td")
        content_lines.append(f"{leading} TL")
        for ln in page_lines:
            esc = _escape_pdf_text(ln)
            content_lines.append(f"({esc}) Tj")
            content_lines.append("T*")
        # Footer page number.
        content_lines.append(f"0 {-leading} Td")
        content_lines.append(f"(Page {page_index}/{len(pages)}) Tj")
        content_lines.append("ET")
        content_stream = ("\n".join(content_lines) + "\n").encode("latin-1", "replace")
        content_obj = b"<< /Length " + str(len(content_stream)).encode() + b" >>\nstream\n" + content_stream + b"endstream"
        content_id = add_obj(content_obj)
        content_obj_ids.append(content_id)

        # Page object
        page_obj = (
            b"<< /Type /Page /Parent 2 0 R "
            b"/MediaBox [0 0 "
            + str(page_w).encode()
            + b" "
            + str(page_h).encode()
            + b"] "
            b"/Resources << /Font << /F1 "
            + str(font_id).encode()
            + b" 0 R >> >> "
            b"/Contents "
            + str(content_id).encode()
            + b" 0 R >>"
        )
        page_id = add_obj(page_obj)
        page_obj_ids.append(page_id)

    # Patch Pages object with Kids/Count.
    kids = b"[ " + b" ".join(f"{pid} 0 R".encode() for pid in page_obj_ids) + b" ]"
    pages_obj = b"<< /Type /Pages /Kids " + kids + b" /Count " + str(len(page_obj_ids)).encode() + b" >>"
    objects[pages_id - 1] = pages_obj

    # Build final PDF.
    out = bytearray()
    out.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    offsets: list[int] = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode())
        out.extend(obj)
        out.extend(b"\nendobj\n")

    xref_start = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode())

    out.extend(
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root "
        + str(catalog_id).encode()
        + b" 0 R >>\nstartxref\n"
        + str(xref_start).encode()
        + b"\n%%EOF\n"
    )

    out_path.write_bytes(bytes(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-md", type=Path, required=True)
    ap.add_argument("--out-pdf", type=Path, required=True)
    args = ap.parse_args()

    md = args.in_md.read_text(encoding="utf-8")
    write_simple_pdf(md, args.out_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

