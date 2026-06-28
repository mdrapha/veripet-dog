#!/usr/bin/env python3
"""Renderiza SVGs standalone para PNG/PDF usando Chrome headless do Windows.

Este script evita depender de matplotlib/Pillow/Inkscape dentro do WSL. Ele
mantem o SVG como fonte editavel e usa o Chrome apenas na etapa de exportacao.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHROME = Path("/mnt/c/Program Files/Google/Chrome/Application/chrome.exe")
TMP = ROOT / "figuras_refeitas_tcc_prism_v2" / "_render_tmp"


def wsl_to_win(path: Path) -> str:
    out = subprocess.check_output(["wslpath", "-w", str(path)], text=True)
    return out.strip()


def file_url(path: Path) -> str:
    win = wsl_to_win(path).replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", win):
        return "file:///" + win
    return "file://" + win


def parse_size(svg_path: Path) -> tuple[int, int]:
    text = svg_path.read_text(encoding="utf-8", errors="ignore")[:2000]
    viewbox = re.search(r'viewBox=["\']\s*[-\d.]+\s+[-\d.]+\s+([\d.]+)\s+([\d.]+)', text)
    if viewbox:
        return max(1, round(float(viewbox.group(1)))), max(1, round(float(viewbox.group(2))))
    width = re.search(r'width=["\']([\d.]+)', text)
    height = re.search(r'height=["\']([\d.]+)', text)
    if width and height:
        return max(1, round(float(width.group(1)))), max(1, round(float(height.group(1))))
    return 2400, 1400


def make_wrapper(svg_path: Path, width: int, height: int, html_path: Path) -> None:
    html_path.write_text(
        f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <style>
    @page {{ size: {width}px {height}px; margin: 0; }}
    html, body {{
      margin: 0;
      width: {width}px;
      height: {height}px;
      background: white;
      overflow: hidden;
    }}
    img {{
      display: block;
      width: {width}px;
      height: {height}px;
    }}
  </style>
</head>
<body><img src="{file_url(svg_path)}" alt=""></body>
</html>
""",
        encoding="utf-8",
    )


def run_chrome(args: list[str]) -> None:
    if not CHROME.exists():
        raise SystemExit(f"Chrome nao encontrado em {CHROME}")
    subprocess.run([str(CHROME), *args], check=True)


def render(svg_path: Path, png: bool, pdf: bool, scale: float) -> None:
    svg_path = svg_path.resolve()
    width, height = parse_size(svg_path)
    out_w = max(1800, round(width * scale))
    out_h = max(1, round(height * (out_w / width)))
    TMP.mkdir(parents=True, exist_ok=True)
    wrapper = TMP / f"{svg_path.stem}.html"
    make_wrapper(svg_path, out_w, out_h, wrapper)
    url = file_url(wrapper)

    if png:
        out = svg_path.with_suffix(".png")
        run_chrome(
            [
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--window-size={out_w},{out_h}",
                f"--screenshot={wsl_to_win(out)}",
                url,
            ]
        )
        print(out)
    if pdf:
        out = svg_path.with_suffix(".pdf")
        run_chrome(
            [
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--window-size={out_w},{out_h}",
                f"--print-to-pdf={wsl_to_win(out)}",
                url,
            ]
        )
        print(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--png", action="store_true", default=False)
    parser.add_argument("--pdf", action="store_true", default=False)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    png = args.png or not args.pdf
    pdf = args.pdf
    for path in args.paths:
        render(path, png=png, pdf=pdf, scale=args.scale)


if __name__ == "__main__":
    main()
