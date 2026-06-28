#!/usr/bin/env python3
"""Gera painéis qualitativos em SVG standalone com imagens embutidas.

Este script usa apenas a biblioteca padrão. Ele não rasteriza imagens, não
depende de Pillow/matplotlib e mantém os JPEGs originais embutidos como data URI
dentro dos SVGs gerados.
"""

from __future__ import annotations

import base64
import csv
import mimetypes
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = ROOT / "classification_exp" / "Images"
RESULTS_DIR = ROOT / "classification_exp" / "figures" / "results"
QUALITATIVE_CSV = (
    RESULTS_DIR
    / "analise_qualitativa_top10"
    / "erros_top10_racas_qualitativo.csv"
)
METRICS_CSV = RESULTS_DIR / "metricas_por_raca_stanford.csv"
OUT_DIR = ROOT / "figuras_refeitas_tcc_prism_v2" / "qualitative"

MOSAIC_CASES = (
    "Eskimo dog",
    "collie",
    "miniature poodle",
    "standard schnauzer",
)

WHITE = "#FFFFFF"
CANVAS = "#F7F9FB"
PANEL = "#FFFFFF"
BORDER = "#D5DCE5"
TEXT = "#1F2A37"
MUTED = "#5B677A"
GREEN = "#2F7D55"
GREEN_LIGHT = "#EAF6EF"
RED = "#A94442"
RED_LIGHT = "#FCEEEE"
BLUE = "#355C9A"


@dataclass(frozen=True)
class Sample:
    image_path: Path
    true_breed: str
    pred_breed: str
    correct: bool
    source: str


def esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def breed_from_image_dir(path: Path) -> str:
    name = path.name.split("-", 1)[1] if "-" in path.name else path.name
    return name.replace("_", " ")


def build_breed_dir_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(IMAGES_DIR.iterdir()):
        if path.is_dir():
            index.setdefault(breed_from_image_dir(path), path)
    return index


def resolve_image_path(raw_path: str) -> Path | None:
    marker = "/Images/"
    if marker in raw_path:
        relative = raw_path.split(marker, 1)[1]
        candidate = IMAGES_DIR / relative
        if candidate.exists():
            return candidate

    filename = Path(raw_path).name
    for candidate in IMAGES_DIR.rglob(filename):
        if candidate.is_file():
            return candidate
    return None


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def wrap_lines(text: str, width: int, max_lines: int | None = None) -> list[str]:
    lines = textwrap.wrap(text, width=width, break_long_words=False)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + "..."
    return lines or [""]


class Svg:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.items: list[str] = []

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = WHITE,
        stroke: str = BORDER,
        sw: float = 1.2,
        rx: float = 0,
    ) -> None:
        self.items.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'rx="{rx:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: int = 18,
        weight: str = "400",
        fill: str = TEXT,
        anchor: str = "start",
    ) -> None:
        self.items.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="DejaVu Sans, Arial, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}">{esc(value)}</text>'
        )

    def multiline(
        self,
        x: float,
        y: float,
        lines: Sequence[str],
        size: int = 15,
        line_gap: float = 18,
        weight: str = "400",
        fill: str = MUTED,
        anchor: str = "start",
    ) -> None:
        for idx, line in enumerate(lines):
            self.text(x, y + idx * line_gap, line, size, weight, fill, anchor)

    def image(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        path: Path,
        stroke: str = BORDER,
    ) -> None:
        self.rect(x, y, w, h, fill="#FAFBFC", stroke=stroke, sw=1.0, rx=4)
        self.items.append(
            f'<image x="{x + 1:.2f}" y="{y + 1:.2f}" width="{w - 2:.2f}" height="{h - 2:.2f}" '
            f'href="{data_uri(path)}" preserveAspectRatio="xMidYMid meet"/>'
        )
        self.rect(x, y, w, h, fill="none", stroke=stroke, sw=1.0, rx=4)

    def save(self, path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    '<?xml version="1.0" encoding="UTF-8"?>',
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
                    f'height="{self.height}" viewBox="0 0 {self.width} {self.height}" '
                    'role="img">',
                    f'<rect width="100%" height="100%" fill="{CANVAS}"/>',
                    *self.items,
                    "</svg>",
                ]
            ),
            encoding="utf-8",
        )


def qualitative_error_samples(rows: Iterable[dict[str, str]]) -> list[Sample]:
    samples: list[Sample] = []
    for row in rows:
        path = resolve_image_path(row["image_path"])
        if path is None:
            continue
        samples.append(
            Sample(
                image_path=path,
                true_breed=row["true_breed"],
                pred_breed=row["pred_breed"],
                correct=False,
                source="CSV de erros qualitativos",
            )
        )
    return samples


def choose_mosaic_samples(samples: Sequence[Sample], breed: str, limit: int = 4) -> list[Sample]:
    candidates = [sample for sample in samples if sample.true_breed == breed]
    counts = Counter(sample.pred_breed for sample in candidates)
    return sorted(
        candidates,
        key=lambda sample: (-counts[sample.pred_breed], sample.pred_breed, sample.image_path.name),
    )[:limit]


def prediction_summary(samples: Sequence[Sample]) -> str:
    counts = Counter(sample.pred_breed for sample in samples)
    parts = [f"{breed} ({count})" for breed, count in counts.most_common()]
    return "Predições mostradas: " + ", ".join(parts)


def draw_mosaic_panel(svg: Svg, x: float, y: float, w: float, h: float, breed: str, samples: Sequence[Sample]) -> None:
    svg.rect(x, y, w, h, fill=PANEL, stroke=BORDER, sw=1.2, rx=8)
    svg.text(x + 24, y + 36, f"Raça real: {breed}", size=22, weight="700", fill=TEXT)
    svg.text(
        x + 24,
        y + 63,
        "Casos classificados incorretamente no conjunto qualitativo",
        size=14,
        fill=MUTED,
    )

    image_w = (w - 66) / 2
    image_h = 158
    cell_gap_x = 18
    cell_gap_y = 32
    start_x = x + 24
    start_y = y + 88

    for idx, sample in enumerate(samples):
        col = idx % 2
        row = idx // 2
        ix = start_x + col * (image_w + cell_gap_x)
        iy = start_y + row * (image_h + cell_gap_y)
        svg.image(ix, iy, image_w, image_h, sample.image_path)
        svg.text(ix, iy + image_h + 20, f"Predição: {sample.pred_breed}", size=13, fill=MUTED)

    summary = prediction_summary(samples)
    footer_lines = wrap_lines(summary, width=62, max_lines=2)
    svg.multiline(x + 24, y + h - 40, footer_lines, size=14, line_gap=17, fill=MUTED)


def generate_mosaicos_erros(samples: Sequence[Sample]) -> Path:
    width, height = 1500, 1330
    svg = Svg(width, height)
    svg.text(54, 54, "Mosaicos de erros qualitativos", size=30, weight="700")
    svg.text(
        54,
        86,
        "Quatro grupos de confusão recorrentes no classificador de raças",
        size=17,
        fill=MUTED,
    )

    margin = 54
    gap = 38
    panel_w = (width - 2 * margin - gap) / 2
    panel_h = 556
    top = 125
    positions = (
        (margin, top),
        (margin + panel_w + gap, top),
        (margin, top + panel_h + gap),
        (margin + panel_w + gap, top + panel_h + gap),
    )

    for (x, y), breed in zip(positions, MOSAIC_CASES):
        chosen = choose_mosaic_samples(samples, breed)
        if len(chosen) < 4:
            raise RuntimeError(f"Poucas imagens resolvidas para o caso: {breed}")
        draw_mosaic_panel(svg, x, y, panel_w, panel_h, breed, chosen)

    out_path = OUT_DIR / "mosaicos_erros_2x2.svg"
    svg.save(out_path)
    return out_path


def correct_samples_from_metrics(limit: int = 6) -> list[Sample]:
    breed_dirs = build_breed_dir_index()
    rows = read_csv(METRICS_CSV)
    rows = sorted(
        rows,
        key=lambda row: (-float(row["Recall"]), -int(row["Support"]), row["Raca"]),
    )

    samples: list[Sample] = []
    for row in rows:
        if float(row["Recall"]) < 1.0:
            continue
        breed = row["Raca"]
        directory = breed_dirs.get(breed)
        if directory is None:
            continue
        images = sorted(directory.glob("*.jpg"))
        if not images:
            continue
        samples.append(
            Sample(
                image_path=images[0],
                true_breed=breed,
                pred_breed=breed,
                correct=True,
                source="Métricas por raça com recall 100%",
            )
        )
        if len(samples) == limit:
            break
    return samples


def incorrect_samples_for_overview(samples: Sequence[Sample], limit: int = 6) -> list[Sample]:
    counts = Counter((sample.true_breed, sample.pred_breed) for sample in samples)
    ordered = sorted(
        samples,
        key=lambda sample: (
            -counts[(sample.true_breed, sample.pred_breed)],
            sample.true_breed,
            sample.pred_breed,
            sample.image_path.name,
        ),
    )

    selected: list[Sample] = []
    used_true_breeds: set[str] = set()
    for sample in ordered:
        if sample.true_breed in used_true_breeds:
            continue
        selected.append(sample)
        used_true_breeds.add(sample.true_breed)
        if len(selected) == limit:
            return selected

    for sample in ordered:
        if sample not in selected:
            selected.append(sample)
            if len(selected) == limit:
                return selected
    return selected


def draw_prediction_card(svg: Svg, x: float, y: float, w: float, h: float, sample: Sample) -> None:
    accent = GREEN if sample.correct else RED
    fill = GREEN_LIGHT if sample.correct else RED_LIGHT
    label = "Correta" if sample.correct else "Incorreta"
    svg.rect(x, y, w, h, fill=PANEL, stroke=BORDER, sw=1.1, rx=8)
    svg.rect(x + 15, y + 18, 86, 28, fill=fill, stroke=accent, sw=1.0, rx=5)
    svg.text(x + 58, y + 38, label, size=13, weight="700", fill=accent, anchor="middle")

    image_x = x + 18
    image_y = y + 58
    image_w = w - 36
    image_h = 154
    svg.image(image_x, image_y, image_w, image_h, sample.image_path, stroke="#CBD3DD")

    lines = []
    lines.extend(wrap_lines(f"Real: {sample.true_breed}", width=36, max_lines=2))
    lines.extend(wrap_lines(f"Predição: {sample.pred_breed}", width=36, max_lines=2))
    svg.multiline(x + 18, y + 238, lines, size=13, line_gap=17, fill=TEXT)


def generate_amostras_predicoes(error_samples: Sequence[Sample]) -> Path | None:
    correct = correct_samples_from_metrics(limit=6)
    incorrect = incorrect_samples_for_overview(error_samples, limit=6)
    if len(correct) < 4 or len(incorrect) < 4:
        return None

    cards = correct + incorrect
    width, height = 1500, 1120
    svg = Svg(width, height)
    svg.text(54, 54, "Amostras de predições", size=30, weight="700")
    svg.text(
        54,
        86,
        "Exemplos corretos e incorretos selecionados de artefatos locais do experimento",
        size=17,
        fill=MUTED,
    )

    cols = 4
    margin_x = 54
    top = 118
    gap_x = 22
    gap_y = 24
    card_w = (width - 2 * margin_x - (cols - 1) * gap_x) / cols
    card_h = 300

    for idx, sample in enumerate(cards):
        row = idx // cols
        col = idx % cols
        x = margin_x + col * (card_w + gap_x)
        y = top + row * (card_h + gap_y)
        draw_prediction_card(svg, x, y, card_w, card_h, sample)

    note = (
        "Corretas: raças com recall 100% nas métricas; "
        "incorretas: linhas do CSV de análise qualitativa."
    )
    svg.text(54, height - 34, note, size=13, fill=MUTED)

    out_path = OUT_DIR / "amostras_predicoes.svg"
    svg.save(out_path)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_csv(QUALITATIVE_CSV)
    error_samples = qualitative_error_samples(rows)

    written: list[Path] = [generate_mosaicos_erros(error_samples)]
    maybe_predictions = generate_amostras_predicoes(error_samples)
    if maybe_predictions is not None:
        written.append(maybe_predictions)

    for path in written:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
