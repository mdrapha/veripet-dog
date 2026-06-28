#!/usr/bin/env python3
"""Gera figuras quantitativas v2 em SVG standalone.

O script usa apenas a biblioteca padrão e escreve exclusivamente em
figuras_refeitas_tcc_prism_v2/quant/.
"""

from __future__ import annotations

import csv
import math
import statistics
import textwrap
from collections import Counter
from html import escape
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figuras_refeitas_tcc_prism_v2" / "quant"

TRAIN_CSV = ROOT / "data" / "datasets" / "original" / "split" / "train.csv"
DOG_CSV = ROOT / "data" / "datasets" / "original" / "dog.csv"
METRICS_CSV = ROOT / "classification_exp" / "figures" / "results" / "metricas_por_raca_stanford.csv"
FAILURES_DIR = ROOT / "classification_exp" / "figures" / "results" / "analise_falhas_top10"
WORST_BREEDS_CSV = FAILURES_DIR / "top10_piores_racas.csv"
CONFUSIONS_CSV = FAILURES_DIR / "distribuicao_erros_top10.csv"
VERIFICATION_CSV = ROOT / "classification_exp" / "figures" / "verificacao" / "verification_results.csv"

BLUE = "#355C9A"
LIGHT_BLUE = "#6E8FC7"
GREEN = "#4C956C"
ORANGE = "#D17B0F"
RED = "#B24A4A"
TEXT = "#333333"
GRID = "#BFC5CE"
BG = "#EEF1F5"
WHITE = "#FFFFFF"


def fmt(value: float | int) -> str:
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def attrs(**values: object) -> str:
    parts: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        key = key.rstrip("_").replace("_", "-")
        parts.append(f'{key}="{escape(str(value), quote=True)}"')
    return " ".join(parts)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def format_percent(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",") + "%"


def format_number(value: int | float) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{int(value):,}".replace(",", ".")


def clean_breed(name: str, max_chars: int | None = None) -> str:
    text = " ".join(name.replace("_", " ").split())
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "..."
    return text


def wrap_label(text: str, width: int, max_lines: int = 2) -> list[str]:
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    if not lines:
        return [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + "..."
    return lines


def nice_ceiling(value: float) -> int:
    if value <= 0:
        return 1
    magnitude = 10 ** math.floor(math.log10(value))
    scaled = value / magnitude
    if scaled <= 1:
        nice = 1
    elif scaled <= 2:
        nice = 2
    elif scaled <= 5:
        nice = 5
    else:
        nice = 10
    return int(nice * magnitude)


def linear_ticks(max_value: float, count: int = 5) -> list[int]:
    top = nice_ceiling(max_value)
    step = nice_ceiling(top / count)
    top = int(math.ceil(max_value / step) * step)
    return list(range(0, top + step, step))


class Svg:
    def __init__(self, width: int, height: int, mm_width: int, title: str, desc: str) -> None:
        self.width = width
        self.height = height
        self.mm_width = mm_width
        self.mm_height = round(mm_width * height / width)
        self.title = title
        self.desc = desc
        self.items: list[str] = []

    def add(self, value: str) -> None:
        self.items.append(value)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = WHITE,
        stroke: str = "none",
        sw: float = 0,
        rx: float = 0,
        opacity: float | None = None,
        extra: str = "",
    ) -> None:
        self.add(
            f"<rect {attrs(x=fmt(x), y=fmt(y), width=fmt(w), height=fmt(h), rx=fmt(rx), fill=fill, stroke=stroke, stroke_width=fmt(sw), opacity=opacity)}{extra}/>"
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = GRID,
        sw: float = 1.2,
        dash: str | None = None,
        opacity: float | None = None,
    ) -> None:
        self.add(
            f"<line {attrs(x1=fmt(x1), y1=fmt(y1), x2=fmt(x2), y2=fmt(y2), stroke=stroke, stroke_width=fmt(sw), stroke_dasharray=dash, opacity=opacity)} />"
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: int = 16,
        weight: int = 400,
        fill: str = TEXT,
        anchor: str = "middle",
        transform: str | None = None,
        extra: str = "",
    ) -> None:
        self.add(
            f"<text {attrs(x=fmt(x), y=fmt(y), font_size=size, font_weight=weight, fill=fill, text_anchor=anchor, transform=transform)}{extra}>{escape(value)}</text>"
        )

    def text_lines(
        self,
        x: float,
        y: float,
        lines: Sequence[str],
        size: int = 16,
        weight: int = 400,
        fill: str = TEXT,
        anchor: str = "middle",
        line_gap: float = 20,
    ) -> None:
        start = y - (len(lines) - 1) * line_gap / 2
        spans = []
        for idx, line in enumerate(lines):
            dy = "0" if idx == 0 else fmt(line_gap)
            spans.append(f"<tspan {attrs(x=fmt(x), dy=dy)}>{escape(line)}</tspan>")
        self.add(
            f"<text {attrs(x=fmt(x), y=fmt(start), font_size=size, font_weight=weight, fill=fill, text_anchor=anchor)}>{''.join(spans)}</text>"
        )

    def save(self, name: str) -> Path:
        OUT.mkdir(parents=True, exist_ok=True)
        style = f"""
  <style>
    svg {{
      background: {WHITE};
      color: {TEXT};
      font-family: "DejaVu Sans", "Liberation Sans", Arial, sans-serif;
      font-kerning: normal;
      letter-spacing: 0;
    }}
    text {{
      dominant-baseline: alphabetic;
    }}
    line {{
      stroke-linecap: round;
    }}
  </style>""".rstrip()
        content = "\n  ".join(self.items)
        output = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{self.mm_width}mm" height="{self.mm_height}mm" viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="{name}-title {name}-desc">
  <title id="{name}-title">{escape(self.title)}</title>
  <desc id="{name}-desc">{escape(self.desc)}</desc>
{style}
  <rect width="100%" height="100%" fill="{WHITE}"/>
  {content}
</svg>
"""
        path = OUT / f"{name}.svg"
        path.write_text(output, encoding="utf-8")
        return path


def y_from_value(value: float, y0: float, height: float, max_value: float) -> float:
    return y0 + height - (value / max_value) * height


def x_from_value(value: float, x0: float, width: float, max_value: float) -> float:
    return x0 + (value / max_value) * width


def draw_y_grid(svg: Svg, x0: float, y0: float, width: float, height: float, ticks: Sequence[int], max_value: int) -> None:
    for tick in ticks:
        y = y_from_value(tick, y0, height, max_value)
        svg.line(x0, y, x0 + width, y, stroke=GRID, sw=0.9, opacity=0.65)
        svg.text(x0 - 16, y + 5, format_number(tick), size=14, fill=TEXT, anchor="end")
    svg.line(x0, y0 + height, x0 + width, y0 + height, stroke=TEXT, sw=1.5)
    svg.line(x0, y0, x0, y0 + height, stroke=TEXT, sw=1.5)


def draw_x_grid(svg: Svg, x0: float, y0: float, width: float, height: float, ticks: Sequence[int], max_value: int) -> None:
    for tick in ticks:
        x = x_from_value(tick, x0, width, max_value)
        svg.line(x, y0, x, y0 + height, stroke=GRID, sw=0.9, opacity=0.65)
        svg.text(x, y0 + height + 28, format_number(tick), size=14, fill=TEXT)
    svg.line(x0, y0 + height, x0 + width, y0 + height, stroke=TEXT, sw=1.5)


def identity_counts() -> tuple[Counter[int], Counter[str]]:
    train_rows = read_csv(TRAIN_CSV)
    dog_rows = read_csv(DOG_CSV)
    images_by_identity = Counter(int(row["label"]) for row in train_rows)
    breed_by_label = {
        idx: (row.get("Breed") or "Unknown").strip() or "Unknown"
        for idx, row in enumerate(dog_rows)
    }
    identities_by_breed: Counter[str] = Counter()
    for label in images_by_identity:
        breed = breed_by_label.get(label, "Unknown")
        if breed.lower() != "unknown":
            identities_by_breed[breed] += 1
    return Counter(images_by_identity.values()), identities_by_breed


def generate_eda_distributions() -> Path:
    images_hist, breed_counts = identity_counts()
    low_keys = [2, 3, 4, 5, 6]
    categories: list[tuple[str, int]] = [(str(key), images_hist.get(key, 0)) for key in low_keys]
    categories.append(("7+", sum(count for key, count in images_hist.items() if key >= 7)))
    top_breeds = breed_counts.most_common(15)

    svg = Svg(
        1700,
        850,
        190,
        "Distribuições exploratórias do conjunto PetFace",
        "Painéis com a distribuição de imagens por identidade e as raças com mais indivíduos no treino.",
    )

    x0, y0, pw, ph = 110, 130, 640, 520
    y_ticks = linear_ticks(max(count for _, count in categories), 5)
    y_max = y_ticks[-1]
    draw_y_grid(svg, x0, y0, pw, ph, y_ticks, y_max)
    step = pw / len(categories)
    bar_w = step * 0.62
    for idx, (label, count) in enumerate(categories):
        x = x0 + idx * step + (step - bar_w) / 2
        bar_h = count / y_max * ph
        svg.rect(x, y0 + ph - bar_h, bar_w, bar_h, fill=BLUE)
        svg.text(x + bar_w / 2, y0 + ph + 32, label, size=15)
        if count > 0:
            y_label = max(y0 + ph - bar_h - 10, y0 + 18)
            svg.text(x + bar_w / 2, y_label, format_number(count), size=13, fill=TEXT)
    svg.text(x0 + pw / 2, 82, "(a) Imagens por identidade", size=21, weight=700)
    svg.text(x0 + pw / 2, 725, "Imagens por identidade", size=18, weight=700)
    svg.text(
        38,
        y0 + ph / 2,
        "Número de identidades",
        size=18,
        weight=700,
        transform=f"rotate(-90 38 {fmt(y0 + ph / 2)})",
    )

    x1, y1, pw2, ph2 = 1115, 110, 480, 575
    max_breed_count = max(count for _, count in top_breeds)
    x_ticks = linear_ticks(max_breed_count, 5)
    x_max = x_ticks[-1]
    draw_x_grid(svg, x1, y1, pw2, ph2, x_ticks, x_max)
    step2 = ph2 / len(top_breeds)
    bar_h2 = step2 * 0.62
    for idx, (breed, count) in enumerate(top_breeds):
        y = y1 + idx * step2 + (step2 - bar_h2) / 2
        bw = count / x_max * pw2
        label_y = y + bar_h2 * 0.66
        svg.text(x1 - 18, label_y, clean_breed(breed, 32), size=14, anchor="end")
        svg.rect(x1, y, bw, bar_h2, fill=GREEN)
        label_x = min(x1 + bw + 10, x1 + pw2 - 2)
        anchor = "start" if label_x < x1 + pw2 - 2 else "end"
        svg.text(label_x, label_y, format_number(count), size=13, fill=TEXT, anchor=anchor)
    svg.text(x1 + pw2 / 2, 82, "(b) Raças mais frequentes", size=21, weight=700)
    svg.text(x1 + pw2 / 2, 760, "Número de indivíduos", size=18, weight=700)
    svg.text(835, y1 + ph2 / 2, "Raça", size=18, weight=700)

    total_identities = sum(images_hist.values())
    total_breeds = sum(breed_counts.values())
    svg.text(110, 806, f"Identidades no treino: {format_number(total_identities)}", size=14, fill=TEXT, anchor="start")
    svg.text(1115, 806, f"Identidades com raça conhecida: {format_number(total_breeds)}", size=14, fill=TEXT, anchor="start")
    return svg.save("eda_distributions")


def generate_accuracy_distribution() -> Path:
    rows = read_csv(METRICS_CSV)
    accuracies = [float(row["Recall"]) * 100 for row in rows]
    start = math.floor(min(accuracies) / 5) * 5
    end = 100
    bins = list(range(int(start), end + 5, 5))
    counts = [0 for _ in bins[:-1]]
    for value in accuracies:
        for idx in range(len(bins) - 1):
            last = idx == len(bins) - 2
            if bins[idx] <= value < bins[idx + 1] or (last and value <= bins[idx + 1]):
                counts[idx] += 1
                break

    median = statistics.median(accuracies)
    mean = sum(accuracies) / len(accuracies)
    svg = Svg(
        1200,
        760,
        180,
        "Distribuição de acurácias por raça",
        "Histograma do recall por raça no classificador Stanford Dogs, com marcação da mediana.",
    )

    x0, y0, pw, ph = 105, 100, 980, 500
    y_ticks = linear_ticks(max(counts), 5)
    y_max = y_ticks[-1]
    draw_y_grid(svg, x0, y0, pw, ph, y_ticks, y_max)

    interval = pw / len(counts)
    bar_w = interval * 0.78
    for idx, count in enumerate(counts):
        x = x0 + idx * interval + (interval - bar_w) / 2
        bh = count / y_max * ph
        svg.rect(x, y0 + ph - bh, bar_w, bh, fill=BLUE)
    for tick in bins:
        x = x0 + (tick - bins[0]) / (bins[-1] - bins[0]) * pw
        svg.text(x, y0 + ph + 30, str(tick), size=14)

    median_x = x0 + (median - bins[0]) / (bins[-1] - bins[0]) * pw
    svg.line(median_x, y0, median_x, y0 + ph, stroke=ORANGE, sw=2.3, dash="7 6")
    svg.text(median_x + 9, y0 + 28, f"Mediana {format_percent(median)}", size=15, weight=700, fill=ORANGE, anchor="start")
    svg.text(x0 + pw - 6, y0 + 58, f"Média {format_percent(mean)}", size=15, fill=TEXT, anchor="end")
    svg.text(x0 + pw / 2, 690, "Acurácia por raça (%)", size=19, weight=700)
    svg.text(
        38,
        y0 + ph / 2,
        "Número de raças",
        size=19,
        weight=700,
        transform=f"rotate(-90 38 {fmt(y0 + ph / 2)})",
    )
    svg.text(x0 + pw, 72, f"n = {format_number(len(accuracies))} raças", size=14, fill=TEXT, anchor="end")
    return svg.save("distribuicao_acuracias")


def generate_top10_worst_breeds() -> Path:
    rows = read_csv(WORST_BREEDS_CSV)[:10]
    data = [
        (clean_breed(row["breed"]), float(row["accuracy"]) * 100, int(row["support"]), int(row["errors"]))
        for row in rows
    ]
    data.sort(key=lambda item: item[1])

    svg = Svg(
        1320,
        780,
        180,
        "Dez raças com menor acurácia",
        "Barras horizontais com as dez menores acurácias por raça e o suporte de cada raça.",
    )
    x0, y0, pw, ph = 390, 95, 780, 540
    ticks = list(range(0, 101, 20))
    draw_x_grid(svg, x0, y0, pw, ph, ticks, 100)
    step = ph / len(data)
    bar_h = step * 0.56
    for idx, (breed, accuracy, support, errors) in enumerate(data):
        y = y0 + idx * step + (step - bar_h) / 2
        label_y = y + bar_h * 0.66
        svg.text(x0 - 20, label_y, breed, size=15, anchor="end")
        svg.rect(x0, y, pw, bar_h, fill=BG)
        color = RED if accuracy < 70 else BLUE
        bw = accuracy / 100 * pw
        svg.rect(x0, y, bw, bar_h, fill=color)
        svg.text(
            min(x0 + bw + 12, x0 + pw - 4),
            label_y,
            f"{format_percent(accuracy)}  n={support}  erros={errors}",
            size=14,
            anchor="start" if bw < pw * 0.83 else "end",
        )
    svg.text(x0 + pw / 2, 716, "Acurácia (%)", size=19, weight=700)
    svg.text(154, y0 + ph / 2, "Raça", size=19, weight=700)
    return svg.save("top10_piores_racas")


def generate_main_confusions() -> Path:
    rows = read_csv(CONFUSIONS_CSV)
    data = sorted(
        rows,
        key=lambda row: (-int(row["count"]), row["true_breed"].lower(), row["pred_breed"].lower()),
    )[:12]
    max_count = max(int(row["count"]) for row in data)
    ticks = list(range(0, max_count + 1))

    svg = Svg(
        1500,
        840,
        190,
        "Principais confusões de classificação",
        "Barras horizontais com os pares raça real e raça predita mais frequentes entre os erros das dez piores raças.",
    )
    x0, y0, pw, ph = 650, 105, 720, 610
    draw_x_grid(svg, x0, y0, pw, ph, ticks, max_count)
    step = ph / len(data)
    bar_h = step * 0.54
    svg.text(235, 74, "Raça real", size=15, weight=700)
    svg.text(450, 74, "Predição", size=15, weight=700)
    for idx, row in enumerate(data):
        count = int(row["count"])
        pct = float(row["pct_errors"])
        y = y0 + idx * step + (step - bar_h) / 2
        label_y = y + bar_h * 0.66
        true_breed = clean_breed(row["true_breed"], 28)
        pred_breed = clean_breed(row["pred_breed"], 28)
        svg.text(310, label_y, true_breed, size=14, anchor="end")
        svg.line(330, label_y - 5, 365, label_y - 5, stroke=GRID, sw=1.3)
        svg.text(383, label_y, pred_breed, size=14, anchor="start")
        svg.rect(x0, y, pw, bar_h, fill=BG)
        bw = count / max_count * pw
        color = ORANGE if count >= 5 else LIGHT_BLUE
        svg.rect(x0, y, bw, bar_h, fill=color)
        svg.text(
            min(x0 + bw + 12, x0 + pw - 4),
            label_y,
            f"{count} erros ({format_percent(pct)})",
            size=14,
            anchor="start" if bw < pw * 0.8 else "end",
        )
    svg.text(x0 + pw / 2, 790, "Contagem de erros", size=19, weight=700)
    svg.text(650, 790, "Percentual entre os erros da raça real.", size=13, fill=TEXT, anchor="start")
    return svg.save("principais_confusoes")


def method_label(raw: str) -> str:
    return raw.split()[0]


def generate_verification_summary() -> Path:
    rows = read_csv(VERIFICATION_CSV)
    metrics = [
        ("Val AUC", "AUC validação"),
        ("Test AUC", "AUC teste"),
        ("Test Accuracy", "Acurácia global"),
        ("Intra-breed Accuracy", "Acurácia intra-raça"),
    ]
    methods = [(method_label(row["Método"]), [float(row[key]) * 100 for key, _ in metrics]) for row in rows]
    colors = [BLUE, GREEN, ORANGE, RED]

    svg = Svg(
        1280,
        760,
        180,
        "Resumo quantitativo de verificação",
        "Comparação entre Softmax e ArcFace em AUC de validação, AUC de teste, acurácia global e acurácia intra-raça.",
    )
    x0, y0, pw, ph = 105, 105, 990, 480
    ticks = list(range(0, 101, 20))
    draw_y_grid(svg, x0, y0, pw, ph, ticks, 100)
    group_w = pw / len(metrics)
    bar_w = 58
    gap = 12
    for metric_idx, (_, label) in enumerate(metrics):
        center = x0 + metric_idx * group_w + group_w / 2
        for method_idx, (_, values) in enumerate(methods):
            value = values[metric_idx]
            x = center + (method_idx - (len(methods) - 1) / 2) * (bar_w + gap) - bar_w / 2
            bh = value / 100 * ph
            color = colors[method_idx % len(colors)]
            bar_top = y0 + ph - bh
            svg.rect(x, bar_top, bar_w, bh, fill=color)
            label_y = bar_top - 10
            label_fill = TEXT
            if label_y < y0 + 18:
                label_y = bar_top + 20
                label_fill = WHITE
            svg.text(x + bar_w / 2, label_y, format_percent(value), size=13, fill=label_fill)
        svg.text_lines(center, y0 + ph + 48, wrap_label(label, 13, max_lines=2), size=15, line_gap=18)
    svg.text(
        38,
        y0 + ph / 2,
        "Valor (%)",
        size=19,
        weight=700,
        transform=f"rotate(-90 38 {fmt(y0 + ph / 2)})",
    )
    legend_x = x0 + pw - 195
    for idx, (method, _) in enumerate(methods):
        y = 52 + idx * 32
        svg.rect(legend_x, y - 15, 22, 16, fill=colors[idx % len(colors)])
        svg.text(legend_x + 34, y - 1, method, size=15, anchor="start")

    threshold_text = " | ".join(
        f"{method_label(row['Método'])}: limiar {float(row['Threshold (from val)']):.3f}".replace(".", ",")
        for row in rows
    )
    pairs = rows[0].get("Intra-breed Pairs (test)", "")
    svg.text(x0, 718, f"{threshold_text} | pares intra-raça no teste: {pairs}", size=13, fill=TEXT, anchor="start")
    return svg.save("verification_summary")


def main() -> None:
    generated = [
        generate_eda_distributions(),
        generate_accuracy_distribution(),
        generate_top10_worst_breeds(),
        generate_main_confusions(),
        generate_verification_summary(),
    ]
    print("Figuras quantitativas geradas:")
    for path in generated:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
