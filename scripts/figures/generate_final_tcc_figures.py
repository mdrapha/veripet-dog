#!/usr/bin/env python3
"""Gera a rodada final das figuras do TCC em pasta plana.

As figuras quantitativas e conceituais são salvas em PNG, PDF e SVG quando
possível. Painéis com fotografias são salvos em PNG e PDF, preservando os
arquivos-fonte e evitando refazer dados não versionados.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-veripet")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figuras_refeitas_tcc_prism_final"
SOURCES = OUT / "_sources"

STANFORD_IMAGES = ROOT / "classification_exp" / "Images"
PETFACE_IMAGES = ROOT / "data" / "local_dataset" / "dog_20k"

METRICS = ROOT / "classification_exp" / "figures" / "results" / "metricas_por_raca_stanford.csv"
INFERENCE = ROOT / "classification_exp" / "figures" / "results" / "inferencia_teste_completa.csv"
WORST = ROOT / "classification_exp" / "figures" / "results" / "analise_falhas_top10" / "top10_piores_racas.csv"
CONFUSIONS = ROOT / "classification_exp" / "figures" / "results" / "analise_falhas_top10" / "distribuicao_erros_top10.csv"
QUAL_ERRORS = ROOT / "classification_exp" / "figures" / "results" / "analise_qualitativa_top10" / "erros_top10_racas_qualitativo.csv"
VERIFICATION_RESULTS = ROOT / "classification_exp" / "figures" / "verificacao" / "verification_results.csv"
RESULTS_COMPARISON_OLD = ROOT / "classification_exp" / "figures" / "verificacao" / "results_comparison.png"
TRAIN_SPLIT = ROOT / "data" / "datasets" / "original" / "split" / "train.csv"
DOG_CSV = ROOT / "data" / "datasets" / "original" / "dog.csv"
VERIFICATION_PAIRS = ROOT / "data" / "datasets" / "original" / "split" / "verification_test_pairs.csv"
CLASSIFICATION_NB = ROOT / "classification_exp" / "classificacao_stanford_dogs (3).ipynb"
VERIFICATION_NB = ROOT / "classification_exp" / "petface_verification.ipynb"

BLUE = "#355C9A"
LIGHT_BLUE = "#6E8FC7"
GREEN = "#4C956C"
ORANGE = "#D17B0F"
RED = "#B24A4A"
TEXT = "#333333"
GRID = "#BFC5CE"
BG = "#EEF1F5"
WHITE = "#FFFFFF"

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCES.mkdir(parents=True, exist_ok=True)
    for src in [
        Path(__file__),
        METRICS,
        INFERENCE,
        WORST,
        CONFUSIONS,
        QUAL_ERRORS,
        VERIFICATION_RESULTS,
    ]:
        if src.exists():
            shutil.copy2(src, SOURCES / src.name)


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": TEXT,
            "axes.linewidth": 0.8,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig(fig: plt.Figure, name: str, *, svg: bool = True, pdf: bool = True) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=320, bbox_inches="tight", pad_inches=0.08)
    if pdf:
        fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.08)
    if svg:
        fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def setup_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.7, alpha=0.45)
    ax.set_axisbelow(True)


def clean_breed(value: str) -> str:
    return " ".join(str(value).replace("_", " ").split())


def wrap(value: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(clean_breed(value), width=width, break_long_words=False))


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    label: str,
    color: str,
    *,
    note: str | None = None,
    fill: str = WHITE,
    lw: float = 1.6,
) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=lw,
        edgecolor=color,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * (0.58 if note else 0.5), label, ha="center", va="center", fontsize=11, weight="bold")
    if note:
        ax.text(x + w / 2, y + h * 0.27, note, ha="center", va="center", fontsize=8.5, color=TEXT)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = TEXT) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.45,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def pipeline_veripet() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.27, 0.56), 0.50, 0.29, boxstyle="round,pad=0.015,rounding_size=0.02", fc="#F7F9FC", ec="#D7DDE6", lw=0.9))
    ax.add_patch(FancyBboxPatch((0.27, 0.16), 0.50, 0.29, boxstyle="round,pad=0.015,rounding_size=0.02", fc="#F7F9FC", ec="#D7DDE6", lw=0.9))
    ax.text(0.29, 0.82, "Classificação de raça", ha="left", va="center", fontsize=9.5, color=BLUE, weight="bold")
    ax.text(0.29, 0.42, "Verificação individual", ha="left", va="center", fontsize=9.5, color=GREEN, weight="bold")

    add_box(ax, (0.04, 0.39), (0.14, 0.20), "Imagem\nde entrada", BLUE, note="consulta")
    add_box(ax, (0.29, 0.63), (0.14, 0.14), "Classificador\nde raça", BLUE)
    add_box(ax, (0.56, 0.63), (0.14, 0.14), "Probabilidades\nde raça", BLUE)
    add_box(ax, (0.29, 0.23), (0.14, 0.14), "Verificador\nindividual", GREEN)
    add_box(ax, (0.56, 0.23), (0.14, 0.14), "Similaridade\nde identidade", GREEN)
    add_box(ax, (0.81, 0.44), (0.14, 0.17), "Combinação\ndos sinais", ORANGE, note="regra de decisão")
    add_box(ax, (0.81, 0.18), (0.14, 0.14), "Decisão\nfinal", RED)

    add_arrow(ax, (0.18, 0.49), (0.27, 0.70), BLUE)
    add_arrow(ax, (0.18, 0.49), (0.27, 0.30), GREEN)
    add_arrow(ax, (0.43, 0.70), (0.56, 0.70), BLUE)
    add_arrow(ax, (0.43, 0.30), (0.56, 0.30), GREEN)
    add_arrow(ax, (0.70, 0.70), (0.81, 0.54), ORANGE)
    add_arrow(ax, (0.70, 0.30), (0.81, 0.51), ORANGE)
    add_arrow(ax, (0.88, 0.44), (0.88, 0.32), ORANGE)

    save_fig(fig, "pipeline_veripet")


def mlp_network() -> None:
    fig, ax = plt.subplots(figsize=(9.8, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xs = [0.14, 0.40, 0.65, 0.88]
    ys = [
        np.linspace(0.28, 0.72, 5),
        np.linspace(0.22, 0.78, 6),
        np.linspace(0.30, 0.70, 4),
        np.linspace(0.40, 0.60, 2),
    ]
    labels = ["Camada de\nentrada", "Camadas\nocultas", "Camadas\nocultas", "Camada de\nsaída"]
    colors = [BLUE, BLUE, LIGHT_BLUE, GREEN]

    for li in range(len(xs) - 1):
        for y1 in ys[li]:
            for y2 in ys[li + 1]:
                ax.plot([xs[li], xs[li + 1]], [y1, y2], color="#D4DAE3", lw=0.75, zorder=1)
    for x, ylist, color, label in zip(xs, ys, colors, labels):
        ax.text(x, 0.91, label, ha="center", va="center", fontsize=10.5, weight="bold", color=color)
        for y in ylist:
            ax.add_patch(plt.Circle((x, y), 0.027, ec=color, fc=WHITE, lw=1.5, zorder=3))
            ax.add_patch(plt.Circle((x, y), 0.0075, ec=color, fc=color, lw=0.5, zorder=4))

    ax.text(0.50, 0.08, "Conexões totalmente conectadas entre camadas sucessivas", ha="center", va="center", fontsize=9, color="#5B677A")
    save_fig(fig, "mlp_network")


def mycnn() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, (0.04, 0.41), (0.12, 0.22), "Imagem\nde entrada", BLUE, note="224 x 224 x 3")
    add_box(ax, (0.25, 0.55), (0.16, 0.18), "Camadas\nconvolucionais", BLUE, note="filtros locais")
    add_box(ax, (0.45, 0.55), (0.16, 0.18), "Extração de\ncaracterísticas", LIGHT_BLUE, note="padrões visuais")
    add_box(ax, (0.25, 0.23), (0.36, 0.14), "Representações progressivamente mais abstratas", LIGHT_BLUE, fill="#F7F9FC")
    add_box(ax, (0.69, 0.43), (0.14, 0.20), "Camadas\ntotalmente\nconectadas", GREEN)
    add_box(ax, (0.88, 0.43), (0.10, 0.20), "Saída da\nclassificação", GREEN, note="classe")

    for sx, ex, y in [(0.16, 0.25, 0.52), (0.41, 0.45, 0.64), (0.61, 0.69, 0.53), (0.83, 0.88, 0.53)]:
        add_arrow(ax, (sx, y), (ex, y), "#9AA4B2")

    # Blocos empilhados discretos para sugerir mapas de características.
    for base_x, base_y, color in [(0.26, 0.76, BLUE), (0.46, 0.76, LIGHT_BLUE)]:
        for i in range(4):
            ax.add_patch(Rectangle((base_x + i * 0.011, base_y + i * 0.01), 0.07, 0.05, ec=color, fc="#EEF3FA", lw=0.8))
    ax.plot([0.27, 0.59], [0.30, 0.30], color=LIGHT_BLUE, lw=1.0, alpha=0.75)
    for x in np.linspace(0.29, 0.57, 4):
        ax.scatter([x], [0.30], s=18, color=LIGHT_BLUE)

    save_fig(fig, "mycnn")


def parse_classification_history() -> pd.DataFrame:
    data = json.loads(CLASSIFICATION_NB.read_text(encoding="utf-8"))
    lines: list[str] = []
    for out in data["cells"][16].get("outputs", []):
        if "text" in out:
            lines.append("".join(out["text"]) if isinstance(out["text"], list) else out["text"])
    rows = []
    pattern = re.compile(r"Ep\s+(\d+).*?Train L:(\d+\.\d+) A:(\d+\.\d+).*?Val L:(\d+\.\d+) A:(\d+\.\d+)")
    for line in "\n".join(lines).splitlines():
        match = pattern.search(line)
        if match:
            e, tl, ta, vl, va = match.groups()
            rows.append({"epoch": int(e), "train_loss": float(tl), "train_acc": float(ta), "val_loss": float(vl), "val_acc": float(va)})
    return pd.DataFrame(rows)


def curvas_treinamento() -> None:
    df = parse_classification_history()
    best_epoch = int(df.loc[df["val_loss"].idxmin(), "epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharex=True)
    axes[0].plot(df["epoch"], df["train_loss"], color=BLUE, lw=2.0, label="Treino")
    axes[0].plot(df["epoch"], df["val_loss"], color=ORANGE, lw=2.0, label="Validação")
    axes[0].axvline(best_epoch, color=RED, lw=1.3, ls="--", label="Melhor época")
    axes[0].set_title("(a) Perda")
    axes[0].set_ylabel("Perda")
    axes[1].plot(df["epoch"], df["train_acc"] * 100, color=BLUE, lw=2.0, label="Treino")
    axes[1].plot(df["epoch"], df["val_acc"] * 100, color=ORANGE, lw=2.0, label="Validação")
    axes[1].axvline(best_epoch, color=RED, lw=1.3, ls="--", label="Melhor época")
    axes[1].set_title("(b) Acurácia")
    axes[1].set_ylabel("Acurácia (%)")
    for ax in axes:
        ax.set_xlabel("Época")
        setup_axis(ax)
        ax.legend(frameon=False, fontsize=8.8)
    save_fig(fig, "curvas_treinamento")


def eda_distributions() -> None:
    for suffix in ("png", "pdf", "svg"):
        stale = OUT / f"eda_distributions.{suffix}"
        if stale.exists():
            stale.unlink()

    train = pd.read_csv(TRAIN_SPLIT)
    dog = pd.read_csv(DOG_CSV)
    per_id = train.groupby("label").size()
    bins = pd.cut(per_id, bins=[1, 2, 3, 4, 5, 6, np.inf], labels=["2", "3", "4", "5", "6", "7+"], right=True)
    counts = bins.value_counts().sort_index()

    breed_counts = dog.loc[train["label"].unique(), "Breed"].value_counts().head(20).sort_values()

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.bar(counts.index.astype(str), counts.values, color=BLUE, width=0.62)
    ax.set_title("(a) Imagens por identidade")
    ax.set_xlabel("Imagens por identidade")
    ax.set_ylabel("Número de identidades")
    setup_axis(ax)
    save_fig(fig, "eda_distributions_a")

    colors = ["#9AA4B2" if b in {"Mixed Breed", "Unknown"} else GREEN for b in breed_counts.index]
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.barh([clean_breed(x) for x in breed_counts.index], breed_counts.values, color=colors, height=0.72)
    ax.set_xscale("log")
    ax.set_title("(b) Raças mais frequentes")
    ax.set_xlabel("Número de indivíduos (escala log)")
    ax.set_ylabel("Raça")
    setup_axis(ax, grid_axis="x")
    for y, value in enumerate(breed_counts.values):
        ax.text(value * 1.08, y, f"{int(value):,}".replace(",", "."), va="center", fontsize=8.2)
    save_fig(fig, "eda_distributions_b")


def distribuicao_acuracias() -> None:
    df = pd.read_csv(METRICS)
    recall_col = "recall" if "recall" in df.columns else "Recall"
    values = df[recall_col].to_numpy() * 100
    median = float(np.median(values))
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.hist(values, bins=np.arange(25, 105, 5), color=BLUE, alpha=0.82, edgecolor=WHITE, linewidth=0.9)
    ax.axvline(median, color=ORANGE, lw=2.0, ls="--")
    ax.text(median + 1.1, ax.get_ylim()[1] * 0.88, f"Mediana: {median:.1f}%".replace(".", ","), color=ORANGE, weight="bold", fontsize=9)
    ax.set_xlabel("Acurácia por raça (%)")
    ax.set_ylabel("Número de raças")
    setup_axis(ax)
    save_fig(fig, "distribuicao_acuracias")


def top10_piores_racas() -> None:
    df = pd.read_csv(WORST).sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    ax.barh([clean_breed(x) for x in df["breed"]], df["accuracy"] * 100, color=BLUE, alpha=0.88)
    ax.set_xlabel("Acurácia (%)")
    ax.set_ylabel("Raça")
    setup_axis(ax, grid_axis="x")
    for i, row in enumerate(df.itertuples()):
        ax.text(row.accuracy * 100 + 1.2, i, f"n={row.support}", va="center", fontsize=8.2, color="#5B677A")
    ax.set_xlim(0, 100)
    save_fig(fig, "top10_piores_racas")


def principais_confusoes() -> None:
    df = pd.read_csv(CONFUSIONS)
    keep = pd.read_csv(WORST).head(6)["breed"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.4))
    for idx, (ax, breed) in enumerate(zip(axes.flat, keep)):
        sub = df[df["true_breed"] == breed].head(4).sort_values("pct_errors")
        ax.barh([clean_breed(x) for x in sub["pred_breed"]], sub["pct_errors"], color=GREEN, alpha=0.9)
        ax.set_title(wrap(breed, 22), fontsize=8.7, pad=8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% dos erros" if idx >= 3 else "")
        setup_axis(ax, grid_axis="x")
        ax.tick_params(labelsize=8)
    fig.subplots_adjust(wspace=0.52, hspace=0.48, bottom=0.10, top=0.91)
    save_fig(fig, "principais_confusoes")


def verification_summary() -> None:
    df = pd.read_csv(VERIFICATION_RESULTS)
    metrics = ["Test AUC", "Test Accuracy", "Intra-breed Accuracy"]
    labels = ["AUC teste", "Acurácia teste", "Acurácia intra-raça"]
    x = np.arange(len(metrics))
    width = 0.33
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for i, (method, color) in enumerate(zip(df["Método"], [BLUE, GREEN])):
        vals = df.loc[df["Método"] == method, metrics].iloc[0].to_numpy(dtype=float) * 100
        bars = ax.bar(x + (i - 0.5) * width, vals, width=width, label=method.replace(" (ConvNeXt-S)", ""), color=color, alpha=0.9)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.1f}%".replace(".", ","), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Valor (%)")
    ax.set_ylim(75, 101)
    setup_axis(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.subplots_adjust(bottom=0.24)
    save_fig(fig, "verification_summary")


def parse_verification_history() -> pd.DataFrame:
    data = json.loads(VERIFICATION_NB.read_text(encoding="utf-8"))
    rows = []
    pattern = re.compile(r"Epoch\s+(\d+)/(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Train Acc:\s+([\d.]+)\s+\|\s+Verif AUC:\s+([\d.]+)")
    for cell_id, method in [(14, "Softmax"), (16, "ArcFace")]:
        text = []
        for out in data["cells"][cell_id].get("outputs", []):
            if "text" in out:
                text.append("".join(out["text"]) if isinstance(out["text"], list) else out["text"])
        for line in "\n".join(text).splitlines():
            match = pattern.search(line)
            if match:
                epoch, _, loss, acc, auc = match.groups()
                rows.append({"método": method, "epoch": int(epoch), "loss": float(loss), "acc": float(acc), "auc": float(auc)})
    return pd.DataFrame(rows)


def training_curves() -> None:
    df = parse_verification_history()
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6))
    specs = [("loss", "Perda de treino", "Perda"), ("acc", "Acurácia de treino", "Acurácia (%)"), ("auc", "AUC de validação", "AUC")]
    for ax, (col, title, ylabel) in zip(axes, specs):
        for method, color in [("Softmax", BLUE), ("ArcFace", GREEN)]:
            sub = df[df["método"] == method]
            y = sub[col] * (100 if col == "acc" else 1)
            ax.plot(sub["epoch"], y, marker="o", ms=4, lw=2, color=color, label=method)
        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel)
        setup_axis(ax)
        ax.legend(frameon=False, fontsize=8.5)
    save_fig(fig, "training_curves")


def resolve_stanford(path_text: str) -> Path | None:
    marker = "/Images/"
    if marker in path_text:
        candidate = STANFORD_IMAGES / path_text.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    name = Path(path_text).name
    hits = list(STANFORD_IMAGES.rglob(name))
    return hits[0] if hits else None


def load_square(path: Path, size: tuple[int, int] = (360, 280)) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return ImageOps.fit(img, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def add_image_card(ax: plt.Axes, img: Image.Image, title: str, lines: list[str], color: str) -> None:
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor(color)
    ax.set_title(title, fontsize=8.5, weight="bold", color=color, pad=4)
    ax.text(0.5, -0.10, "\n".join(lines), transform=ax.transAxes, ha="center", va="top", fontsize=7.4, linespacing=1.15)


def amostras_predicoes() -> None:
    df = pd.read_csv(INFERENCE)
    df["img"] = df["image_path"].map(resolve_stanford)
    df = df[df["img"].notna()].copy()
    correct = df[df["correct_breed"].astype(str) == "True"].sort_values("top1_confidence", ascending=False).drop_duplicates("true_breed").head(6)
    incorrect = df[df["correct_breed"].astype(str) == "False"].sort_values("top1_confidence", ascending=False).drop_duplicates("true_breed").head(6)

    fig, axes = plt.subplots(2, 6, figsize=(12.4, 4.25))
    for row_idx, (subset, title, color) in enumerate([(correct, "Exemplos corretos", GREEN), (incorrect, "Exemplos incorretos", RED)]):
        for col_idx, (_, row) in enumerate(subset.iterrows()):
            img = load_square(row["img"])
            confidence = f"{row['top1_confidence'] * 100:.1f}".replace(".", ",")
            lines = [f"Real: {wrap(row['true_breed'], 18)}", f"Predita: {wrap(row['pred_breed'], 18)}", f"Confiança: {confidence}%"]
            add_image_card(axes[row_idx, col_idx], img, title if col_idx == 0 else "", lines, color)
    fig.subplots_adjust(wspace=0.22, hspace=0.36, bottom=0.13, top=0.91)
    save_fig(fig, "amostras_predicoes", svg=False)


def pil_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.FreeTypeFont, fill: str = TEXT, anchor: str | None = None) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def make_case_tile(case: str, rows: pd.DataFrame, title: str) -> Image.Image:
    thumb = (250, 170)
    pad = 16
    title_h = 62
    label_h = 44
    tile = Image.new("RGB", (2 * thumb[0] + 3 * pad, title_h + 2 * (thumb[1] + label_h) + 3 * pad), WHITE)
    draw = ImageDraw.Draw(tile)
    font_b = ImageFont.truetype(BOLD, 15)
    font_s = ImageFont.truetype(FONT, 12)
    for line_i, line in enumerate(textwrap.wrap(title, width=50, break_long_words=False)):
        pil_text(draw, (pad, 22 + line_i * 18), line, font_b, TEXT)
    for i, (_, row) in enumerate(rows.head(4).iterrows()):
        path = resolve_stanford(row["image_path"])
        if not path:
            continue
        img = load_square(path, thumb)
        x = pad + (i % 2) * (thumb[0] + pad)
        y = title_h + pad + (i // 2) * (thumb[1] + label_h + pad)
        tile.paste(img, (x, y))
        draw.rectangle([x, y, x + thumb[0], y + thumb[1]], outline="#D5DCE5", width=2)
        label = f"Real: {clean_breed(row['true_breed'])}\nPredita: {clean_breed(row['pred_breed'])}"
        pil_text(draw, (x, y + thumb[1] + 15), label, font_s, "#4B5563")
    return tile


def mosaicos_erros_2x2() -> None:
    df = pd.read_csv(QUAL_ERRORS)
    cases = [
        ("Eskimo dog", "(a) Eskimo dog confundido com Siberian husky"),
        ("collie", "(b) Collie confundido com Border collie"),
        ("miniature poodle", "(c) Miniature poodle confundido com variantes de poodle"),
        ("standard schnauzer", "(d) Standard schnauzer confundido com miniature schnauzer"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2))
    for ax, (case, title) in zip(axes.flat, cases):
        sub = df[df["true_breed"] == case]
        tile = make_case_tile(case, sub, title)
        ax.imshow(tile)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.03, hspace=0.10)
    save_fig(fig, "mosaicos_erros_2x2", svg=False)


def results_comparison() -> None:
    # Dados brutos de ROC/similaridade não estão versionados. Esta versão preserva
    # os painéis originais gerados pelo experimento como camada de dados, removendo
    # títulos, eixos e legendas antigas para redesenhar a camada editorial.
    img = Image.open(RESULTS_COMPARISON_OLD).convert("RGB")

    def crop_panel(x0: int, x1: int, y0: int = 49, y1: int = 797) -> np.ndarray:
        return np.asarray(img.crop((x0 + 3, y0 + 3, x1 - 3, y1 - 3)))

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.85))
    specs = [
        {
            "crop": crop_panel(94, 981),
            "extent": (0.0, 1.0, 0.0, 1.0),
            "title": "(a) Curva ROC",
            "xlabel": "Taxa de falso positivo",
            "ylabel": "Taxa de verdadeiro positivo",
            "xlim": (0, 1),
            "ylim": (0, 1),
            "legend": [
                ("Softmax (AUC = 0,9549)", "#4C72B0"),
                ("ArcFace (AUC = 0,9740)", "#DD8452"),
            ],
        },
        {
            "crop": crop_panel(1087, 1974),
            "extent": (-0.3, 1.05, 0.0, 4.0),
            "title": "(b) Softmax",
            "xlabel": "Similaridade do cosseno",
            "ylabel": "Densidade",
            "xlim": (-0.3, 1.05),
            "ylim": (0, 4.0),
            "threshold": 0.5219817161560059,
        },
        {
            "crop": crop_panel(2079, 2966),
            "extent": (-0.2, 1.05, 0.0, 7.8),
            "title": "(c) ArcFace",
            "xlabel": "Similaridade do cosseno",
            "ylabel": "Densidade",
            "xlim": (-0.2, 1.05),
            "ylim": (0, 7.8),
            "threshold": 0.22444689273834229,
        },
    ]
    for ax, spec in zip(axes, specs):
        ax.imshow(spec["crop"], extent=spec["extent"], origin="upper", aspect="auto", zorder=0)
        ax.set_xlim(*spec["xlim"])
        ax.set_ylim(*spec["ylim"])
        ax.set_title(spec["title"], fontsize=10.5, weight="bold", pad=6)
        ax.set_xlabel(spec["xlabel"], fontsize=8.6)
        ax.set_ylabel(spec["ylabel"], fontsize=8.6)
        ax.tick_params(axis="both", labelsize=7.8)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.8)
        ax.add_patch(Rectangle((0.62, 0.84), 0.34, 0.13, transform=ax.transAxes, facecolor=WHITE, edgecolor="none", zorder=4))
        if "legend" in spec:
            handles = [plt.Line2D([], [], color=color, lw=2.0, label=label) for label, color in spec["legend"]]
            ax.legend(handles=handles, loc="lower right", fontsize=7.5, frameon=True, framealpha=0.96, edgecolor="#D5DCE5")
        else:
            threshold = spec["threshold"]
            handles = [
                Patch(facecolor="#8AC09B", edgecolor="none", label="Pares negativos"),
                Patch(facecolor="#F0CF85", edgecolor="none", label="Pares positivos"),
                plt.Line2D([], [], color=RED, lw=1.2, linestyle="--", label=f"Limiar = {threshold:.3f}".replace(".", ",")),
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=7.2, frameon=True, framealpha=0.96, edgecolor="#D5DCE5")
    fig.subplots_adjust(wspace=0.30, left=0.065, right=0.995, bottom=0.18, top=0.87)
    save_fig(fig, "results_comparison", svg=False)


def resolve_petface(path_text: str) -> Path | None:
    rel = path_text
    if rel.startswith("dog/"):
        rel = rel.split("/", 1)[1]
    candidate = PETFACE_IMAGES / rel
    if candidate.exists():
        return candidate
    return None


def prediction_examples_results() -> None:
    df = pd.read_csv(VERIFICATION_PAIRS)

    def rel(path_text: str) -> str:
        return path_text.split("/", 1)[1] if str(path_text).startswith("dog/") else str(path_text)

    def photo_score(path: Path | None) -> float:
        if path is None or not path.exists():
            return -1.0
        try:
            im = Image.open(path).convert("RGB")
        except OSError:
            return -1.0
        if min(im.size) < 170:
            return -1.0
        arr = np.asarray(ImageOps.fit(im, (96, 96), method=Image.Resampling.LANCZOS)).astype(np.float32)
        white = np.mean(np.all(arr > 246, axis=2))
        black = np.mean(np.all(arr < 10, axis=2))
        gray_std = float(arr.mean(axis=2).std())
        chroma = float((arr.max(axis=2) - arr.min(axis=2)).mean())
        if white > 0.55 or black > 0.55 or gray_std < 18:
            return -1.0
        return gray_std + 0.18 * chroma + 0.015 * min(im.size)

    def choose(pair_type: str, n: int) -> list[pd.Series]:
        chosen_rows: list[pd.Series] = []
        pool = df[df["pair_type"] == pair_type].copy()
        if pair_type != "negative_random" and "shared_breed" in pool.columns:
            generic = {"Mixed Breed", "Unknown", "nan", "None"}
            pool = pool[~pool["shared_breed"].astype(str).isin(generic)]
        for _, row in pool.iterrows():
            p1 = resolve_petface(row["filename1"])
            p2 = resolve_petface(row["filename2"])
            score = min(photo_score(p1), photo_score(p2))
            if score < 0:
                continue
            row = row.copy()
            row["_p1"] = p1
            row["_p2"] = p2
            row["_score"] = score
            chosen_rows.append(row)
            if len(chosen_rows) >= n:
                break
        return chosen_rows

    chosen = choose("positive", 2) + choose("negative_same_breed", 2) + choose("negative_random", 2)
    if len(chosen) < 6:
        raise RuntimeError("Nao foi possivel selecionar pares de verificacao suficientes com imagens locais.")

    type_map = {
        "positive": "Positivo",
        "negative_same_breed": "Negativo (mesma raça)",
        "negative_random": "Negativo (raças diferentes)",
    }
    card_w, card_h = 650, 390
    cols, rows_n = 3, 2
    pad = 30
    canvas = Image.new("RGB", (cols * card_w + (cols + 1) * pad, rows_n * card_h + (rows_n + 1) * pad), WHITE)
    draw = ImageDraw.Draw(canvas)
    font_b = ImageFont.truetype(BOLD, 24)
    font_s = ImageFont.truetype(FONT, 17)
    font_m = ImageFont.truetype(FONT, 15)
    for idx, row in enumerate(chosen):
        col, row_pos = idx % cols, idx // cols
        x = pad + col * (card_w + pad)
        y = pad + row_pos * (card_h + pad)
        color = GREEN if int(row["label"]) == 1 else BLUE
        draw.rounded_rectangle([x, y, x + card_w, y + card_h], radius=10, outline="#D5DCE5", width=2, fill="#FFFFFF")
        label = "Mesmo cão" if int(row["label"]) == 1 else "Cães diferentes"
        pil_text(draw, (x + 18, y + 32), f"{type_map.get(row['pair_type'], row['pair_type'])}", font_b, color)
        pil_text(draw, (x + 18, y + 60), f"Real: {label}", font_s, TEXT)
        if str(row.get("shared_breed", "nan")) != "nan":
            pil_text(draw, (x + 18, y + 83), f"Raça/grupo: {clean_breed(row['shared_breed'])}", font_m, "#5B677A")
        for j, path in enumerate([row["_p1"], row["_p2"]]):
            img = ImageOps.fit(Image.open(path).convert("RGB"), (292, 215), method=Image.Resampling.LANCZOS)
            ix = x + 18 + j * 322
            iy = y + 108
            canvas.paste(img, (ix, iy))
            draw.rectangle([ix, iy, ix + 292, iy + 215], outline="#D5DCE5", width=2)
        pil_text(draw, (x + 18, y + card_h - 33), "Par do conjunto de teste de verificação", font_m, "#5B677A")
    canvas.save(OUT / "prediction_examples_results.png")
    canvas.save(OUT / "prediction_examples_results.pdf", resolution=300.0)


def main() -> None:
    ensure_dirs()
    configure()
    pipeline_veripet()
    mlp_network()
    mycnn()
    eda_distributions()
    curvas_treinamento()
    distribuicao_acuracias()
    top10_piores_racas()
    principais_confusoes()
    verification_summary()
    training_curves()
    amostras_predicoes()
    mosaicos_erros_2x2()
    results_comparison()
    prediction_examples_results()


if __name__ == "__main__":
    main()
