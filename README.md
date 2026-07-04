# VeriPet Dog

Clean research repository for the dog experiments behind VeriPet: breed classification on Stanford Dogs and individual biometric verification on PetFace Dog.

This repository intentionally contains only code, notebooks, paper material, and small result tables. It does not include datasets, trained weights, raw images, extracted archives, or large generated artifacts.

## What Is Included

- `src/veripet/research/`: reusable research utilities for dataset manifests, path configuration, pair mining, sampling, metrics, evaluation, and Colab workflows.
- `notebooks/`: cleaned dog notebooks used for the experiments.
- `scripts/`: helper scripts for analysis and figure generation.
- `results/`: small CSV/JSON result artifacts used to document reported metrics.
- `paper/`: IEEE article, Beamer presentation, bibliography, and figures for the course delivery.
- `tests/research/`: lightweight tests for the reusable research code.

## What Is Not Included

The following files are deliberately excluded:

- Stanford Dogs images and annotations.
- PetFace Dog images and archives.
- model checkpoints such as `.pth`, `.pt`, `.ckpt`;
- local Drive exports, extracted datasets, `.mat` files, and temporary outputs.

Use the original dataset sources and respect their licenses/access terms.

## Repository Layout

```text
veripet-dog/
├── docs/
├── notebooks/
├── paper/
│   ├── ieee_article/
│   └── presentation/
├── results/
│   ├── classification/
│   └── verification/
├── scripts/
├── src/veripet/research/
└── tests/research/
```

## Main Results

Breed classification:

- Dataset: Stanford Dogs.
- Model: ConvNeXt-Tiny.
- Accuracy: 92.55%.
- Macro-F1: 92.4%.

Individual verification:

- Dataset: PetFace Dog.
- Model: ConvNeXt-Small.
- Best loss: ArcFace.
- AUC: 97.40%.
- Accuracy: 92.52%.
- Intra-breed accuracy: 87.88%.

Breed-based filtering:

- The breed classifier helps interpretation but does not robustly improve over the ArcFace verifier.
- Breed Top-1 agreement is too strict and rejects many true positive pairs.
- Breed probability overlap is close to the verifier but remains slightly below the ArcFace baseline.

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Run tests:

```bash
pytest tests/research -q
```

## Notebooks

Recommended order:

1. `notebooks/01_stanford_dogs_classification.ipynb`
2. `notebooks/02_breed_error_analysis.ipynb`
3. `notebooks/03_petface_dog_verification.ipynb`
4. `notebooks/04_breed_verifier_filter.ipynb`
5. `notebooks/05_fusion_verifier_classifier.ipynb`
6. `notebooks/06_qualitative_analysis.ipynb`
7. `notebooks/07_dog_classification_optuna_baselines.ipynb`
8. `notebooks/08_dog_verification_loss_sweep.ipynb`

The notebooks expect datasets and weights to be mounted locally or through Google Drive. See `docs/DATASETS.md` and `docs/REPRODUCIBILITY.md`.

The two newer experiment notebooks bootstrap the package automatically in
Colab by cloning and installing this public repository:

```python
https://github.com/mdrapha/veripet-dog.git
```

## Paper and Presentation

The course deliverables are in:

- `paper/ieee_article/main.tex`
- `paper/presentation/main.tex`
- `paper/supplementary_material.md`

Compile the article with pdfLaTeX and BibTeX:

```bash
cd paper/ieee_article
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Compile the presentation:

```bash
cd paper/presentation
pdflatex main.tex
pdflatex main.tex
```

## Dataset Policy

Do not commit raw datasets, extracted images, local samples, or trained checkpoints. The `.gitignore` is configured to block common dataset and model artifact paths, but check `git status` before publishing.
