# Reproducibility

This repository keeps the dog experiment code and cleaned notebooks, but not the datasets or trained model weights.

## Environment

The reported experiments were run in Google Colab using an NVIDIA A100 GPU with 40 GB of memory.

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Reproduction Outline

1. Prepare Stanford Dogs and PetFace Dog outside the repository.
2. Place or mount datasets using the structure in `docs/DATASETS.md`.
3. Run the breed classification notebook:

```text
notebooks/01_stanford_dogs_classification.ipynb
```

4. Run the breed error analysis notebook:

```text
notebooks/02_breed_error_analysis.ipynb
```

5. Run the verification notebook:

```text
notebooks/03_petface_dog_verification.ipynb
```

6. Run the breed filter and fusion notebooks:

```text
notebooks/04_breed_verifier_filter.ipynb
notebooks/05_fusion_verifier_classifier.ipynb
```

7. Use `notebooks/06_qualitative_analysis.ipynb` and scripts under `scripts/` to regenerate qualitative material and paper figures.

## Validation

Run lightweight tests for the reusable research code:

```bash
pytest tests/research -q
```

These tests validate path configuration, manifest handling, sampling, pair generation, taxonomy helpers, and verification metrics. They do not require the full datasets.

## Result Tables

Small result tables are versioned under:

```text
results/classification/
results/verification/
```

They are included to document the metrics reported in the paper and presentation. They are not a substitute for the original datasets.
