# VeriPet Supplementary Material

This supplementary material supports the IEEE research article and the course presentation.

## Repository

- URL: https://github.com/mdrapha/veripet.git
- Research package: `src/veripet/research`
- Dog notebooks: `notebooks/`
- Article package: `paper/ieee_article`
- Presentation package: `paper/presentation`

## Datasets

The experiments use two datasets because the tasks require different annotations.

### Stanford Dogs

- Task: breed classification.
- Images: 20,580.
- Classes: 120 dog breeds.
- Annotations: Pascal VOC bounding boxes.
- Split: 70% training, 15% validation, 15% test.

### PetFace Dog

- Task: individual biometric verification.
- Identities used: 46,755.
- Training images: 168,348.
- Test verification pairs: 35,366 in the main verification protocol.
- Fusion evaluation pairs: 37,492, split into positive pairs, same-breed hard negatives, and different-breed easy negatives.

Datasets are not redistributed with this repository. They must be obtained from the original sources and used according to their licenses and access conditions.

## Reproducibility Workflow

1. Prepare Stanford Dogs and PetFace Dog outside this repository.
2. Train/evaluate breed classification with `notebooks/01_stanford_dogs_classification.ipynb`.
3. Run breed error analysis with `notebooks/02_breed_error_analysis.ipynb`.
4. Train/evaluate individual verification with `notebooks/03_petface_dog_verification.ipynb`.
5. Run filter/fusion analysis with `notebooks/04_breed_verifier_filter.ipynb` and `notebooks/05_fusion_verifier_classifier.ipynb`.
6. Generate qualitative support material with `notebooks/06_qualitative_analysis.ipynb`.

## Main Results

### Breed Classification

- Model: ConvNeXt-Tiny.
- Dataset: Stanford Dogs.
- Accuracy: 92.55%.
- Macro-F1: 92.4%.
- Weighted-F1: 92.6%.

### Individual Verification

- Model: ConvNeXt-Small.
- Best loss: ArcFace.
- AUC: 97.40%.
- Accuracy: 92.52%.
- Intra-breed accuracy: 87.88%.

### Breed-Based Filtering

- ArcFace verifier: 93.33% overall accuracy in the fusion evaluation set.
- Breed Top-1 rule: 76.23% overall accuracy, with high false rejection of positive pairs.
- Breed overlap: 93.10% overall accuracy, close to but not better than the ArcFace verifier.

## Expected Computing Environment

- Google Colab or equivalent GPU environment.
- NVIDIA A100 40 GB used in the reported experiments.
- PyTorch 2.x.
- Mixed precision enabled.
- Batch sizes:
  - 256 for breed classification.
  - 128 for verification.

## Generated Deliverables

- IEEE article source: `paper/ieee_article/main.tex`
- Article README: `paper/ieee_article/README.md`
- English Beamer presentation: `paper/presentation/main.tex`
- Presentation README: `paper/presentation/README.md`

## Compilation

Compile the IEEE article with:

```bash
cd paper/ieee_article
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Compile the presentation with:

```bash
cd paper/presentation
pdflatex main.tex
pdflatex main.tex
```
