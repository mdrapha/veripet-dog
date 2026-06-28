# Notebooks

These notebooks are the dog-focused experiment notebooks kept for reproducibility.

## Execution Order

1. `01_stanford_dogs_classification.ipynb`
   - Trains/evaluates the ConvNeXt-Tiny breed classifier on Stanford Dogs.

2. `02_breed_error_analysis.ipynb`
   - Produces per-breed metrics, top confusions, and qualitative error analysis.

3. `03_petface_dog_verification.ipynb`
   - Trains/evaluates ConvNeXt-Small verification models on PetFace Dog, including Softmax and ArcFace.

4. `04_breed_verifier_filter.ipynb`
   - Tests the breed classifier as a lightweight filter before identity verification.

5. `05_fusion_verifier_classifier.ipynb`
   - Evaluates score-level fusion and breed compatibility against the ArcFace verifier.

6. `06_qualitative_analysis.ipynb`
   - Generates qualitative examples for discussion and presentation.

## Notes

- Outputs were cleared to keep the repository small.
- Paths inside the notebooks still assume a Google Drive / Colab setup. Adjust paths according to your local or cloud environment.
- Datasets and model weights are not included in this repository.
