# Dataset Setup

This repository does not redistribute datasets. Download datasets from their original sources and use them according to their licenses and access conditions.

## Stanford Dogs

Used for breed classification.

Expected content:

- images;
- Pascal VOC-style bounding-box annotations;
- 120 breed classes.

Recommended local/Drive organization:

```text
veripet/dogs/classification/
├── images/
├── annotations/
├── split/
├── artifacts/
├── results/
└── manifests/
```

The classification workflow uses a stratified split:

- 70% train;
- 15% validation;
- 15% test.

## PetFace Dog

Used for individual biometric verification.

Expected content:

- dog images grouped or referenced by identity;
- identity labels;
- verification pair CSV files;
- breed metadata when available.

Recommended local/Drive organization:

```text
veripet/dogs/verification/
├── images/
├── annotations/
├── split/
├── artifacts/
├── results/
└── manifests/
```

The verification experiments require:

- training identity manifest;
- validation verification pairs;
- test verification pairs;
- trained checkpoints for follow-up/fusion notebooks.

## What Should Stay Out of Git

Never commit:

- raw images;
- extracted archives;
- `.zip`, `.tar`, `.tar.gz`;
- `.mat` split files;
- model checkpoints;
- local Drive mirrors;
- temporary generated samples.
