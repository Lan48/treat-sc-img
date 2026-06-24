# treat-sc-img

Spatial transcriptomics as images for large-scale pretraining.

## Overview

This repository implements workflows for spatial transcriptomics analysis with a focus on
croppable patch construction, multi-channel feature design, KNN-based classification, and
supervised domain detection. The code is organized around AnnData (`.h5ad`) inputs and
supports both embedding-based features and HVG-based features.

## Paper Reference

The current README is aligned with the ideas described in the paper:

- arXiv: [2603.13432](https://arxiv.org/abs/2603.13432)

## Main Workflows

- `cls_knn.py` and `cls_knn2.py`: KNN classification with optional spatial smoothing and
  support for single-file or multi-file `.h5ad` inputs.
- `domain_detection_supervsed.py`: supervised MLP/Transformer-style domain detection on
  AnnData features.
- `domain_detection_supervsed2.py`: multi-file training/testing split with HVG selection
  controlled by train/test source.
- `domain_detection_supervsed6.py`: prediction pipeline that writes labels back into `adata.obs`
  for downstream visualization and analysis.

## Expected Data

Inputs are AnnData objects containing:

- `adata.obs` labels such as `Ground Truth` or `ground_truth`
- `adata.obsm['spatial']` for spatial coordinates when spatial smoothing is used
- `adata.obsm[...]` embeddings such as `STAGATE` or `X_emb_scGPTspatial`

## Typical Usage

```bash
python cls_knn.py --h5ad_path /path/to/data.h5ad --feature_mode emb --emb_name STAGATE
python domain_detection_supervsed.py
python domain_detection_supervsed6.py
```

## Notes

- Paths in the scripts are examples and should be adjusted for your local dataset layout.
- Large data files, model outputs, and local caches are ignored by default.
