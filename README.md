# Hybrid Music Clustering (CSE 425)

Course project for CSE 425 (Neural Networks). It clusters songs by combining **audio** and **lyrics** in a hybrid convolutional variational autoencoder (VAE), on both English and Bangla music.

## Approach

| Task | What it does |
|---|---|
| Easy | Baseline clustering on audio features |
| Medium | A convolutional VAE on mel-spectrograms; clustering in the learned latent space |
| Hard | A **hybrid conv-VAE** that fuses a mel-spectrogram encoder with a lyrics encoder, then clusters the joint latent space |

- **Audio**: 5-second clips converted to 64-band mel-spectrograms (`librosa`)
- **Lyrics**: sentence embeddings from `sentence-transformers/all-mpnet-base-v2`
- **Model**: audio and lyrics branches are fused into a 64-dimensional latent space and trained with a reconstruction plus KL loss (PyTorch)
- **Clustering**: K-Means, Agglomerative and DBSCAN
- **Metrics**: Silhouette, Davies-Bouldin, NMI, ARI and cluster purity against genre labels

## Repository layout

```
project/
  notebooks/cse425.ipynb   # the full pipeline
  Report/                  # the written report (PDF)
  results/visualisation/   # generated plots
  data/audio, data/lyrics  # dataset folders (files are not committed)
```

## Data

The notebook was run on Kaggle and reads these inputs:

- Bangla speech clips (`bangla-speech-2k`)
- English genre audio (`genres-original`, GTZAN)
- English lyrics CSV and Bangla lyrics CSV

Large data files (`*.csv`, `*.zip`) are ignored by git.

## Running

1. Open `project/notebooks/cse425.ipynb` on Kaggle (or locally with the datasets in place and the paths at the top of the notebook updated).
2. Install: `torch`, `librosa`, `transformers`, `scikit-learn`, `pandas`, `numpy`, `tqdm`.
3. Run the cells in order.

## Findings

The VAE latent space shows weak but detectable structure (Silhouette around 0.10), while the baseline clustering stays poorly separated. See the report for the full comparison.