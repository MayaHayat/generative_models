# CovidGAN-Pytorch

A PyTorch reconstruction of **CovidGAN** (Waheed, Goyal, Gupta, Khanna, Al-Turjman, Pinheiro,
*"CovidGAN: Data Augmentation Using Auxiliary Classifier GAN for Improved Covid-19 Detection"*,
IEEE Access, vol. 8, 2020): an Auxiliary Classifier GAN that synthesizes chest X-ray (CXR) images
to augment a small COVID-19 dataset, plus the VGG16-based CNN it's used to improve.

The paper reports **85% → 95% accuracy** on COVID-19 detection when a CNN trained on real data
alone (CNN-AD) is retrained on real + CovidGAN-synthesized data (CNN-SA).

> **Reproducibility note.** We could not find a published, direct reproduction of this +10-point
> claim. The closest independent test is Fedoruk et al., ["Performance of GAN-based augmentation
> for deep learning COVID-19 image classification"](https://arxiv.org/abs/2304.09067)
> (AIP Conference Proceedings, 2023; follow-up:
> [arXiv:2401.14705](https://arxiv.org/abs/2401.14705)), which cites CovidGAN's 85%→95% result as
> its benchmark and tests the same claim on a large modern COVID-19 CXR dataset (~21K images). They
> find GAN augmentation does **not** reliably help — their best model (EfficientNet-B0) reached
> 90.2% with simple class balancing and *dropped* to 84.1% with GAN augmentation added — and
> conclude that dataset size, not augmentation, is the dominant driver of accuracy: GAN-based
> augmentation "underperforms in the case of smaller datasets." That matches what this
> reconstruction independently found (see the classifier results below): once trained on a larger,
> cleaner modern dataset, the baseline CNN-AD already scores well above the paper's reported 85%,
> leaving little headroom for augmentation to fill.

## Architecture

- **Generator** — label (Embedding→Dense→7×7×1) concatenated with noise (Dense→ReLU→7×7×1024),
  four transpose-conv upsampling blocks (5×5, stride 2, BatchNorm+ReLU, Tanh on the last) taking
  7×7×1025 → 112×112×3. ~22M params.
- **Discriminator** — five 3×3 conv blocks (BatchNorm, LeakyReLU(0.2), Dropout(0.5)) downsampling
  112×112×3 → 7×7×512, flattened into two heads: a sigmoid validity head and a softmax class head
  (AC-GAN). ~2M params.
- **Classifier** — frozen ImageNet VGG16 conv base + GlobalAveragePooling → Dense(64, ReLU) →
  Dropout(0.5) → Dense(2, softmax). ~14.7M params, only ~33K trainable.

See `covidgan/models.py` for the exact layer-by-layer implementation, matching Figs. 1–3 of the
paper.

> **Note on the noise dimension:** an earlier exploratory pass at this generator used a 20,000-d
> noise vector, which inflates the first generator layer to >20M params on its own and doesn't
> match Fig. 3 of the paper, which labels the noise-branch dense layer's input `?×100`. The
> models here use `z_dim=100`.

## Setup

```bash
pip install -r requirements.txt
```

Needs `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `pillow`. A CUDA GPU is strongly
recommended — the paper trains CovidGAN for 2000 epochs (~5h on an RTX 2060); on CPU that is not
practical, though every script also runs correctly (just slowly) on CPU for a smoke test.

## Getting the dataset

The paper merges three public sources into 403 COVID-CXR + 721 Normal-CXR images:

1. [IEEE Covid Chest X-ray Dataset](https://github.com/ieee8023/covid-chestxray-dataset) — clone it;
   `prepare_dataset.py` reads its `metadata.csv` directly and filters to COVID-19 PA/AP images.
2. [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
   (Kaggle) — use its `Normal/` class folder as `--normal-dir`; its `COVID/` folder can be passed as
   an extra `--extra-covid-dir`.
3. [COVID-19 Chest X-ray Dataset Initiative](https://github.com/agchung/Figure1-COVID-chestxray-dataset)
   — another `--extra-covid-dir`.

```bash
python prepare_dataset.py \
    --ieee-covid-root ../covid-chestxray-dataset \
    --extra-covid-dir /path/to/Figure1-COVID-chestxray-dataset/images \
    --normal-dir /path/to/COVID-19_Radiography_Dataset/Normal \
    --out-dir data
```

This merges the sources, drops near-duplicates with a perceptual average-hash (standing in for
the paper's "Image Hashing method"), and writes a stratified `data/manifest.csv` — by default the
paper's exact split (331/72 COVID train/test, 601/120 Normal train/test) when there's enough data,
otherwise a proportional fallback.

## Pipeline

```bash
# 1. Train CovidGAN (generator + discriminator)
python train_gan.py --manifest data/manifest.csv --out-dir runs/gan
#   defaults: batch 64, lr 2e-4, Adam beta1 0.5, 2000 epochs — same as the paper

# 2. Sample the trained generator to build the synthetic augmentation pool
python generate_synthetic.py --checkpoint runs/gan/checkpoints/covidgan_final.pt \
    --out-dir data/synthetic
#   defaults: 1669 synthetic COVID-CXR, 1399 synthetic Normal-CXR — same as the paper

# 3. Baseline: train the CNN on real data only (CNN-AD)
python train_classifier.py --manifest data/manifest.csv --mode ad --out-dir runs/cnn_ad

# 4. Augmented: train the CNN on real + synthetic data (CNN-SA)
python train_classifier.py --manifest data/manifest.csv --mode sa \
    --synthetic-dir data/synthetic --out-dir runs/cnn_sa
```

Each classifier run writes to its `--out-dir`:

- `metrics.txt` — per-class precision/recall/F1/support/specificity + macro & weighted averages
  and overall accuracy (Table 1 layout).
- `confusion_matrix.png` — Fig. 6 / Fig. 7 equivalent.
- `pca.png` (SA mode only) — PCA scatter of penultimate-layer features colored by class and
  real/synthetic origin (Fig. 5 equivalent).
- `classifier.pt` — trained weights.

Compare `runs/cnn_ad/metrics.txt` against `runs/cnn_sa/metrics.txt` to reproduce the paper's
headline 85% → 95% accuracy jump.

## Repo layout

```
covidgan/
  models.py     Generator, Discriminator, build_classifier
  data.py       dataset loading, dedup, split, torch Dataset
  metrics.py    classification table, confusion matrix, PCA plot
prepare_dataset.py    raw folders -> data/manifest.csv
train_gan.py          train CovidGAN
generate_synthetic.py sample the trained generator into a synthetic pool
train_classifier.py   train + evaluate the detection CNN (AD or SA mode)
CovidGAN_Colab.ipynb   ready-to-run Colab notebook for the full pipeline on a free GPU
```
