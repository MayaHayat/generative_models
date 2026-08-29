# CovidGAN-Pytorch

A PyTorch reconstruction of **CovidGAN** (Waheed, Goyal, Gupta, Khanna, Al-Turjman, Pinheiro,
*"CovidGAN: Data Augmentation Using Auxiliary Classifier GAN for Improved Covid-19 Detection"*,
IEEE Access, vol. 8, 2020): an Auxiliary Classifier GAN that synthesizes chest X-ray (CXR) images
to augment a small COVID-19 dataset, plus the VGG16-based CNN it's used to improve.

The paper reports **85% → 95% accuracy** on COVID-19 detection when a CNN trained on real data
alone (CNN-AD) is retrained on real + CovidGAN-synthesized data (CNN-SA).

The work is in two stages. **Stage 1** reconstructs the paper faithfully and diagnoses why the
reported lift does not reproduce. **Stage 2** proposes and evaluates an improved architecture on
both the classifier and the generator side.

> **The written report is not in this repository** — it is submitted separately. This README covers
> the code only: what each script does and how to run it. Numbers cited by the report live in
> [`stage2/results/`](stage2/results).

## Architecture

- **Generator** — label (Embedding→Dense→7×7×1) concatenated with noise (Dense→ReLU→7×7×1024),
  four transpose-conv upsampling blocks (5×5, stride 2, BatchNorm+ReLU, Tanh on the last) taking
  7×7×1025 → 112×112×3. ~22M params.
- **Discriminator** — five 3×3 conv blocks (BatchNorm, LeakyReLU(0.2), Dropout(0.5)) downsampling
  112×112×3 → 7×7×512, flattened into two heads: a sigmoid validity head and a softmax class head
  (AC-GAN). ~2M params.
- **Classifier** — frozen ImageNet VGG16 conv base + GlobalAveragePooling → Dense(64, ReLU) →
  Dropout(0.5) → Dense(2, softmax). ~14.7M params, only ~33K trainable.

See [`covidgan/models.py`](covidgan/models.py) for the layer-by-layer implementation, matching
Figs. 1–3 of the paper. [`covidgan/models_improved.py`](covidgan/models_improved.py) and
[`stage2/`](stage2) hold the Stage 2 variants.

> **Note on the noise dimension:** an earlier exploratory pass at this generator used a 20,000-d
> noise vector, which inflates the first generator layer to >20M params on its own and doesn't
> match Fig. 3 of the paper, which labels the noise-branch dense layer's input `?×100`. The
> models here use `z_dim=100`.

## Setup

```bash
pip install -r requirements.txt
```

`torch`, `torchvision`, `numpy`, `pillow`, `scikit-learn`, `matplotlib`, plus `torchmetrics` (the
FID metric) and `certifi` (`evaluate_fid.py` points `SSL_CERT_FILE` at a working CA bundle before
torchmetrics downloads the pretrained InceptionV3 weights — macOS python.org builds ship without
one). `tqdm` is optional: if absent, progress bars degrade to a no-op and everything still runs.

A CUDA GPU is strongly recommended — the paper trains CovidGAN for 2000 epochs (~5h on an RTX
2060). Every script also runs on CPU (`--device cpu`) or Apple silicon (`--device mps`) correctly,
just slowly; use `--epochs` to cut a smoke test down.

## Getting the dataset

**The datasets are not included in this repository.** The paper merges three public sources into
403 COVID-CXR + 721 Normal-CXR images:

1. [IEEE Covid Chest X-ray Dataset](https://github.com/ieee8023/covid-chestxray-dataset) — clone it;
   `prepare_dataset.py` reads its `metadata.csv` directly and filters to COVID-19 PA/AP images.
2. [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
   (Kaggle) — use its `Normal/` class folder as `--normal-dir`; its `COVID/` folder can be passed as
   an extra `--extra-covid-dir`.
3. [COVID-19 Chest X-ray Dataset Initiative](https://github.com/agchung/Figure1-COVID-chestxray-dataset)
   — another `--extra-covid-dir`.

```bash
python prepare_dataset.py \
    --ieee-covid-root /path/to/covid-chestxray-dataset \
    --extra-covid-dir /path/to/Figure1-COVID-chestxray-dataset/images \
    --normal-dir /path/to/COVID-19_Radiography_Dataset/Normal \
    --out-dir data
```

This merges the sources, drops near-duplicates with a perceptual average-hash (standing in for
the paper's "Image Hashing method"), and writes a stratified `data/manifest.csv` — by default the
paper's exact split (331/72 COVID train/test, 601/120 Normal train/test) when there's enough data,
otherwise a proportional fallback. Every other script reads that manifest.

### Same-source split (source-bias A/B)

The default collection draws COVID and Normal from **different** datasets, so a classifier can
separate the classes on non-pathological cues (resolution, borders, embedded text, brightness) —
a shortcut that inflates accuracy above the paper's. To measure how much of the accuracy is real
signal vs. this source bias, pull **both** classes from one dataset's per-class subfolders with a
single processing pipeline:

```bash
python prepare_dataset.py \
    --same-source-root /path/to/COVID-19_Radiography_Dataset \
    --max-covid 403 --max-normal 721 \
    --out-dir data
```

`--same-source-root` expects `COVID/` and `Normal/` subfolders under one root (override the names
with `--covid-subdir` / `--normal-subdir`).

## Stage 1 — reconstruction

```bash
# 1. Train CovidGAN (generator + discriminator)
python train_gan.py --manifest data/manifest.csv --out-dir runs/gan
#   defaults: batch 64, lr 2e-4, Adam beta1 0.5, 2000 epochs — same as the paper
#   --resume <checkpoint> continues an interrupted run; --seed controls initialisation

# 2. Sample the trained generator to build the synthetic augmentation pool
python generate_synthetic.py --checkpoint runs/gan/checkpoints/covidgan_final.pt \
    --out-dir data/synthetic
#   defaults: 1669 synthetic COVID-CXR, 1399 synthetic Normal-CXR — same as the paper

# 3. Baseline: train the CNN on real data only (CNN-AD)
python train_classifier.py --manifest data/manifest.csv --mode ad --out-dir runs/cnn_ad

# 4. Augmented: train the CNN on real + synthetic data (CNN-SA)
python train_classifier.py --manifest data/manifest.csv --mode sa \
    --synthetic-dir data/synthetic --out-dir runs/cnn_sa

# Generator quality
python evaluate_fid.py --real-manifest data/manifest.csv --synthetic-dir data/synthetic

# Multi-seed check on the faithful Stage 1 classifier
python stage1_multiseed.py
```

Each classifier run writes to its `--out-dir`:

- `metrics.txt` — per-class precision/recall/F1/support/specificity + macro & weighted averages
  and overall accuracy (Table 1 layout).
- `confusion_matrix.png` — Fig. 6 / Fig. 7 equivalent.
- `pca.png` (SA mode only) — PCA scatter of penultimate-layer features colored by class and
  real/synthetic origin (Fig. 5 equivalent).
- `classifier.pt` — trained weights.

Comparing `runs/cnn_ad/metrics.txt` against `runs/cnn_sa/metrics.txt` is the paper's actual claim.

## Stage 2 — improved architecture

Run as modules from this directory (`python -m stage2.<name>`).

**Track A — the classifier.** Unfreeze encoder blocks over a BatchNorm head:

```bash
python -m stage2.train_stage2 --mode ad --unfreeze-blocks 2 --out-dir runs/stage2_cnn_ad
python -m stage2.train_stage2 --mode sa --unfreeze-blocks 2 \
    --synthetic-dir data/synthetic --out-dir runs/stage2_cnn_sa

# multi-seed AD-vs-SA (mean +/- std); run for uf=1 and uf=2
python -m stage2.multiseed --seeds 0 1 2 3 4 --unfreeze-blocks 2 --out-root runs/stage2_multiseed_uf2

# supporting analyses
python -m stage2.data_scarcity --fractions 0.1 0.25 0.5 1.0 --seeds 0 1 2 --unfreeze-blocks 2
python -m stage2.diagnostic_curve --unfreeze-blocks 2 --epochs 25
python -m stage2.compare_results          # assembles the paper/reconstruction/improved table
```

Ablations: `--unfreeze-blocks {0..5}` (0 = Stage 1 behaviour + BN head); `--no-head-bn` drops the
BatchNorm.

**Track B — the generator.** Noise 0.02→1.0, DiffAugment, spectral norm, and a projection
discriminator that removes the class-fingerprint incentive:

```bash
python -m stage2.train_gan_manual --disc acgan      --out-dir runs/gan_acgan      --epochs 300
python -m stage2.train_gan_manual --disc projection --out-dir runs/gan_projection --epochs 300

python -m stage2.generate_improved \
    --checkpoint runs/gan_projection/checkpoints/covidgan_final.pt --out-dir data/synth_projection

# generator quality + the synthetic-only transfer probe (train on synthetic, test on real)
python evaluate_fid.py --real-manifest data/manifest.csv --real-split test \
    --synthetic-dir data/synth_projection
python -m stage2.synthetic_only_probe --synthetic-dir data/synth_projection --out-dir runs/probe_projection

# frozen-head scarcity sweep
python -m stage2.data_scarcity --frozen --synthetic-dir data/synth_projection \
    --fractions 0.1 0.25 0.5 1.0 --seeds 0 1 2

python -m stage2.make_figures             # regenerates stage2/results/figures/
```

## Repo layout

```
covidgan/                 Stage 1 package
  models.py               Generator, Discriminator, build_classifier, pick_device
  models_improved.py      Stage 2 architecture variants
  data.py                 dataset loading, dedup, split, torch Dataset
  metrics.py              classification table, confusion matrix, PCA plot

prepare_dataset.py        raw folders -> data/manifest.csv
train_gan.py              train CovidGAN
generate_synthetic.py     sample the trained generator into a synthetic pool
train_classifier.py       train + evaluate the detection CNN (AD or SA mode)
evaluate_fid.py           FID of a synthetic pool
stage1_multiseed.py       multi-seed Stage 1 runs

stage2/                   Stage 2 package
  train_stage2.py         improved classifier (encoder unfreezing + BN head)
  multiseed.py            5-seed AD-vs-SA comparison
  data_scarcity.py        real-data subsampling sweep
  diagnostic_curve.py     train/test curve (diagnostic only)
  compare_results.py      assembles the comparison table
  train_gan_manual.py     improved GAN (noise 1.0 + DiffAugment + spectral norm)
  gan_improved.py         AC-GAN discriminator variant
  gan_projection.py       projection discriminator
  diffaugment.py          differentiable augmentation
  generate_improved.py    sample a Stage 2 generator
  synthetic_only_probe.py train-on-synthetic / test-on-real transfer probe
  make_figures.py         regenerates the figures
  model.py                Stage 2 classifier definition
  results/                executed numbers (CSV/JSON) + figures the report cites

notebooks/                Colab notebook + the Stage 2 GAN comparison
```

Generated artifacts (`data/`, `runs/`, model weights) are gitignored — every one is reproducible
from the commands above.

## Second dataset

[`../DDSM-ACGAN-Pytorch`](../DDSM-ACGAN-Pytorch) applies this same architecture to CBIS-DDSM
mammography as an independent validation of every conclusion drawn here. It is a separate,
self-contained folder with its own README; nothing in this folder depends on it.
