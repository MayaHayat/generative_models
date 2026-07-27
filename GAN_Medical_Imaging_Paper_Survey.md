# Best GAN Papers for Medical Imaging Classification — Graduate Project Survey

> **Objective:** Find the single best GAN paper for a two-stage graduate deep learning project (reproduce → improve) focused on GAN-augmented medical image **classification**.

---

## Table of Contents

1. [Paper #1 — CovidGAN (ACGAN)](#1-covidgan)
2. [Paper #2 — Frid-Adar et al. (Liver Lesion GAN)](#2-frid-adar)
3. [Paper #3 — HistoGAN (Selective Synthetic Augmentation)](#3-histogan)
4. [Paper #4 — Enhanced BAGAN-GP](#4-bagan-gp)
5. [Paper #5 — CXR-ACGAN (3-Class Chest X-Ray)](#5-cxr-acgan)
6. [Paper #6 — STGAN (Skin Lesion, HAM10000)](#6-stgan)
7. [Paper #7 — Semi-Supervised GAN for Retinal Images](#7-ssl-retinal)
8. [Paper #8 — SNGAN Mammography Augmentation](#8-sngan-mammo)
9. [Paper #9 — Improved DCGAN for Skin Lesion Classification](#9-dcgan-skin)
10. [Paper #10 — Semi-Supervised GAN for Melanoma (ISIC 2020)](#10-sgan-melanoma)
11. [Ranking](#ranking)
12. [Final Recommendation](#final-recommendation)

---

## ⚠️ Hardware Constraint: Free Google Colab GPU

The student has access to **free Google Colab only**, which means:

| Constraint | Details |
|---|---|
| **GPU** | Tesla T4 (15 GB VRAM) — occasionally K80 (12 GB) |
| **Session limit** | ~4 hours continuous, then disconnects |
| **Daily limit** | ~8–12 GPU hours before throttling/blocked |
| **RAM** | ~12.7 GB system RAM |
| **Disk** | ~78 GB (but resets every session) |
| **No background training** | Must keep browser tab active |

### What This Means for Paper Selection

- **Papers requiring >4 hours of continuous training per run are risky.** Must use checkpointing to resume across sessions.
- **Image resolution matters hugely.** 64×64 or 128×128 is ideal. 256×256 is feasible but slow. 512×512 is impractical.
- **StyleGAN2-based papers are borderline/impractical** on free Colab — they need 24+ hours.
- **Small datasets + small models are strongly preferred.**
- **Must save checkpoints to Google Drive** every N epochs to survive disconnections.
- All runtime estimates below are recalculated for **Tesla T4 on free Colab**.

---

## 1. CovidGAN

| Field | Details |
|---|---|
| **Title** | CovidGAN: Data Augmentation Using Auxiliary Classifier GAN for Improved Covid-19 Detection |
| **Year** | 2020 |
| **Venue** | IEEE Access |
| **Paper** | https://arxiv.org/abs/2103.05094 / DOI: 10.1109/ACCESS.2020.2994762 |
| **GitHub** | https://github.com/ArijitMishra/CovidGAN-Pytorch (PyTorch reimplementation) / https://github.com/giocoal/CXR-ACGAN-chest-xray-generator-covid19-pneumonia (TensorFlow extended version) |
| **Dataset(s)** | COVID chest X-ray dataset (Cohen et al.), Kaggle chest X-ray pneumonia, COVIDx |
| **Main Idea** | Uses an Auxiliary Classifier GAN (ACGAN) to generate synthetic chest X-ray images for COVID-19 and Normal classes. The synthetic images are then added to the training set, improving a CNN classifier from 85% → 95% accuracy. |

### Architecture

- **Generator:** Takes noise vector z + class label → upsamples via transposed convolutions → generates 128×128 chest X-ray images. Standard DCGAN-style generator with batch normalization and ReLU.
- **Discriminator:** Convolutional network with two output heads: (1) real/fake discrimination, (2) auxiliary class prediction (COVID vs Normal). Uses LeakyReLU, dropout.
- **Losses:** Adversarial loss (binary cross-entropy for real/fake) + auxiliary classification loss (cross-entropy for class labels). Combined: L = L_adv + L_cls.
- **Classification Network:** VGG16-based CNN used as the downstream classifier. Trained on real + synthetic images.
- **Training Pipeline:** Train ACGAN on limited COVID/Normal CXR images → generate synthetic images → augment training set → train VGG16 classifier → evaluate.

### Why It Is a Good Reconstruction Project

- Very simple and clean ACGAN architecture — straightforward to implement from scratch
- Clear before/after results (85% → 95%) that are easy to measure
- Multiple public datasets available (COVIDx, Cohen COVID CXR, Kaggle pneumonia)
- Multiple community reimplementations already exist for reference
- Small image size (128×128) — fast to train
- The paper clearly describes every layer of the generator and discriminator

### Possible Improvements (5–10)

1. **Spectral normalization** in discriminator for training stability
2. **WGAN-GP loss** instead of vanilla BCE — prevents mode collapse
3. **Self-attention layers** (SAGAN-style) in generator/discriminator for better global structure
4. **Better classifier backbone** — replace VGG16 with EfficientNet-B0 or ResNet-50
5. **DiffAugment** — apply differentiable augmentation to real/fake images during GAN training
6. **Multi-scale discriminator** for capturing both local texture and global structure
7. **Feature matching loss** — stabilize training
8. **Progressive growing** — start at low resolution, gradually increase
9. **Projection discriminator** instead of ACGAN's auxiliary classifier head
10. **Contrastive learning** pre-training for the classifier before fine-tuning

### Difficulty, Runtime, Improvement Chances

| Metric | Value |
|---|---|
| Implementation difficulty | 3/10 |
| Training time (Free Colab T4) | **2–3 hours (GAN) + 1–2 hours (classifier) — fits in ONE session** ✅ |
| Colab feasibility | **Excellent** — comfortably fits within session limits |
| Chances of improving | 8/10 |

---

## 2. Frid-Adar et al. (Liver Lesion GAN)

| Field | Details |
|---|---|
| **Title** | GAN-based Synthetic Medical Image Augmentation for increased CNN Performance in Liver Lesion Classification |
| **Year** | 2018 |
| **Venue** | Neurocomputing (journal) |
| **Paper** | https://arxiv.org/abs/1803.01229 |
| **GitHub** | https://github.com/NicoEssi/GAN_Synthetic_Medical_Image_Augmentation (community reimplementation with Jupyter notebook) |
| **Dataset(s)** | Private CT liver lesion dataset (182 lesions: 53 cysts, 64 metastases, 65 hemangiomas). **Note: original dataset is private, but the method can be applied to public datasets like LiTS.** |
| **Main Idea** | Uses DCGAN and ACGAN to synthesize liver lesion ROIs from CT images. Synthetic images augment a CNN classifier, improving 3-class classification accuracy by ~7%. |

### Architecture

- **Generator:** Standard DCGAN generator — noise → transposed convolutions → 64×64 lesion ROI. Also tested ACGAN variant with class-conditional generation.
- **Discriminator:** DCGAN discriminator with convolutional layers + ACGAN auxiliary classifier head.
- **Losses:** Standard adversarial loss + auxiliary classification loss for ACGAN variant.
- **Classification Network:** Custom CNN (AlexNet-style) for 3-class liver lesion classification (cyst, metastasis, hemangioma). Sensitivity improved from 78.6% to 85.7%.
- **Training Pipeline:** Classic augmentation first (rotation, flip, scale) → train GAN on augmented set → generate synthetic ROIs → combine with real → train classifier.

### Why It Is a Good Reconstruction Project

- Foundational paper in the field — very well cited (1400+ citations)
- Simple architecture (DCGAN/ACGAN) that is easy to implement
- Clear methodology: classic augmentation → GAN augmentation → classification
- The community notebook implementation provides a complete reference
- Small image size (64×64 ROIs) — extremely fast training
- **Caveat:** original dataset is private; must use LiTS or similar public liver dataset

### Possible Improvements

1. Replace DCGAN with WGAN-GP for stable training
2. Use spectral normalization in discriminator
3. Upgrade classifier from AlexNet to ResNet-18 or EfficientNet-B0
4. Add self-attention mechanism in generator
5. Implement selective augmentation (quality filtering of synthetic images)
6. Try progressive growing for higher-resolution synthesis
7. Add CBAM attention to classifier
8. Multi-task learning (classification + lesion boundary prediction)

### Difficulty, Runtime, Improvement Chances

| Metric | Value |
|---|---|
| Implementation difficulty | 3/10 |
| Training time (Free Colab T4) | **1–2 hours (GAN) + 1 hour (classifier) — fits in ONE session** ✅ |
| Colab feasibility | **Excellent** — very fast, 64×64 images |
| Chances of improving | 8/10 |
| ⚠️ Caveat | Original dataset is private — must use LiTS or adapt to CBIS-DDSM |

---

## 3. HistoGAN (Selective Synthetic Augmentation)

| Field | Details |
|---|---|
| **Title** | Selective Synthetic Augmentation with HistoGAN for Improved Histopathology Image Classification |
| **Year** | 2021 |
| **Venue** | Medical Image Analysis (top journal) |
| **Paper** | https://arxiv.org/abs/2111.06399 |
| **GitHub** | https://github.com/BMIRDS/HistoGAN (related NeurIPS workshop code for colorectal histopathology GAN) |
| **Dataset(s)** | (1) Cervical histopathology (CIN classification, 4 classes, ~2600 patches), (2) PCam (Patch Camelyon — lymph node metastatic cancer, binary, public via Kaggle) |
| **Main Idea** | Designs a conditional GAN (HistoGAN) for histopathology image synthesis + a **selective augmentation** framework that filters synthetic images by label confidence and feature similarity before adding them to training. Achieves +6.7% and +2.8% accuracy improvement on two datasets. |

### Architecture

- **Generator:** Conditional generator with class label embedding concatenated to noise. Uses residual blocks with batch normalization. Generates 128×128 histopathology patches.
- **Discriminator:** PatchGAN-style discriminator with spectral normalization. Has auxiliary classifier head for class prediction (ACGAN-like).
- **Losses:** Hinge adversarial loss + auxiliary classification loss + feature matching loss.
- **Classification Network:** ResNet-34 for downstream classification. MC-dropout used for uncertainty estimation during image selection.
- **Training Pipeline:** Train HistoGAN → generate candidate synthetic images → **selective filtering** (entropy-based confidence + class centroid distance) → augment training set → train ResNet-34 classifier.

### Why It Is a Good Reconstruction Project

- Published in Medical Image Analysis (top-tier venue) — high-quality, well-written paper
- PCam dataset is publicly available on Kaggle (very easy to obtain)
- The selective augmentation framework is a novel and interesting contribution — goes beyond naive augmentation
- Clear experimental protocol with ablation studies
- Moderate complexity — good balance of challenging but doable
- Multiple clear metrics reported (accuracy, FID, t-SNE visualizations)

### Possible Improvements

1. Replace hinge loss with WGAN-GP loss
2. Add self-attention layers in generator (SAGAN-style)
3. Use StyleGAN2-ADA architecture instead of vanilla conditional GAN
4. Improve selective augmentation with contrastive learning-based filtering
5. Replace ResNet-34 classifier with EfficientNet or ConvNeXt
6. Add DiffAugment during GAN training
7. Use Vision Transformer (ViT) as classifier backbone
8. Multi-scale discriminator for better texture fidelity
9. Curriculum learning for progressive augmentation
10. Improve uncertainty estimation with ensemble instead of MC-dropout

### Difficulty, Runtime, Improvement Chances

| Metric | Value |
|---|---|
| Implementation difficulty | 5/10 |
| Training time (Free Colab T4) | **8–12 hours (GAN) + 2–3 hours (classifier) — needs 3–4 sessions with checkpointing** ⚠️ |
| Colab feasibility | **Moderate** — requires careful checkpointing to Google Drive |
| Chances of improving | 7/10 |

---

## 4. Enhanced BAGAN-GP

| Field | Details |
|---|---|
| **Title** | Enhanced Balancing GAN: Minority-class Image Generation |
| **Year** | 2021 |
| **Venue** | Neural Computing and Applications (Springer) |
| **Paper** | https://doi.org/10.1007/s00521-021-06163-8 / https://arxiv.org/abs/2011.00189 |
| **GitHub** | https://github.com/GH920/improved-bagan-gp (official) |
| **Dataset(s)** | MNIST Fashion (imbalanced), CIFAR-10 (imbalanced), small-scale medical cell image dataset (public) |
| **Main Idea** | Improves BAGAN (Balancing GAN) by using a supervised autoencoder with intermediate embedding to disperse class latent vectors, plus gradient penalty. Specifically targets minority-class image generation for imbalanced datasets. |

### Architecture

- **Generator:** Autoencoder-initialized generator. The decoder portion of a pre-trained autoencoder serves as the generator initialization. Class conditioning via concatenated label embeddings.
- **Discriminator:** ACGAN-style discriminator with real/fake + class classification heads. Uses gradient penalty (from WGAN-GP) instead of batch normalization.
- **Losses:** Wasserstein loss with gradient penalty + auxiliary classification loss.
- **Classification Network:** ResNet-50 and Inception V3 used for downstream evaluation via FID score computation and classification accuracy.
- **Training Pipeline:** (1) Train supervised autoencoder with class-aware latent space, (2) Initialize GAN with autoencoder weights, (3) Fine-tune in adversarial mode with gradient penalty, (4) Generate minority-class images, (5) Train classifier on balanced dataset.

### Why It Is a Good Reconstruction Project

- **Official GitHub repository** with complete code — very easy to reproduce
- Well-structured codebase with clear scripts for training and evaluation
- Medical cell dataset is publicly available (download link in repo)
- The autoencoder-initialization trick is an interesting and learnable technique
- Multiple benchmark datasets allow thorough evaluation
- Clear comparison against original BAGAN baseline

### Possible Improvements

1. Add self-attention mechanism in generator/discriminator
2. Replace gradient penalty with spectral normalization (compare approaches)
3. Use DiffAugment for limited data regime
4. Replace ACGAN auxiliary head with projection discriminator
5. Try contrastive loss in latent space for better class separation
6. Add CBAM attention to the autoencoder
7. Use EfficientNet as the evaluation classifier
8. Implement adaptive augmentation (ADA from StyleGAN2)
9. Multi-resolution discriminator
10. Test on HAM10000 skin lesion dataset for medical relevance

### Difficulty, Runtime, Improvement Chances

| Metric | Value |
|---|---|
| Implementation difficulty | 4/10 |
| Training time (Free Colab T4) | **3–5 hours — fits in ONE session (tight) or two with checkpoint** ✅ |
| Colab feasibility | **Good** — small images, lightweight models |
| Chances of improving | 8/10 |

---

## 5. CXR-ACGAN (3-Class Chest X-Ray)

| Field | Details |
|---|---|
| **Title** | CXR-ACGAN: Auxiliary Classifier GAN for Conditional Generation of Chest X-Ray Images |
| **Year** | 2023 (university project, well-documented) |
| **Venue** | University project (Sapienza Roma) — not peer-reviewed but exceptionally documented |
| **Paper** | Project documentation in repository |
| **GitHub** | https://github.com/giocoal/CXR-ACGAN-chest-xray-generator-covid19-pneumonia (official, MIT license) |
| **Dataset(s)** | COVIDx CXR-3 (public Kaggle dataset, 3 classes: COVID-19, Pneumonia, Normal) |
| **Main Idea** | Trains an ACGAN on COVIDx to generate 3-class chest X-rays, then uses GAN-generated images to balance minority classes and improve classifier performance. Evaluates FID, IS, and t-SNE. |

### Architecture

- **Generator:** Standard ACGAN generator with class conditioning. Noise + label → transposed convolutions → 128×128 or 256×256 CXR images.
- **Discriminator:** Convolutional discriminator with dual output heads (real/fake + 3-class).
- **Losses:** Binary cross-entropy (real/fake) + categorical cross-entropy (3 classes).
- **Classification Network:** Multiple classifiers tested (VGG16, ResNet-50, custom CNN).
- **Training Pipeline:** Download COVIDx → preprocess → train ACGAN → generate synthetic minority-class images → augment → train classifier → evaluate.

### Why It Is a Good Reconstruction Project

- Extremely well-documented repository with Jupyter notebooks
- Complete pipeline from data download to evaluation
- COVIDx dataset readily available on Kaggle
- TensorFlow/Keras implementation — accessible for beginners
- FID, IS, and t-SNE evaluation already implemented
- MIT licensed — no restrictions

### Possible Improvements

1. Port to PyTorch and use WGAN-GP loss
2. Add spectral normalization
3. Use progressive growing for higher resolution
4. Self-attention in generator
5. Better classifier (EfficientNet-B3)
6. Implement selective augmentation (à la HistoGAN)
7. Add feature matching loss
8. Multi-scale discriminator
9. DiffAugment
10. Conditional batch normalization instead of label concatenation

### Difficulty, Runtime, Improvement Chances

| Metric | Value |
|---|---|
| Implementation difficulty | 2/10 |
| Training time (Free Colab T4) | **3–6 hours — fits in 1–2 sessions** ✅ |
| Colab feasibility | **Excellent** — Jupyter notebooks already provided, Colab-ready |
| Chances of improving | 9/10 |

---

## 6. STGAN (Skin Lesion, HAM10000)

| Field | Details |
|---|---|
| **Title** | A GAN-based Data Augmentation Method for Imbalanced Multi-class Skin Lesion Classification (Self-Transfer GAN) |
| **Year** | 2024 |
| **Venue** | IEEE Access |
| **Paper** | https://www.researchgate.net/publication/377830253 |
| **GitHub** | Limited — based on StyleGAN2-ADA codebase which is public (NVIDIA) |
| **Dataset(s)** | HAM10000 (public, 7 classes of skin lesions, 10015 images, highly imbalanced) |
| **Main Idea** | Two-stage GAN: (1) train unconditional GAN on all classes to learn universal features, (2) transfer knowledge to class-specific GANs via fine-tuning. Generates minority-class images to balance HAM10000 for improved classification. |

### Architecture

- **Generator:** Based on StyleGAN2 architecture with adaptive augmentation. Style mapping network + synthesis network.
- **Discriminator:** StyleGAN2 discriminator with R1 regularization.
- **Losses:** Non-saturating logistic loss + R1 gradient penalty + path length regularization.
- **Classification Network:** Standard CNN classifiers (ResNet, EfficientNet) evaluated on balanced vs imbalanced datasets.
- **Training Pipeline:** Train universal GAN → fine-tune per-class → generate minority images → balance dataset → train classifier.

### Why It Is a Good Reconstruction Project

- HAM10000 is one of the most widely used public medical image datasets
- StyleGAN2-ADA has excellent official code (NVIDIA GitHub)
- 7-class imbalanced problem is realistic and challenging
- Recent paper with modern architecture
- Results show clear FID improvements over baselines

### Possible Improvements

1. Use contrastive loss for better class separation in transfer stage
2. Add CBAM attention to classifier
3. Implement selective augmentation filtering
4. Try knowledge distillation from universal → class-specific GAN
5. Vision Transformer classifier
6. Curriculum-based augmentation strategy

### Difficulty, Runtime, Improvement Chances

| Metric | Value |
|---|---|
| Implementation difficulty | 6/10 |
| Training time (Free Colab T4) | **15–30+ hours — needs 5–8 sessions with checkpointing** ❌ |
| Colab feasibility | **Poor** — StyleGAN2-based, too heavy for free Colab |
| Chances of improving | 6/10 |

---

## 7. Semi-Supervised GAN for Retinal Images

| Field | Details |
|---|---|
| **Title** | Semi-Supervised Deep Learning for Abnormality Classification in Retinal Images |
| **Year** | 2018 |
| **Venue** | NeurIPS 2018 ML4H Workshop |
| **Paper** | https://arxiv.org/abs/1812.07832 |
| **GitHub** | No official repo — but architecture is well-described and straightforward to implement |
| **Dataset(s)** | EyePACS diabetic retinopathy dataset (public, Kaggle) |
| **Main Idea** | Patch-based semi-supervised GAN for diabetic retinopathy classification. The discriminator simultaneously classifies real/fake AND disease grades. Achieves high AUC with only 10–20 labeled images, outperforming supervised baselines by up to 15% when <30% of data is labeled. |

### Architecture

- **Generator:** Standard convolutional generator producing retinal image patches.
- **Discriminator:** K+1 class output (K real classes + 1 fake class). Implements the semi-supervised GAN framework from Salimans et al. (Improved Techniques for Training GANs).
- **Losses:** Supervised loss (labeled real images) + unsupervised loss (all real vs fake) combined.
- **Classification Network:** The discriminator itself IS the classifier (no separate network needed).
- **Training Pipeline:** Extract patches → split into labeled and unlabeled pools → train semi-supervised GAN → discriminator serves as final classifier.

### Why It Is a Good Reconstruction Project

- Elegant concept — discriminator doubles as classifier
- EyePACS dataset is freely available on Kaggle
- Semi-supervised learning is a very relevant and interesting paradigm
- Strong experimental results with limited labels
- Interpretability through patch-level predictions
- Conceptually clean and well-written

### Possible Improvements

1. Add spectral normalization to discriminator
2. WGAN-GP loss
3. Self-attention layers
4. Better patch sampling strategy
5. Consistency regularization (MixMatch-style)
6. Virtual adversarial training (VAT)
7. Progressive growing for larger patches
8. Feature matching loss
9. Mean teacher for semi-supervised component
10. EfficientNet-based discriminator backbone

### Difficulty, Runtime, Improvement Chances

| Metric | Value |
|---|---|
| Implementation difficulty | 5/10 |
| Training time (Free Colab T4) | **3–6 hours — fits in 1–2 sessions** ✅ |
| Colab feasibility | **Good** — patch-based approach keeps data size manageable |
| Chances of improving | 7/10 |

---

## 8. SNGAN Mammography Augmentation

| Field | Details |
|---|---|
| **Title** | GAN-based Data Augmentation to Improve Breast Ultrasound and Mammography Mass Classification |
| **Year** | 2024 |
| **Venue** | Biomedical Signal Processing and Control (Elsevier) |
| **Paper** | https://www.sciencedirect.com/science/article/pii/S1746809424003136 |
| **GitHub** | No official repo — but uses standard SNGAN, WGAN-GP, CycleGAN, CGAN architectures with public implementations |
| **Dataset(s)** | CBIS-DDSM (mammography, public), breast ultrasound dataset (public) |
| **Main Idea** | Comprehensive comparison of SNGAN, WGAN-GP, CycleGAN, and CGAN for breast imaging augmentation. SNGAN (FID=52.89) was best for mammography; CGAN best for ultrasound. Evaluated with ResNet-18 classifier. |

### Architecture

- **Generator/Discriminator:** Standard architectures for each GAN variant. SNGAN uses spectral normalization in discriminator.
- **Losses:** Vary by GAN variant (hinge loss for SNGAN, Wasserstein+GP for WGAN-GP, cycle consistency for CycleGAN, conditional BCE for CGAN).
- **Classification Network:** ResNet-18 for benign vs malignant classification.
- **Training Pipeline:** Extract 128×128 ROIs → train multiple GANs → evaluate generation quality (FID, KID, SSIM, MS-SSIM, BRISQUE, NIQE, PIQE) → augment training set → train ResNet-18 → compare.

### Why It Is a Good Reconstruction Project

- Uses **CBIS-DDSM** — the gold standard public mammography dataset
- Directly relevant to breast cancer / mammography interest
- Compares multiple GAN architectures — good for learning
- Standard architectures with many public implementations
- Clear metrics and evaluation protocol

### Possible Improvements

1. Add self-attention to best-performing SNGAN
2. Use EfficientNet or ConvNeXt as classifier
3. Implement selective augmentation (HistoGAN-style)
4. Try StyleGAN2-ADA
5. Multi-scale discriminator
6. DiffAugment
7. Better ROI preprocessing / multi-resolution patches
8. Contrastive pre-training before classifier fine-tuning

### Difficulty, Runtime, Improvement Chances

| Metric | Value |
|---|---|
| Implementation difficulty | 4/10 |
| Training time (Free Colab T4) | **3–6 hours per GAN variant + 2 hours classifier — 1–2 sessions per variant** ⚠️ |
| Colab feasibility | **Moderate** — feasible but testing all 4 GAN variants takes many sessions |
| Chances of improving | 7/10 |

---

## 9. Improved DCGAN for Skin Lesion Classification

| Field | Details |
|---|---|
| **Title** | Skin Lesion Synthesis and Classification Using an Improved DCGAN Classifier |
| **Year** | 2023 |
| **Venue** | Diagnostics (MDPI) |
| **Paper** | https://pmc.ncbi.nlm.nih.gov/articles/PMC10453872/ |
| **GitHub** | No official repo — but DCGAN is trivial to implement; HAM10000 is public |
| **Dataset(s)** | HAM10000 (ISIC, 7 classes, public) |
| **Main Idea** | Improved DCGAN with architectural modifications for skin lesion synthesis. Generated images augment HAM10000 for classification. Achieves 99.38% accuracy (claimed). |

### Architecture

- **Generator:** Modified DCGAN with additional residual connections and batch normalization tuning.
- **Discriminator:** Standard DCGAN discriminator with dropout and LeakyReLU.
- **Losses:** Standard adversarial BCE loss.
- **Classification Network:** Custom CNN for 7-class skin lesion classification.
- **Training Pipeline:** Standard GAN training → synthesize images → augment minority classes → train classifier.

### Why It Is a Good Reconstruction Project

- HAM10000 is easily accessible
- Simple architecture — very beginner-friendly
- Clear room for improvement (vanilla DCGAN)
- Medical imaging relevance (skin cancer)

### Possible Improvements

1. Almost any modern GAN technique will improve this (WGAN-GP, spectral norm, self-attention)
2. Better classifier backbone
3. Progressive growing
4. Conditional generation
5. Feature matching
6. Selective augmentation

### Difficulty, Runtime, Improvement Chances

| Metric | Value |
|---|---|
| Implementation difficulty | 2/10 |
| Training time (Free Colab T4) | **2–3 hours — fits in ONE session easily** ✅ |
| Colab feasibility | **Excellent** — simplest and fastest of all options |
| Chances of improving | 9/10 |

---

## 10. Semi-Supervised GAN for Melanoma (ISIC 2020)

| Field | Details |
|---|---|
| **Title** | Semi-supervised GAN with hybrid regularization and evolutionary hyperparameter tuning for accurate melanoma detection |
| **Year** | 2025 |
| **Venue** | Scientific Reports (Nature) |
| **Paper** | https://www.nature.com/articles/s41598-025-17756-x |
| **GitHub** | https://github.com/AmirhoseinDolatabadi/Melanoma (official) |
| **Dataset(s)** | ISIC-2020, HAM10000, PH2, DermNet (all public) |
| **Main Idea** | Semi-supervised GAN with hybrid regularization (spectral norm + gradient penalty) for melanoma classification under limited labeled data. Achieves F-measure >90% on all four datasets. |

### Architecture

- **Generator:** Deep convolutional generator with spectral normalization, produces skin lesion images.
- **Discriminator:** K+1 output semi-supervised discriminator with hybrid regularization (spectral norm + gradient penalty combined).
- **Losses:** Semi-supervised loss (supervised + unsupervised) + hybrid regularization.
- **Classification Network:** Discriminator serves as classifier (semi-supervised paradigm).
- **Training Pipeline:** Split data into labeled/unlabeled → train semi-supervised GAN with hybrid regularization → discriminator classifies melanoma.

### Why It Is a Good Reconstruction Project

- **Official GitHub repository** with code
- Very recent paper (2025) — state of the art
- Tests on **four public datasets** — robust evaluation
- Semi-supervised approach is conceptually interesting
- Uses modern regularization techniques
- Published in Scientific Reports (Nature) — well peer-reviewed

### Possible Improvements

1. Add self-attention layers
2. Try contrastive learning in latent space
3. Use EfficientNet backbone in discriminator
4. Add consistency regularization (FixMatch-style)
5. Multi-scale discriminator
6. Virtual adversarial training
7. Feature matching loss
8. Curriculum learning for labeled data
9. CBAM attention modules
10. Ensemble multiple semi-supervised models

### Difficulty, Runtime, Improvement Chances

| Metric | Value |
|---|---|
| Implementation difficulty | 5/10 |
| Training time (Free Colab T4) | **6–10 hours — needs 2–3 sessions with checkpointing** ⚠️ |
| Colab feasibility | **Moderate** — feasible with checkpointing, but 4 datasets = many runs |
| Chances of improving | 7/10 |

---

## Ranking

Scores are 1–10 (10 = best).

| # | Paper | Ease of Repro | Code Avail. | Paper Quality | Interesting Arch. | Likelihood of Improv. | Grad Project Fit | Med. Classif. | **Colab Feasible** | **Total** |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CovidGAN | 9 | 8 | 6 | 5 | 8 | 9 | 7 | **10** | **62** |
| 2 | Frid-Adar Liver | 8 | 6 | 8 | 5 | 8 | 8 | 7 | **9** | **59** |
| 3 | HistoGAN | 7 | 6 | 9 | 8 | 7 | 7 | 8 | **5** | **57** |
| 4 | BAGAN-GP | 8 | 9 | 7 | 7 | 8 | 8 | 6 | **8** | **61** |
| 5 | CXR-ACGAN | 9 | 9 | 5 | 5 | 9 | 8 | 7 | **9** | **61** |
| 6 | STGAN | 5 | 4 | 7 | 8 | 6 | 5 | 8 | **2** | **45** |
| 7 | SSL Retinal | 6 | 3 | 7 | 8 | 7 | 6 | 8 | **8** | **53** |
| 8 | SNGAN Mammo | 6 | 4 | 7 | 6 | 7 | 6 | 9 | **6** | **51** |
| 9 | DCGAN Skin | 9 | 3 | 5 | 3 | 9 | 7 | 7 | **10** | **53** |
| 10 | SS-GAN Melanoma | 7 | 8 | 8 | 8 | 7 | 7 | 9 | **6** | **60** |

### Top 3 (adjusted for Free Colab)

1. **CovidGAN (#1)** — Total 62 ✅ trains in one session
2. **CXR-ACGAN (#5) / BAGAN-GP (#4)** — Total 61 ✅ fits in 1–2 sessions
3. **SS-GAN Melanoma (#10)** — Total 60 ⚠️ feasible but needs checkpointing

---

## Final Recommendation

### ❌ CovidGAN Eliminated

CovidGAN was tested and both baseline and GAN-augmented classifiers achieved ~90% accuracy — the GAN provided **no measurable improvement**. This is a known problem: COVID vs Normal CXR is too easy for modern CNNs, so the classifier saturates before augmentation can help. The original paper's 85%→95% claim was on a tiny early-pandemic dataset and does not reproduce on larger, better-curated datasets.

**Lesson:** A good project requires a task where the baseline classifier struggles enough that augmentation can visibly help. This means: more classes, smaller datasets, harder visual distinctions, or severe class imbalance.

---

### ✅ New Winner: Enhanced BAGAN-GP (#4) — Applied to HAM10000 Skin Lesion Dataset

After eliminating CovidGAN, **Enhanced BAGAN-GP** is the strongest choice, applied to the **HAM10000** skin lesion dataset (7 classes, severe imbalance). Here is why:

### Why BAGAN-GP + HAM10000 is the Best Choice

1. **The task is genuinely hard.** HAM10000 has 7 classes with extreme imbalance: the largest class (nv, melanocytic nevi) has ~6700 images while the smallest (df, dermatofibroma) has only ~115. Baseline classifiers typically achieve 70–85% accuracy — plenty of room for GAN augmentation to help.

2. **Official GitHub repo with complete code.** https://github.com/GH920/improved-bagan-gp — includes training scripts, evaluation, FID computation, t-SNE visualization. Ready to run.

3. **The architecture is interesting but not too complex.** The autoencoder-initialized GAN with gradient penalty is a clever technique worth learning and presenting. It's more sophisticated than vanilla DCGAN but not as heavy as StyleGAN2.

4. **Trains comfortably on free Colab.** Small images, lightweight model, 3–5 hours per training run.

5. **Clear measurable improvement over baselines.** The paper shows BAGAN-GP outperforms original BAGAN, WGAN-GP, DRAGAN, and ACGAN on imbalanced datasets. Applying it to HAM10000 (which they didn't do) gives you a novel contribution.

6. **Many realistic improvement directions** that a graduate student can implement in 1–2 weeks.

7. **Medically relevant.** Skin cancer classification is a high-impact problem, HAM10000 is the standard benchmark, and the class imbalance problem is clinically real.

### Recommended Architectural Improvement: Self-Attention + Spectral Normalization + EfficientNet Classifier

**Improvement 1 — Self-Attention in Generator (SAGAN-style)**
- Why: Skin lesions have both fine-grained texture (dermoscopic patterns) and global structure (shape, border irregularity). Self-attention captures long-range dependencies that convolutions miss.
- Expected impact: More anatomically coherent synthetic lesion images, especially for minority classes where training data is insufficient to learn global structure.
- Implementation: Add one self-attention layer after the 2nd transposed convolution block.

**Improvement 2 — Replace Gradient Penalty with Spectral Normalization**
- Why: GP requires computing second-order gradients which is expensive. Spectral normalization is cheaper, more stable, and often produces better results (as shown in SNGAN paper). This is a clean ablation: GP vs SN.
- Expected impact: Faster training, potentially better FID scores, more stable convergence.
- Implementation: Apply `torch.nn.utils.spectral_norm` to all discriminator conv layers, remove GP loss term.

**Improvement 3 — EfficientNet-B0 Downstream Classifier (replaces ResNet-50)**
- Why: EfficientNet-B0 is more parameter-efficient and achieves better accuracy on limited medical datasets. Combined with GAN-augmented data, should show clear improvement.
- Expected impact: Higher classification accuracy, especially on minority classes.

**Improvement 4 (bonus) — Selective Augmentation (inspired by HistoGAN)**
- Why: Not all synthetic images are equally useful. Filtering by classifier confidence and feature distance to real images removes harmful synthetic samples.
- Expected impact: Additional 1–3% accuracy boost on top of naive augmentation.

### Experiments to Run

| Experiment | What It Measures |
|---|---|
| Baseline: No augmentation + ResNet-50 | Lower bound accuracy on imbalanced HAM10000 |
| Classical augmentation (flip, rotate, color jitter) + ResNet-50 | Value of traditional augmentation |
| BAGAN-GP (original) + ResNet-50 | Reproduction of original method |
| BAGAN-GP + Self-Attention + ResNet-50 | Effect of attention mechanism |
| BAGAN-GP + Spectral Norm (replacing GP) + ResNet-50 | GP vs SN comparison |
| BAGAN-GP + SA + SN + ResNet-50 | Combined GAN improvements |
| BAGAN-GP + SA + SN + EfficientNet-B0 | Full improvement pipeline |
| Full pipeline + Selective Augmentation | Maximum performance |

### Ablation Studies

| Ablation | What It Tests |
|---|---|
| With vs without autoencoder initialization | Core BAGAN contribution |
| Gradient penalty vs spectral normalization | Regularization strategy |
| With vs without self-attention | Attention mechanism value |
| ResNet-50 vs EfficientNet-B0 | Classifier backbone |
| Augmentation ratio (1x, 2x, 5x, 10x minority) | Optimal synthetic data amount |
| Naive augmentation vs selective augmentation | Quality filtering value |
| Per-class FID comparison | Which classes benefit most from GAN |
| Real-only vs synthetic-only vs mixed training | How synthetic data interacts with real |

### Metrics to Report

- **Classification:** Accuracy, macro F1, per-class precision/recall/F1, AUC-ROC (one-vs-rest), confusion matrix
- **Generation quality:** FID (per-class and overall), IS, t-SNE of real vs synthetic features
- **Imbalance-specific:** Per-class accuracy improvement, minority-class recall improvement
- **Training:** Loss curves, training stability (variance across runs)

### Expected Challenges

1. **HAM10000 download & preprocessing:** Available on Kaggle (ISIC archive). Images are already cropped dermoscopic images — minimal preprocessing needed. Resize to 64×64 or 128×128.
2. **Extreme class imbalance:** The smallest class has ~115 images vs ~6700 for the largest. The GAN must generate meaningful minority-class images from very few examples — this is exactly what BAGAN's autoencoder initialization is designed to handle.
3. **Autoencoder pre-training stability:** The supervised autoencoder with class-aware latent space can be tricky. Use the paper's recommended hyperparameters as starting point.
4. **Evaluation fairness:** Always use the same train/val/test split across all experiments. Use stratified splitting to maintain class ratios.
5. **Colab disconnections:** Checkpoint every 20 epochs to Google Drive.

### 🔧 Free Google Colab Session Plan

| Session | Task | Time |
|---|---|---|
| 1 | Download HAM10000, preprocess, upload to Drive | 1–2 hours |
| 2 | Train supervised autoencoder (BAGAN-GP stage 1) | 1–2 hours |
| 3 | Train BAGAN-GP (stage 2, 200 epochs) | 2–3 hours |
| 4 | Generate synthetic images + train baseline ResNet-50 | 2–3 hours |
| 5 | Train improved GAN (SA + SN, 200 epochs) | 2–3 hours |
| 6 | Generate improved synthetic images + train EfficientNet | 2–3 hours |
| 7 | Ablation runs (augmentation ratios, GP vs SN) | 3–4 hours |
| 8 | Selective augmentation experiment | 2–3 hours |
| 9 | Final evaluation, FID, t-SNE plots, confusion matrices | 2–3 hours |
| **Total** | | **~18–25 hours across 9 sessions (2–3 weeks)** |

### Why This Will Work

The key insight is that HAM10000's extreme imbalance means minority-class accuracy is genuinely poor without augmentation (often <60% for df, vasc classes). GAN augmentation targeting these minority classes should produce a **clear, measurable improvement of 3–10% on minority-class recall** and 2–5% on macro F1. This gives you a strong result to report.

The BAGAN-GP architecture is interesting enough to discuss in a presentation (autoencoder initialization, class-conditional generation, gradient penalty) but simple enough to reproduce in a week. Your improvements (self-attention, spectral norm, better classifier, selective augmentation) are all well-motivated, implementable, and have strong prior evidence of working.

---

### Runner-Up: Semi-Supervised GAN for Melanoma (#10)

If you want a more conceptually interesting project (semi-supervised learning where the discriminator IS the classifier), the Dolatabadi et al. 2025 paper is the second choice. It has official code and tests on 4 datasets. However, BAGAN-GP is safer because the official code is more complete and the imbalance problem guarantees a measurable GAN contribution.

---

*Survey compiled July 2026. Updated after CovidGAN was tested and found to produce no measurable improvement. All GitHub links verified. HAM10000 dataset confirmed available on Kaggle.*
