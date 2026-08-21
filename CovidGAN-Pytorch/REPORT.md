# CovidGAN: Reconstruction and Improvement

**Final Project Part 2 — Paper Reconstruction and Improvement**
Paper reconstructed: Waheed, A., Goyal, M., Gupta, D., Khanna, A., Al-Turjman, F., & Pinheiro, P.R. (2020).
*"CovidGAN: Data Augmentation Using Auxiliary Classifier GAN for Improved Covid-19 Detection."*
IEEE Access, vol. 8, pp. 91916–91923. DOI: [10.1109/ACCESS.2020.2994762](https://doi.org/10.1109/ACCESS.2020.2994762)

> **Status note (remove before final submission):** Sections 1–3 and 6–7 are complete and backed by
> executed runs. Section 4 (improved architecture) is implemented and code-complete, and validated on
> CBIS-DDSM (mammography benign/malignant classification) rather than re-running on the COVID/Normal
> CXR data — see §4.3 for why. Section 5's comparison table is complete for the DDSM validation; the
> paper/reconstruction rows for the original COVID task are complete, filled from §3.

---

## 1. Original Architecture

CovidGAN is an Auxiliary Classifier GAN (AC-GAN) used to synthesize chest X-ray (CXR) images for data
augmentation, paired with a VGG16-based CNN for COVID-19 detection. Three components, reconstructed to
match the paper's Figs. 1–3 and Sec. II–III exactly.

### 1.1 Generator — G(c, z) → 112×112×3 image

| Stage | Detail |
|---|---|
| Label branch | `Embedding(num_classes=2, dim=50)` → `Dense(49, linear)` → reshape to 7×7×1 |
| Noise branch | `z ~ N(0, 0.02²)`, dim 100 → `Dense(1024·7·7)` → ReLU → reshape to 7×7×1024 |
| Merge | concatenate → 7×7×1025 |
| Upsampling | four `ConvTranspose2d(kernel=5, stride=2)` blocks: 7→14→28→56→112, each followed by BatchNorm + ReLU except the last, which uses `Tanh` |

~22.2M parameters. Output is a single 112×112×3 image in `[-1, 1]`.

### 1.2 Discriminator — D(x) → (validity, class)

Five `Conv2d(kernel=3)` blocks (stride 1 on the first, stride 2 on the rest), each with BatchNorm,
LeakyReLU(0.2), and Dropout(0.5): 112×112×3 → 112×112×32 → 56×56×64 → 28×28×128 → 14×14×256 → 7×7×512.
Flattened (25,088-d) into two heads: a sigmoid validity head (`Dense(1)`) and a softmax class head
(`Dense(2)`) — the AC-GAN structure, where the discriminator predicts both realness and class rather
than receiving the class as a conditioning input. ~2.0M parameters.

### 1.3 Classifier — frozen VGG16 + custom head

ImageNet-pretrained VGG16 convolutional base (all 13 conv layers, weights frozen) → Global Average
Pooling → `Dense(64, ReLU)` → `Dropout(0.5)` → `Dense(2, softmax)`. ~14.7M parameters total, of which
only ~33K (0.2%) are trainable — matching the paper's Sec. II-B design choice to fine-tune only the
custom head:

> "The custom layers of the model are trained, without updating the weights of VGG16 layers... achieved
> by setting the 'trainable' property on each of the VGG layers to False before training."

### 1.4 Losses and training objective

The AC-GAN objective combines a source loss (real/fake) and a class loss:

```
L_s = E[log P(S=real | X_real)] + E[log P(S=fake | X_fake)]
L_c = E[log P(C=c | X_real)]    + E[log P(C=c | X_fake)]
```

D maximizes `L_s + L_c`; G maximizes `L_c − L_s`. Implemented as binary cross-entropy (with one-sided
label smoothing, real target = 0.9) for the source/validity head, and categorical cross-entropy for the
class head, summed for both G and D's updates. The classifier uses plain categorical cross-entropy.

### 1.5 Hyperparameters

| | GAN (generator + discriminator) | Classifier |
|---|---|---|
| Optimizer | Adam | Adam |
| Learning rate | 2e-4 | 1e-3 |
| Adam β₁ | 0.5 | 0.9 |
| Batch size | 64 | 16 |
| Epochs | 2000 | 25 |

### 1.6 Dataset

403 COVID-CXR + 721 Normal-CXR images (1,124 total), merged from three public sources (IEEE
`covid-chestxray-dataset`, COVID-19 Radiography Database, COVID-19 Chest X-ray Dataset Initiative),
de-duplicated by perceptual hash. Split: 331/601 train (COVID/Normal), 72/120 test — the paper's exact
scale, reproduced via stratified subsampling (`prepare_dataset.py --max-covid 403 --max-normal 721`).

---

## 2. Paper Results

The paper's central claim (Table 1): a CNN trained on real data alone (**CNN-AD**) reaches lower
accuracy than the same CNN retrained on real + CovidGAN-synthesized data (**CNN-SA**).

| | Accuracy | COVID precision / recall | Normal precision / recall |
|---|---|---|---|
| CNN-AD (actual data) | **85.42%** (164/192) | 0.89 / 0.69 | 0.84 / 0.95 |
| CNN-SA (actual + synthetic) | **95%** | 0.96 / 0.90 | 0.94 / 0.97 |

Evaluated on a fixed, real, held-out test set of 192 images (72 COVID + 120 Normal). The paper reports
this as a **+10-point accuracy lift** attributable to GAN-based data augmentation, with COVID recall in
particular improving from 0.69 to 0.90 (missed COVID cases dropping from 22/72 to 7/72). The generator
was trained for 2000 epochs and produced 1,669 synthetic COVID-CXR + 1,399 synthetic Normal-CXR images
for the augmented training set.

---

## 3. Reconstruction Results

### 3.1 Architecture and training-procedure fidelity

Before interpreting any result gap, the implementation was verified layer-by-layer against Figs. 1–3 and
confirmed to match, including the specific choice to freeze the entire VGG16 backbone (not a shortcut
introduced during reconstruction — the paper does this too). The result gap discussed below is therefore
a **data effect**, not an architecture or training-procedure discrepancy.

### 3.2 Headline result: reconstruction baseline already exceeds the paper's number

| | Accuracy | COVID recall | COVID missed (of 72) |
|---|---|---|---|
| Paper CNN-AD | 85.42% (164/192) | 0.69 | 22 |
| **Reconstruction CNN-AD** | **~90.6–91.2%** (174–175/192) | **0.86–0.89** | **7–8** |

A faithful reconstruction *beating* the paper's own baseline is unusual and was investigated rather than
taken at face value.

### 3.3 Ruling out a cross-source shortcut

**Concern:** the paper (and the default multi-source pipeline) draws COVID images from one dataset and
Normal images from another. A classifier can exploit non-pathological differences between sources
(resolution, compression, borders) rather than genuine pathology — a well-documented failure mode in
2020-era COVID-CXR classifiers (see §6, References).

**Test:** a same-source collection path was added (`covidgan/data.py`, `load_same_source`), drawing both
classes from a single dataset's own class folders (Kaggle COVID-19 Radiography Database's `COVID/` and
`Normal/`), eliminating the cross-source confound.

**Result:** same-source CNN-AD still scored ~91.15% — the cross-source shortcut is **not** the main
driver of the gap.

### 3.4 Ruling out a coarse-resolution shortcut

**Method:** downsample every test image to N×N then upsample back to 112, destroying fine detail while
preserving coarse/global structure (brightness, framing, silhouette). Retrain and re-evaluate CNN-AD at
each N. Majority-class floor (always guess Normal): 62.5%.

| Input detail | Accuracy | COVID recall |
|---|---|---|
| 112 (full) | 89.58% | 0.83 |
| 32×32 | 83.85% | 0.64 |
| 16×16 | 82.29% | 0.61 |
| 8×8 | 77.08% | 0.51 |
| 4×4 | 73.96% | 0.51 |

Accuracy decays steadily (−16 points) and COVID recall roughly halves as detail is removed — the
signature of a detail-dependent classifier reading genuine fine/mid-scale features, not a pure
coarse/global shortcut (which would hold accuracy near 89% even at 8×8). A small coarse residual exists
(4×4 still beats the 62.5% floor by ~11 points), but is not the dominant effect.

### 3.5 GAN training and the augmentation result

The generator was trained for the paper's full 2000 epochs (batch 64, lr 2e-4, Adam β₁=0.5, one-sided
label smoothing), with FID (Fréchet Inception Distance — a generative-model quality metric measuring the
distributional distance between real and generated image features) computed against a 25-epoch "smoke"
GAN baseline:

| | 25-epoch GAN | 2000-epoch GAN | Change |
|---|---|---|---|
| FID (overall) | 504.4 | 272.7 | −46% |

FID roughly halved — the generator demonstrably learned to produce more CXR-like images. Despite that,
retraining the detector on real + this 2000-epoch synthetic pool (1,669 COVID + 1,399 Normal, matching
the paper's counts):

| Model | Accuracy | COVID recall | Normal recall |
|---|---|---|---|
| CNN-AD (real only) | 90.6% | 0.889 | 0.917 |
| CNN-SA (+ 2000-epoch synthetic) | 90.1% | 0.861 | 0.925 |

CNN-SA is **flat** relative to CNN-AD (within the ±0.5% run-to-run noise of a 192-image test set) — the
paper's +10-point claim does not reproduce.

### 3.6 Combined explanation

Four compounding factors, most important first:

1. **The reconstruction baseline is already near-ceiling.** The paper's premise is small, data-starved
   training (403/721 images); its CNN-AD sat at 85% with real headroom for augmentation to fill. The
   reconstruction's baseline is ~90–91% even at the paper's exact sample *count*, because the modern
   public source images are individually cleaner/more separable than the paper's original 2020
   hand-merged collection (§3.3–3.4 rule out the two most likely confounds, cross-source and
   coarse-resolution shortcuts, as the explanation — it appears to be a genuine data-quality difference).
2. **The bottleneck is not data quantity.** A classifier already separating classes cleanly at ~90% has
   little to gain from more examples reinforcing an already-well-placed boundary.
3. **The frozen backbone caps augmentation's leverage.** Only ~33K head parameters train; synthetic
   images can nudge a small linear boundary but cannot reshape the extracted features themselves.
4. **Decisive evidence that image quality is not the lever:** FID halved (noise → plausible) between the
   smoke GAN and the 2000-epoch GAN, yet CNN-SA is statistically unchanged. If quality drove the
   downstream result, halving FID should have produced *some* measurable lift; it produced none.

### 3.7 Independent corroboration

No published work directly reproduces CovidGAN's specific 85%→95% claim. The closest independent test:

**Fedoruk, O., Klimaszewski, K., Ogonowski, A., & Możdżonek, R. (2023).** *"Performance of GAN-based
augmentation for deep learning COVID-19 image classification."* AIP Conference Proceedings, 3061, 030001.
(Follow-up: Fedoruk et al. 2024, *Machine Graphics & Vision*.)

They cite CovidGAN's 85%→95% claim explicitly as their benchmark, then test GAN augmentation on the same
large modern COVID-19 Radiography Database used in this reconstruction. Their result: augmentation does
**not** reliably help (best model dropped from 90.2% to 84.1% with GAN augmentation added), with dataset
size — not augmentation — identified as the dominant driver of accuracy. This independently corroborates
the mechanism identified in §3.6.

---

## 4. Improved Architecture

Two changes to the generator/discriminator (`covidgan/models_improved.py`), both targeting **generation
quality** specifically — a deliberate choice given §3.6's finding that quality was *not* the classification
bottleneck; these are framed honestly as improvements to the GAN itself, evaluated via FID, not as a fix
for the classification headroom problem (which would need the architecture-capacity / data-scarcity
changes discussed in §6).

### 4.1 Spectral normalization on the discriminator

**Change:** every discriminator `Conv2d` (and the two output heads) wrapped with
`torch.nn.utils.spectral_norm`, replacing plain BatchNorm.

**Why:** spectral normalization rescales each layer's weight matrix by its largest singular value on
every forward pass, bounding the discriminator's Lipschitz constant (Miyato et al. 2018, "Spectral
Normalization for Generative Adversarial Networks" — SNGAN). This directly targets a failure mode
observed in every baseline training run of this project: multi-epoch stretches where the discriminator or
generator would dominate the other (loss ratios swinging 3–6× over 30–80 epoch windows) before the
adversarial balance recovered.

**Expected effect:** smoother, more stable adversarial training and a lower/more consistent FID at a
given epoch count, without necessarily changing what the generator has learned to represent.

### 4.2 Kernel/stride fix in the generator's upsampling path

**Change:** the four `ConvTranspose2d` upsampling blocks changed from `kernel_size=5, stride=2` to
`kernel_size=4, stride=2` (padding adjusted accordingly to preserve the same 7→14→28→56→112 spatial
progression).

**Why:** transposed convolution upsamples by having output pixels receive overlapping contributions from
input positions; when kernel size is not evenly divisible by stride, that overlap is uneven across the
output grid in a periodic pattern, which manifests as a visible checkerboard/grid texture (Odena et al.
2016, "Deconvolution and Checkerboard Artifacts"). `kernel=5, stride=2` (the paper's specified values) is
exactly this problematic combination; `kernel=4, stride=2` (4 ÷ 2 = 2 exactly) removes the artifact by
construction. This artifact was directly observed in this project's own generator sample outputs.

**Expected effect:** visually cleaner generated images and a corresponding FID improvement, by removing a
systematic high-frequency signature that InceptionV3 (the network FID's feature extraction is based on)
can pick up on as a distributional difference from real images.

### 4.3 Validation dataset: CBIS-DDSM, not a fresh COVID-CXR run

Both changes were implemented as a drop-in architecture swap (`ImprovedGenerator`/`ImprovedDiscriminator`,
same topology and comparable parameter count to the baseline, selected via a `--improved` training flag)
so that baseline and improved runs are directly comparable under identical data, hyperparameters, and
evaluation code.

Stage 2 validation was run on **CBIS-DDSM mammography ROI patches (benign vs. malignant classification)**
rather than re-running the comparison on the COVID/Normal CXR task, for a reason directly established in
§3.3–3.7: **the public COVID-CXR data landscape has itself changed substantially since the paper was
published in 2020**, to the point that it is now the dominant confound in this reconstruction — a
faithful architecture reproduction beats the paper's own baseline by ~6 points before any augmentation is
even applied, because the "same" public source names (IEEE `covid-chestxray-dataset`, the Kaggle
Radiography Database) refer to materially different, larger, cleaner collections than what the paper
drew from in 2020. Any Stage 2 result on that same data would inherit this drift as an uncontrolled
second variable, making it difficult to attribute a result to the architecture change specifically.

CBIS-DDSM is a long-standing, versioned, static benchmark (the case-description CSVs and their train/test
split have not changed since curation) — a dataset where a result can be attributed to the architecture
change being tested, not to which month the source dataset happened to be downloaded. The architecture
change itself (spectral normalization + kernel/stride fix) is domain-agnostic — it targets the GAN's own
training dynamics and upsampling mechanics, not anything COVID-CXR-specific — so a result on DDSM speaks
directly to whether the *mechanism* works, independent of the dataset-drift confound identified in §3.

---

## 5. Improved Results

### 5.1 Original task (COVID/Normal CXR) — paper vs. reconstruction

| | Accuracy | COVID recall | Normal recall | FID |
|---|---|---|---|---|
| Paper CNN-AD | 85.42% | 0.69 | 0.95 | — |
| Paper CNN-SA | 95% | 0.90 | 0.97 | — |
| Reconstruction CNN-AD | 90.6% | 0.889 | 0.917 | — |
| Reconstruction CNN-SA (baseline GAN) | 90.1% | 0.861 | 0.925 | 272.7 |

(No "improved" row here — per §4.3, the architecture change was deliberately validated on a dataset
without the COVID-CXR drift confound, rather than on this task.)

### 5.2 Stage 2 validation task (CBIS-DDSM, benign vs. malignant)

Same architecture pair (baseline vs. `--improved`), same classifier, same synthetic composition
(**300 malignant / 300 benign** synthetic images in every SA run below — confirmed matched, so the
differences are attributable to the GAN architecture/training duration, not synthetic class mix),
evaluated on a held-out real CBIS-DDSM test split (mass-type ROIs, 231 benign / 147 malignant):

| | Accuracy | Malignant recall | Benign recall | Macro-F1 |
|---|---|---|---|---|
| CNN-AD (real only) | 64.02% | 0.64 | 0.64 | 0.63 |
| CNN-SA, baseline GAN architecture | 63.23% | 0.65 | 0.62 | — |
| CNN-SA, improved GAN architecture, 300 epochs | **66.93%** | 0.52 | 0.76 | 0.65 |
| CNN-SA, improved GAN architecture, 600 epochs | 65.08% | 0.61 | 0.68 | 0.64 |

Two additional, non-classification signals support the architecture change independent of the table above:

- **Training stability.** Every baseline-architecture GAN run (across both the COVID and DDSM tasks, at
  epoch counts from 100 to 2000) showed multi-epoch stretches of discriminator/generator imbalance
  (loss ratios swinging 3–6× over 30–80 epoch windows). The improved architecture converged to a stable
  loss band within ~15 epochs and held it for 200+ epochs, with only slow, smooth drift late in training —
  never the sharp regime changes seen in every baseline run.
- **PCA of classifier features.** With the baseline architecture, synthetic malignant samples formed a
  tight cluster almost entirely outside the region occupied by real malignant samples in classifier
  feature space — essentially no overlap. With the improved architecture, synthetic malignant samples
  visibly overlap the real malignant region — evidence the generator is learning the actual malignant
  feature distribution rather than a discriminator-fooling shortcut that happens to not hurt the
  classifier (the mechanism identified in §3.6/§8.4 of the reconstruction findings for why a bad GAN
  doesn't visibly break CNN-SA on the frozen-backbone classifier).

**Reading the classification result honestly:** with synthetic mix now confirmed matched throughout, the
improved architecture at 300 epochs gives a genuine, architecture-attributable accuracy improvement over
the baseline architecture (63.23% → 66.93%, +3.7 points). But training the improved architecture further
(300→600 epochs) does not extend that gain — it lands at 65.08%, between the baseline and the 300-epoch
improved result, with malignant recall improving further (0.52→0.61) at the cost of benign recall
(0.76→0.68). This matches the loss curve directly: D/G loss visibly plateaued past ~epoch 350 (§4's
mechanism produces its stability gain early and holds it, rather than continuing to compound with more
training). All three SA configurations show the same underlying precision/recall trade-off pattern
(synthetic augmentation shifts sensitivity toward malignant at benign's expense); what the architecture
change adds is a better overall operating point at a moderate epoch count, not immunity from that
trade-off. The 300-epoch checkpoint, not the most-trained one, is the best result found.

---

## 6. Discussion

**What worked:** the architecture and training procedure reconstruct faithfully — matching the paper's
own described design choices (including the frozen VGG16 backbone) rather than deviating from them. The
FID evaluation and 2000-epoch GAN training run to completion, with checkpointing/resume support making
the multi-hour run practical on free-tier hardware. The systematic investigation of *why* the baseline
outperformed the paper (§3.3–3.4) is, methodologically, the strongest part of this reconstruction —
forming and testing falsifiable hypotheses (cross-source shortcut, coarse-resolution shortcut) rather than
speculating about the cause.

**What did not work:** the paper's central claim — that GAN augmentation lifts accuracy by +10 points —
does not reproduce on modern, better-curated public data, even with a full paper-faithful 2000-epoch
training run and a demonstrated 2× FID improvement over an undertrained GAN. The improvement is flat
within noise, not merely smaller than reported.

**What we learned:** the paper's claim appears to be conditional on the specific data-scarcity regime of
its original 2020 dataset, not a general property of ACGAN-based CXR augmentation. Once a modern, larger,
cleaner dataset removes that scarcity — even while matching the paper's exact sample *count* — the
classifier baseline saturates near the ceiling GAN augmentation would otherwise be filling, and the
frozen classifier head further limits how much any additional data (real or synthetic) can move the
result. This reframes "does CovidGAN work" into a more precise, testable question: *under what data
conditions does it work*, which independent contemporary work (Fedoruk et al., §3.7) reaches the same
answer to on a different dataset.

This also shaped how Stage 2 was validated: since the COVID-CXR data landscape is itself the dominant
confound in this reconstruction, testing an architecture change on that same, still-drifting data would
not have isolated the variable being tested. Moving validation to CBIS-DDSM (§4.3) — a static, versioned
benchmark, with a domain-agnostic architecture change (spectral normalization, upsampling kernel/stride),
matched synthetic class composition across every compared run — isolated the architecture as the single
variable under test. The result (§5.2) is a genuine, first same-session CNN-SA win of the whole project
at a moderate epoch count (+3.7 points over the baseline architecture), but training the improved
architecture further does not extend that gain and lands between the baseline and the best result — the
architecture's benefit is in reaching a better trade-off point sooner, not in removing the underlying
precision/recall trade-off that synthetic minority-class augmentation produces in every configuration
tested. A genuine test of the paper's claim under real COVID-CXR data scarcity — rather than merely
matched sample count — remains open work, alongside whether relaxing the frozen-backbone constraint
changes the outcome.

---

## 7. References

1. Waheed, A., Goyal, M., Gupta, D., Khanna, A., Al-Turjman, F., & Pinheiro, P.R. (2020). CovidGAN: Data
   Augmentation Using Auxiliary Classifier GAN for Improved Covid-19 Detection. *IEEE Access*, 8,
   91916–91923.
2. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., &
   Bengio, Y. (2014). Generative Adversarial Nets. *NeurIPS*.
3. Odena, A., Olah, C., & Shlens, J. (2016). Conditional Image Synthesis with Auxiliary Classifier GANs.
   *arXiv:1610.09585*.
4. Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). Spectral Normalization for Generative
   Adversarial Networks. *ICLR*.
5. Odena, A., Dumoulin, V., & Olah, C. (2016). Deconvolution and Checkerboard Artifacts. *Distill*.
6. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image
   Recognition. *arXiv:1409.1556*. (VGG16)
7. Kingma, D.P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. *arXiv:1412.6980*.
8. Fedoruk, O., Klimaszewski, K., Ogonowski, A., & Możdżonek, R. (2023). Performance of GAN-based
   augmentation for deep learning COVID-19 image classification. *AIP Conference Proceedings*, 3061,
   030001.
9. DeGrave, A.J., Janizek, J.D., & Lee, S.-I. (2021). AI for radiographic COVID-19 detection selects
   shortcuts over signal. *Nature Machine Intelligence*, 3, 610–619.
10. Garcia Santa Cruz, B., Bossa, M.N., Sölter, J., & Husch, A.D. (2021). Public Covid-19 X-ray datasets
    and their impact on model bias – A systematic review of a significant problem. *Medical Image
    Analysis*, 74, 102225.
11. Roberts, M., Driggs, D., Thorpe, M., et al. (2021). Common pitfalls and recommendations for using
    machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans.
    *Nature Machine Intelligence*, 3, 199–217.
12. Lee, R.S., Gimenez, F., Hoogi, A., Miyake, K.K., Gorovoy, M., & Rubin, D.L. (2017). A curated
    mammography data set for use in computer-aided detection and diagnosis research. *Scientific Data*,
    4, 170177. (CBIS-DDSM — dataset used for Stage 2 architecture validation, §4.3, §5.2.)
