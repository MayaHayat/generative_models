# DDSM-ACGAN — all tests and results

*DDSM only. The COVID-CXR sub-project is deliberately excluded.*

*Every number is pulled from Weights & Biases (`maya-hayat-ariel-university/ddsm-acgan`) with the
run ID given, so anything here can be traced back to its run.*

> ⚠️ **Status:** `FINDINGS.md` and `STAGE2_FINDINGS.md` still state numbers that §6 below retracts.
> They have not been corrected yet. This document is the current state of truth.

---

## 1. The task

| | |
|---|---|
| Data | CBIS-DDSM mammography ROI patches, **mass-type only** (calcifications excluded) |
| Task | Benign vs Malignant |
| Train / Test | 1,318 / 378 (231 benign, 147 malignant) |
| Split | CBIS-DDSM's own official case-description CSVs, not re-randomized |
| Always-guess-benign floor | **61.1%** — any result must beat this to mean anything |

**The question being tested:** does adding GAN-generated fake ROIs to the training set improve the
classifier?

| Term | Meaning |
|---|---|
| **AD** | Trained on **real images only**. The baseline. |
| **SA** | Trained on **real + synthetic**. Should beat AD if augmentation works. |
| **Frozen** | VGG16 backbone locked; ~33K trainable params (the original paper's design) |
| **Unfrozen (uf=2)** | Top 2 VGG16 blocks train too; millions of params |
| **Pool** | The 600 fake images (300 benign + 300 malignant) a generator produces |
| **Pool draw** | *Which* random noise vectors made that pool — this turned out to matter a lot |

---

## 2. Two data bugs found and fixed before any result counted

| Bug | Effect | Fix |
|---|---|---|
| Mask folders globbed as images | Silently doubled image counts with binary segmentation masks | Exclude any path containing `mask(s)`/`label(s)` |
| **16-bit PNGs destroyed on load** | CBIS-DDSM ROIs are `I;16`; `PIL.convert("RGB")` truncates them to near-white — every training image was effectively blank | Per-image min-max normalise before RGB conversion |

The second one explains both the early "collapsed"-looking GAN samples and a classifier that could
barely beat the majority-class floor. **All results below post-date both fixes.**

---

## 3. The three generators being compared

| Name | Architecture | Epochs | Run ID |
|---|---|---|---|
| **baseline** | Paper-faithful (kernel 5, BatchNorm D) | 300 | `fil0gesf` |
| **improved ep300** | Spectral-norm D + kernel 4 upsampling | 300 | `62aw102a` |
| **improved ep600** | Same architecture, trained longer | 600 | `cugo455s` |

The "improved" architecture makes two changes: **spectral normalisation** on the discriminator
(bounds its Lipschitz constant, targeting training instability) and **kernel 5→4** in the generator's
upsampling (removes checkerboard artifacts by construction, since 4÷2 divides evenly).

---

## 4. Test 1 — Does the improved architecture train more stably? ✅ YES

Same data, same hyperparameters, only the architecture differs:

| | D_loss variability | G_loss variability |
|---|---|---|
| Baseline (full 300 epochs) | 0.351 | 0.985 |
| Baseline (last 50 epochs) | 0.246 | 0.741 |
| **Improved (epochs 201–300)** | **0.031** | **0.025** |
| **Improved (epochs 351–600)** | **0.026** | **0.018** |

- ✅ **10–30× lower loss variance.** Not "looks calmer" — a measured order-of-magnitude reduction.
- The baseline never settles; it swings 3–6× over 30–80 epoch windows at every epoch count tested.
- ⚠️ This is a claim about *training dynamics only*. Whether it produces better images, or a better
  classifier, are separate questions — tested in §5 and §7. **They do not follow automatically.**

---

## 5. Test 2 — Does the improved architecture make better images? ❌ NO

### 5.1 The full training curve (run `1fzjlyh6`)

Every saved checkpoint of both architectures, one 600-image pool each, all scored against the same
real reference at n=600. KID, lower = better:

| epoch | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** | 0.370 | 0.377 | 0.215 | 0.184 | **0.165** ⭐ | 0.317 💥 | — | — | — | — | — | — |
| **improved** | 0.412 | 0.325 | 0.273 | 0.407 | 0.390 | 0.341 | 0.385 | 0.285 | 0.276 | 0.218 | 0.235 | **0.201** |

### 5.2 At matched epochs the baseline wins 5 of 6

| epoch | baseline | improved | winner | σ |
|---|---|---|---|---|
| 50 | 0.370 | 0.412 | baseline | 2.0 |
| 100 | 0.377 | 0.325 | improved | 2.7 |
| 150 | 0.215 | 0.273 | baseline | 3.0 |
| 200 | 0.184 | 0.407 | **baseline** | **10.1** |
| 250 | 0.165 | 0.390 | **baseline** | **9.9** |
| 300 | 0.317 | 0.341 | baseline (weak) | 1.3 |

**The best generator in the entire project is the paper-faithful baseline at epoch 250**
(KID 0.165), beating the improved architecture's best (ep600, 0.201) using 350 fewer epochs.

### 5.3 An earlier single-checkpoint comparison got this backwards

A first pass (run `nhq3jrms`) compared baseline **ep300** against improved **ep600** and reported the
improved architecture winning by −36%, 5.5σ. The curve shows what those two checkpoints actually are:

- baseline ep300 (0.317) = **the baseline's worst checkpoint after epoch 100** — a +92% jump from
  ep250's 0.165
- improved ep600 (0.201) = **the improved architecture's best checkpoint**

That comparison was worst-vs-best, not architecture-vs-architecture. ❌ **The −36% quality win is
retracted.**

Note that `FINDINGS.md` §5's original intuition — that the baseline ep300 checkpoint was "caught
mid-swing in a bad phase" — turns out to be **correct for generation quality**, even though §6.2
below shows it is wrong for classification. Right instinct, wrong metric.

### 5.4 Loss stability does not imply output stability

The improved architecture has 10–30× lower *loss* variance (§4) yet its *quality* swings just as
violently — ep150→200 is a +49% KID jump. The baseline, whose losses oscillate wildly, produced the
best generator in the project.

**A smooth loss curve carried no information about whether the generator was any good.** This is a
real and transferable finding.

### 5.5 Single-checkpoint comparisons are unreliable

| Architecture | Biggest swing between adjacent checkpoints (50 epochs apart) |
|---|---|
| baseline | **+92%** (ep250→300) |
| improved | **+49%** (ep150→200) |

Against error bars of only ±0.015. Any conclusion drawn from one checkpoint per architecture is
measuring where training happened to be sampled — the same failure mode as single pool draws (§6.1)
and single classifier seeds.

😐 In absolute terms every checkpoint is poor (well-trained GANs score FID < 50). None of these
generators produces realistic mammography.

---

## 6. The problem: no random seeds, and a noise floor nobody had measured

Every early result ran without fixed seeds. Once seeds were added, results changed.

### 6.1 The noise floor

| Noise source | Magnitude |
|---|---|
| Classifier random seed | up to **1.6 points** |
| **Which 600 noise vectors formed the pool** | up to **5.3 points** |

**No effect anyone was chasing exceeded 5 points.** Results were substantially measuring randomness.

### 6.2 ❌ Retraction 1 — `FINDINGS.md` §5's "+6.9-point win"

| | baseline SA | improved SA | gap |
|---|---|---|---|
| Unseeded (the claim) | 60.05 (`jkvaptex`) | 66.93 (`y2bvl1oo`) | **+6.88** |
| Seed 0 | 65.34 (`430w9r7n`) | 65.61 (`j20fo7ro`) | +0.27 |
| Seed 1 | 65.34 (`hjozkvn0`) | 64.02 (`m95w4t67`) | −1.32 |
| **Mean** | **65.34** | **64.81** | **−0.52** |

One unlucky draw vs one lucky draw. `FINDINGS.md` §5 blamed a "bad generator checkpoint caught
mid-swing" — but the seeded runs load **that exact same checkpoint file** and score 5.3 points
higher. The checkpoint was never the variable; the noise draw was.

### 6.3 ❌ Retraction 2 — `STAGE2_FINDINGS.md` §2's "+1.33-point reproducible win"

Claimed: with the unfrozen classifier, augmentation gives a real win. Re-tested across **3
independent pool draws**, classifier seed held fixed so only the pool varies:

| Pool draw | baseline SA | improved300 SA |
|---|---|---|
| 1 | 70.63 (`5y8ju407`) | 67.46 (`tmhjvs29`) |
| 2 | 67.46 (`p60qeg51`) | 67.20 (`fte4werz`) |
| 3 | 68.78 (`4cbd3vvk`) | 64.02 (`k8jnco8o`) |
| **Mean** | **68.96** | **66.23** |
| AD (real only), same seed | **69.05** (`m53jdvq1`) | — |

- **SA (68.96) vs AD (69.05) = −0.09.** No lift at all. The 70.63 was the lucky draw.
- baseline − improved = 2.73 pts, but the *within-arm* spread is 3.2–3.4 pts.
  **The noise is bigger than the effect** (t=1.90, p ≈ 0.13). Not established.
- Direction is consistent though: baseline beats improved in **all 3** paired draws.

---

## 7. Test 3 — Does augmentation help? ❌ NO

### 7.1 Unfrozen classifier (best current estimates)

| Training data | Accuracy |
|---|---|
| **Real only (AD)** | **69.44** |
| Real + baseline-GAN fakes | 68.96 |
| Real + improved ep600 fakes | 67.60 |
| Real + improved ep300 fakes | 66.23 |

### 7.2 Frozen classifier (the paper's original design)

| Training data | Seed 0 | Seed 1 | Mean |
|---|---|---|---|
| **Real only (AD)** | 66.14 | 65.87 | **66.00** |
| Real + baseline fakes | 65.34 | 65.34 | 65.34 |
| Real + improved ep300 fakes | 65.61 | 64.02 | 64.81 |
| Real + improved ep600 fakes | 65.61 | 65.61 | 65.61 |

➡️ **In both classifier configurations, adding synthetic images never beats real-only. Best case
is a tie.**

---

## 8. Test 4 — Does the classifier capacity change help? ✅ YES

| Classifier | AD accuracy (mean of 2 seeds) |
|---|---|
| Frozen (paper's design) | 66.00 |
| **Unfrozen, top 2 blocks** | **69.44** |

- ✅ **+3.44 points**, consistent at both seeds individually (66.14→69.84, 65.87→69.05).
- This is the **only** change in the project that reliably improves accuracy — and it has nothing to
  do with the GAN. Giving the classifier room to adapt its features to mammography texture helps.

---

## 9. Test 5 — The punchline: image quality vs. usefulness

The three pools that have *both* a quality score and a classification result:

| Pool | Image quality (KID) | Classifier accuracy |
|---|---|---|
| improved ep600 | 🥇 **best** (0.201) | 🥈 middle (67.60) |
| baseline ep300 | 🥈 middle (0.317) | 🥇 **best** (68.96) |
| improved ep300 | 🥉 worst (0.341) | 🥉 worst (66.23) |

**The rankings do not match. The best generator is not the best augmenter.**

This decoupling is the finding that has survived every round of re-testing — unlike the architecture
claims, it has never reversed. It is now *strengthened* by §5: the single best generator in the
project (baseline ep250, KID 0.165) was never even evaluated for classification, because nothing in
the project's design connected generation quality to the augmentation question.

⚠️ The earlier framing of this section — "a 36% quality improvement produced a 1.4-point accuracy
loss" — rested on the retracted ep300-vs-ep600 quality comparison (§5.3). The decoupling itself
stands on the rank mismatch above, which does not depend on that comparison.

---

## 10. Supporting probes

### 10.1 Can a classifier tell fake from real? — Yes, perfectly, for every pool

Held-out real-vs-synthetic discrimination at uf=2 capacity (`4rokgp51`, `6pr4el0n`, `3fogi0cj`,
`wqhkngnz`):

| Pool | Held-out accuracy | AUC |
|---|---|---|
| baseline | 100% | 1.00 |
| improved ep300 | 100% | 1.00 |
| improved ep600 | 100% | 1.00 |

Saturated — the probe can't rank them. But it does establish that **none of these pools is anywhere
near the real data distribution**, consistent with the poor absolute FID/KID values in §5.

### 10.2 Train on fakes alone, test on real

| Pool | Accuracy | Malignant recall |
|---|---|---|
| baseline (`14vh6zaj`) | 60.05% | 0.109 |
| improved ep300 (`1ppw8hwf`) | 59.79% | 0.231 |
| improved ep600 (`oy40wpcg`) | 65.08% | 0.340 |

All at or barely above the 61.1% always-guess-benign floor. The baseline pool in particular produces
a near-degenerate classifier (0.109 malignant recall). **The synthetic images carry very little
transferable signal on their own.**

---

## 11. Summary — what is actually true

### ✅ Holds up

- **Unfreezing the classifier helps: +3.44 points**, reproduces across seeds. The project's only
  reliable accuracy gain.
- **The improved GAN architecture trains far more stably** — 10–30× lower *loss* variance.
- **Loss stability does not imply output quality** — the improved architecture's KID swings as
  violently as the baseline's despite its smooth loss curve, and the baseline (unstable losses)
  produced the project's best generator.
- **Augmentation does not help.** In every configuration tested, real-only matches or beats
  real+synthetic.
- **Image quality and augmentation value are decoupled** — quality and accuracy rank differently
  across pools. The finding that has survived every round of re-testing.
- **Uncontrolled sampling dominated every comparison** — ~5 accuracy points from the pool draw,
  and 49–92% KID swings between adjacent checkpoints. Larger than every effect studied.

### ❌ Retracted

- `FINDINGS.md` §5's +6.9-point matched-epoch win → **−0.52 with seeds** (§6.2)
- `STAGE2_FINDINGS.md` §2's +1.33-point augmentation win → **−0.09 across 3 pool draws** (§6.3)
- **The improved architecture's −36% image-quality win** → worst-vs-best checkpoint selection; at
  matched epochs the baseline wins 5 of 6 (§5.2, §5.3)
- `ANALYSIS.md` §4.2's "manifold overlap" mechanism — failed three separate tests

### The Stage 2a scorecard

| Claim | Status |
|---|---|
| Improved architecture trains more stably (loss variance) | ✅ True |
| Improved architecture produces better images | ❌ **False** — baseline wins at matched epochs |
| Improved architecture helps classification | ❌ False |

The honest Stage 2a conclusion: **spectral normalisation + the kernel/stride fix stabilised the loss
curve and improved neither generation quality nor downstream accuracy.** A clean negative result
about a well-motivated intervention.

### 🤷 Unresolved

- Improved-GAN augmentation looks ~2.7 pts worse than baseline-GAN augmentation, consistent in
  direction across all 3 paired draws, but **inside the noise** (p ≈ 0.13). No mechanism identified.

---

## 12. One-sentence version

> On CBIS-DDSM, GAN augmentation does not improve benign/malignant classification in any
> configuration tested, and the Stage 2 architecture change stabilised the loss curve without
> improving either image quality or accuracy — the only reliable gain came from giving the
> classifier more capacity, and every other reported gain, including our own, turned out to be a
> single sample from a high-variance process that nobody had measured.

---

## 13. Outstanding corrections

| Document | Section | Action |
|---|---|---|
| `FINDINGS.md` | §5 | Retract; replace with §6.2 above |
| `FINDINGS.md` | §4.5 | Downgrade — shares the uncontrolled pool draw |
| `FINDINGS.md` | §6 | Remove "strongest, most controlled result" framing |
| `STAGE2_FINDINGS.md` | §2 | Retract finding #2 per §6.3 |
| `STAGE2_FINDINGS.md` | §3 | Note the interaction hypothesis is now unsupported |
| `ANALYSIS.md` | §4.2 | Retire the mechanism |
