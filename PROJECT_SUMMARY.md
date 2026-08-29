# Project Summary — every test and result, from the beginning

*Plain-language orientation document covering both sub-projects (`CovidGAN-Pytorch` and
`DDSM-ACGAN-Pytorch`), what was tested, what survived verification, and what was retracted.*

*Read this first. The detailed documents are `CovidGAN-Pytorch/REPORT.md`,
`DDSM-ACGAN-Pytorch/FINDINGS.md`, `DDSM-ACGAN-Pytorch/STAGE2_FINDINGS.md`, and the audit in
`DDSM-ACGAN-Pytorch/ANALYSIS.md`.*

> ⚠️ **Status:** `FINDINGS.md`, `STAGE2_FINDINGS.md`, and `REPORT.md` still contain numbers that
> §7 of this document retracts. They have not yet been corrected. This summary is the current
> state of truth; those documents are not.

---

## 1. What the project is

Two parts, one assignment:

| Part | Goal |
|---|---|
| **Part 1 — Reconstruction** | Rebuild the CovidGAN paper (Waheed et al. 2020) exactly and test whether its result reproduces |
| **Part 2 — Improvement** | Change something in the architecture and measure whether it helps |

The paper's claim: a GAN generates synthetic medical images → you add them to the training set →
the classifier improves. Specifically **85.42% → 95%**, a +10-point jump.

---

## 2. Vocabulary

Everything below depends on these.

| Term | Meaning |
|---|---|
| **AD** | "Actual Data" — classifier trained on **real images only**. The baseline. |
| **SA** | "Synthetic Augmented" — classifier trained on **real + GAN-generated images**. |
| **The claim** | SA should beat AD. That is the entire point of the paper. |
| **ACGAN** | The GAN. Generator makes fake images; Discriminator judges real/fake *and* predicts class. |
| **Frozen classifier** | VGG16 backbone locked; only ~33K parameters train. What the paper did. |
| **Unfrozen (uf=2)** | Top 2 VGG16 blocks train too — millions of parameters. The Stage 2b change. |
| **Pool** | The batch of 600 fake images (300 benign + 300 malignant) a generator produces. |
| **Pool draw / z-seed** | *Which* random noise vectors produced that pool. This turned out to matter enormously. |
| **FID / KID** | Image-quality metrics. Lower = fakes look more like real. KID is reliable at small n; FID is not. |

---

## 3. The two datasets

| | Dataset 1 | Dataset 2 |
|---|---|---|
| Name | COVID / Normal chest X-rays | CBIS-DDSM mammography ROIs |
| Task | COVID vs Normal | Benign vs Malignant |
| Train / Test | 932 / 192 | 1,318 / 378 (231 benign, 147 malignant) |
| Why used | The paper's own task | Static, versioned benchmark — COVID data has drifted since 2020 |

---

## 4. Part 1 results — CovidGAN reconstruction

### 4.1 Does the paper's claim reproduce?

| | Accuracy | COVID recall |
|---|---|---|
| Paper AD | 85.42% | 0.69 |
| Paper SA | **95%** | 0.90 |
| **Reconstruction AD** | **90.6%** | 0.889 |
| **Reconstruction SA** | **90.1%** | 0.861 |

- ❌ **The +10-point claim did not reproduce.** SA was flat, even slightly down.
- ⚠️ The *baseline* already beat the paper's *final* result — which required explaining.

### 4.2 Why is the baseline so high? Two hypotheses, both tested

| Hypothesis | Test | Result |
|---|---|---|
| Classifier cheats by detecting which *source dataset* each class came from | Draw both classes from one source | Still 91.15% → **not the cause** |
| Classifier only reads coarse features (brightness, framing) | Downsample to 32/16/8/4 px, retrain | Accuracy falls 89.6% → 74% → **not the cause** |

**Conclusion:** modern public COVID data is genuinely cleaner than the paper's 2020 collection. The
baseline sits near ceiling, so augmentation has no headroom to fill.

### 4.3 Was the GAN simply undertrained?

| GAN training | FID (lower = better) |
|---|---|
| 25 epochs | 504.4 |
| 2000 epochs | **272.7** (−46%) |

- ✅ The GAN demonstrably improved.
- ❌ SA accuracy did not move.
- 💡 **First appearance of the project's central finding: better fake images ≠ better classifier.**

---

## 5. Part 2 — two separate "Stage 2" changes

This is the single biggest source of confusion in the documents: **two different improvements both
got called "Stage 2."**

| | Stage 2a — **GAN** change | Stage 2b — **Classifier** change |
|---|---|---|
| What | Spectral norm on D; upsampling kernel 5→4 | Unfreeze top 2 VGG16 blocks + BatchNorm head |
| Fixes | Unstable GAN training; checkerboard artifacts | Classifier too small to adapt to mammography |
| Where | `models_improved.py`, `--improved` | `--unfreeze-blocks 2 --head-bn` |

They are independent — any GAN can be paired with any classifier setting, which is why the results
grid became large (3 generators × 2 classifier settings × AD/SA).

---

## 6. Stage 2a — did the improved GAN train better?

| | D_loss variability | G_loss variability |
|---|---|---|
| Baseline GAN | 0.351 | 0.985 |
| **Improved GAN** | **0.031** | **0.025** |

- ✅ **Real and undisputed.** 10–30× more stable training.
- ⚠️ But "trains stably" is a different question from "makes better images" and from "helps the
  classifier." Those are tested separately in §8 — and they do not follow automatically.

---

## 7. The problem the audit found — and what it retracted

Every early result ran **without fixed random seeds**. Once seeds were added, results changed.

### 7.1 The measurement noise floor (nobody had measured this)

| Noise source | Magnitude |
|---|---|
| Classifier random seed | up to **1.6 points** |
| **Which 600 noise vectors formed the pool** | up to **5.3 points** |

**No effect anyone was chasing exceeded 5 points.** Results were substantially measuring randomness.

### 7.2 Retraction 1 — `FINDINGS.md` §5

| | baseline SA | improved SA | gap |
|---|---|---|---|
| Unseeded (the claim) | 60.05 | 66.93 | **+6.88** |
| Seed 0 | 65.34 | 65.61 | +0.27 |
| Seed 1 | 65.34 | 64.02 | −1.32 |
| **Mean** | **65.34** | **64.81** | **−0.52** |

❌ **Retracted.** One unlucky draw vs one lucky draw. The document blamed a "bad generator
checkpoint caught mid-swing" — but the seeded runs load *that exact same checkpoint file* and score
5.3 points higher. The checkpoint was never the variable.

### 7.3 Retraction 2 — `STAGE2_FINDINGS.md` §2

Claimed: with the unfrozen classifier, augmentation gives a real, reproducible +1.33-point win.
Re-tested across **3 independent pool draws**:

| Pool draw | baseline SA | improved300 SA |
|---|---|---|
| 1 | 70.63 | 67.46 |
| 2 | 67.46 | 67.20 |
| 3 | 68.78 | 64.02 |
| **Mean** | **68.96** | **66.23** |
| AD (real only) | **69.05** | — |

❌ **Retracted.** SA (68.96) vs AD (69.05) = **−0.09**. No lift. The 70.63 was the lucky draw.

Also: baseline − improved = 2.73 pts, but the *within-arm* spread is 3.2–3.4 pts.
**The noise exceeds the effect** (p ≈ 0.13). Not established.

---

## 8. Final verified results — DDSM

### 8.1 Classification (unfrozen classifier, best current estimates)

| Training data | Accuracy |
|---|---|
| **Real only (AD)** | **69.44** |
| Real + baseline-GAN fakes | 68.96 |
| Real + improved-GAN ep600 fakes | 67.60 |
| Real + improved-GAN ep300 fakes | 66.23 |

➡️ **Adding synthetic images never helps. The best case is a tie.**

### 8.2 Image quality across the whole training curve (KID, lower = better)

Every saved checkpoint of both architectures, scored against the same real reference (`1fzjlyh6`):

| epoch | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** | 0.370 | 0.377 | 0.215 | 0.184 | **0.165** ⭐ | 0.317 💥 | — | — | — | — | — | — |
| **improved** | 0.412 | 0.325 | 0.273 | 0.407 | 0.390 | 0.341 | 0.385 | 0.285 | 0.276 | 0.218 | 0.235 | 0.201 |

- ❌ **At matched epochs the baseline wins 5 of 6** (by 10σ at ep200 and ep250).
- ❌ **The best generator in the project is the paper-faithful baseline at ep250** (KID 0.165),
  beating the improved architecture's best (ep600, 0.201) with 350 fewer epochs.
- An earlier single-checkpoint comparison (baseline ep300 vs improved ep600) reported the improved
  architecture winning by −36%. That was the baseline's **worst** checkpoint against the improved
  architecture's **best** — retracted.
- 💡 **Loss stability ≠ output stability.** The improved architecture's losses are 10–30× smoother,
  yet its KID swings just as violently (+49% between adjacent checkpoints; the baseline's +92%).
  A smooth loss curve carried no information about generator quality.

### 8.3 The punchline — quality and usefulness rank differently

The three pools with both a quality score and a classification result:

| Pool | Image quality (KID) | Classifier accuracy |
|---|---|---|
| improved ep600 | 🥇 **best** (0.201) | 🥈 middle (67.60) |
| baseline ep300 | 🥈 middle (0.317) | 🥇 **best** (68.96) |
| improved ep300 | 🥉 worst (0.341) | 🥉 worst (66.23) |

**The best generator is not the best augmenter.** This decoupling is the one finding that has never
reversed across any round of re-testing — and §8.2 strengthens it: the project's best generator
(baseline ep250) was never even tested for classification.

### 8.4 Supporting checks

| Test | Result |
|---|---|
| Unfreezing the classifier (AD only) | 66.0% → **69.4%** ✅ real, reproduces across seeds |
| Can a classifier tell fake from real? | **100%** held-out accuracy for *every* pool — all fakes trivially detectable |
| Train on fakes alone, test on real | 60.0 / 59.8 / 65.1% — barely above the 61% always-guess-benign floor |

---

## 9. What is actually true

### ✅ Holds up

- The paper's +10-point augmentation claim **does not reproduce**, on either dataset.
- **Unfreezing the classifier genuinely helps** (+3.4 pts), reproducing across seeds.
- The improved GAN **trains far more stably** (10–30× lower *loss* variance).
- **Loss stability does not imply output quality** — the improved architecture's KID is as volatile
  as the baseline's, and the baseline produced the project's best generator.
- **Better image quality does not produce better classification** — demonstrated twice, on two
  datasets, with two different metrics (FID on COVID-CXR, KID on DDSM).
- **Uncontrolled sampling dominated every comparison** — ~5 accuracy points from the pool draw,
  49–92% KID swings between adjacent checkpoints.

### ❌ Retracted

- `FINDINGS.md` §5's +6.9-point matched-epoch win
- `STAGE2_FINDINGS.md` §2's +1.33-point augmentation win
- The improved architecture's **−36% image-quality win** — worst-vs-best checkpoint selection
- The `ANALYSIS.md` §4.2 "manifold overlap" mechanism — failed three separate tests

### Stage 2a scorecard

| Claim | Status |
|---|---|
| Improved architecture trains more stably (loss variance) | ✅ True |
| Improved architecture produces better images | ❌ **False** — baseline wins at matched epochs |
| Improved architecture helps classification | ❌ False |

### 🤷 Unresolved

- Improved-GAN augmentation looks ~2.7 pts worse than baseline-GAN augmentation, consistent in
  direction across all 3 paired draws, but **inside the noise** (p ≈ 0.13). No mechanism identified.

---

## 10. The one-sentence version

> GAN augmentation does not work here, and the Stage 2 architecture change stabilised the loss curve
> without improving image quality or accuracy — image quality and classifier usefulness turn out to
> be unrelated, and every other reported gain, including our own, was a single sample from a
> high-variance process that nobody had measured.

---

## 11. Outstanding work

| Document | Section | Action |
|---|---|---|
| `FINDINGS.md` | §5 | Retract; replace with §7.2 above |
| `FINDINGS.md` | §4.5 | Downgrade — shares the uncontrolled z-draw |
| `FINDINGS.md` | §6 | Remove "strongest, most controlled result" framing |
| `STAGE2_FINDINGS.md` | §2 | Retract finding #2 per §7.3 |
| `STAGE2_FINDINGS.md` | §3 | Note that the interaction hypothesis is now unsupported |
| `REPORT.md` | §5.2, §6 | Rewrite — currently leans on the retracted DDSM numbers |
| `ANALYSIS.md` | §4.2 | Retire the mechanism |

Recommended framing for the rewrite: Stage 2a improved training stability and, at 600 epochs,
generation quality — and **neither translated into classification gains**. That decoupling, shown on
two datasets, is the project's actual contribution.
