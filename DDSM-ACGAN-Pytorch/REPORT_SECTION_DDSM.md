# Replacement text — report §6, CBIS-DDSM

*Drop-in replacement for §6 "Supportive Cross-Dataset Validation on CBIS-DDSM", plus the DDSM
sentences in the Abstract and §7. Built around the six-row summary table, with balanced accuracy added
as the column that makes the recall analysis defensible.*

*Full evidence and per-run detail: `DDSM-ACGAN-Pytorch/DDSM_RESULTS.md`.*

---

## Why the current §6 needs replacing

| Current claim | Status |
|---|---|
| "naive setup failed to provide a reliable lift (64.02% → 65.34%)" | Right conclusion, mismatched numbers — pairs an *unseeded* AD run with a *seeded* SA run |
| "unfreezing… unlocking a clean, reproducible win (+1.32% lift across seeds)" | **Retracted** — across three pool draws the lift is **−0.97** |
| "improved GAN yielded a massive +6.90% lift… baseline GAN SA fell to 60.05%" | **Retracted** — both were single unseeded draws; the seeded rerun gives **−0.52** |
| "reducing loss standard deviation by 10–30×" | Verified, keep |
| improved GAN "mathematically superior by every quality metric" | **False on DDSM** — the baseline generator reaches a better KID (0.121 vs 0.194) |
| "striking confirmation of the Reversal Paradox" | **Not supported** — the improved generator is worse on quality *and* utility, which is consistent, not paradoxical |
| 16-bit PNG loading fix | Correct, keep |

All four retractions have one cause: those comparisons were single draws from a process whose
run-to-run variance exceeds the effects being measured. §6.2 quantifies it.

---

# 6 Cross-Dataset Validation on CBIS-DDSM

## 6.1 Motivation and task

Is our failure to reproduce the paper's lift an artifact of the modern, cleaner COVID-CXR dataset, or
a property of the AC-GAN method itself? To find out we replicated the entire pipeline on **CBIS-DDSM**,
a static, versioned mammography benchmark, using mass-type ROI lesions only: **1,318 real training
images, 378 test images** (231 benign, 147 malignant). The always-predict-benign floor is **61.11%**.

DDSM is a useful contrast to COVID-CXR precisely because it is *hard*. Our COVID baseline sits near
ceiling at 94.48%, leaving little headroom for augmentation. DDSM leaves plenty.

## 6.2 What we did

We ran the same three-component pipeline (AC-GAN generator, discriminator, VGG16 classifier) with the
same hyperparameters, and applied both Stage 2 tracks: the generator improvements (spectral
normalization; transpose-conv kernel 5 → 4 at stride 2) and the classifier capacity change
(unfreeze top two VGG16 blocks, BatchNorm head, discriminative learning rates).

**Data fix.** DDSM ROIs are 16-bit grayscale PNGs (values to 65,535). Standard PIL conversion
truncates to 8 bits and collapses nearly every pixel to white. We implemented per-image min-max
scaling to [0, 255] before RGB conversion, restoring tissue texture. Every result below post-dates
this fix.

**A measurement problem we had to solve first.** Our initial DDSM experiments ran without fixed
random seeds. When we seeded them and re-ran, several results changed sign — which led us to quantify
three independent variance sources, none previously controlled:

| Source | Magnitude | Evidence |
|---|---|---|
| Classifier random seed | up to **1.85 pts** | AD unfrozen, seeds 0/1/2: 69.84 / 69.05 / 70.90 |
| **Which noise vectors form the synthetic pool** | up to **5.29 pts** | same checkpoint, different `z` draw |
| Regenerating a seeded pool on a different GPU | up to **2.65 pts** | identical seed and code, different session |

The pool draw dominates, and **no effect we were attempting to measure exceeded 5 points**. Any
single-run comparison on this dataset is therefore dominated by sampling noise — which is why four
claims from our earlier DDSM analysis are withdrawn above.

The third source is worth noting for reproducibility practice generally. Classifier training is fully
deterministic given a seed: our real-only runs reproduce to the digit across sessions five days apart.
But synthetic *generation* is not, because cuDNN selects convolution algorithms per device. **A seeded
synthetic pool is reproducible only on the machine that created it.**

Accordingly every number below averages **three independent pool draws** (classifier seed fixed), or
three classifier seeds where no pool is involved.

## 6.3 Main results

Synthetic pools drawn from each architecture's best checkpoint by KID (baseline epoch 650, improved
epoch 750), 300 malignant + 300 benign per pool.

| Configuration | Train acc | **Test acc** | Malignant recall | Benign recall | **Balanced acc** |
|---|---|---|---|---|---|
| SA improved — frozen | 0.820 | 62.70 | 0.610 | 0.637 | 0.624 |
| SA baseline — frozen | 0.848 | 63.80 | 0.556 | 0.691 | 0.623 |
| **AD (real only) — frozen** | 0.795 | **66.05** | 0.535 | 0.740 | **0.638** |
| SA improved — unfrozen (uf=2) | 0.997 | 67.55 | 0.712 | 0.652 | 0.682 |
| SA baseline — unfrozen (uf=2) | 0.997 | 68.21 | 0.706 | 0.667 | 0.686 |
| **AD (real only) — unfrozen (uf=2)** | 0.993 | **69.93** | 0.646 | 0.733 | **0.690** |

Balanced accuracy — the mean of the two class recalls — is included because raw accuracy is
misleading on an imbalanced test set, and because it is what separates the two findings below.

## 6.4 Finding 1: unfreezing the encoder produces genuine learning

| AD frozen → AD unfrozen | Change |
|---|---|
| Test accuracy | **+3.88** (66.05 → 69.93) |
| Malignant recall | **+0.111** (0.535 → 0.646) |
| Benign recall | −0.007 (0.740 → 0.733) |
| **Balanced accuracy** | **+0.052** (0.638 → 0.690) |

Unfreezing catches **16 more cancers out of 147** while losing essentially no benign cases. Both class
recalls and balanced accuracy improve together — the signature of a model that has genuinely learned
better features, not one that has moved its decision threshold. This confirms the **Capacity Knob** of
§7.3 directly and independently on a second dataset: ImageNet features must be allowed to adapt to
medical texture.

## 6.5 Finding 2: augmentation shifts the operating point but does not improve discrimination

Every augmented configuration scores *below* the real-only baseline of the same capacity:

| Configuration | Δ accuracy | Δ malignant recall | Δ benign recall | Δ balanced acc |
|---|---|---|---|---|
| frozen, baseline GAN | −2.25 | +0.020 | −0.049 | −0.015 |
| frozen, improved GAN | −3.35 | +0.075 | −0.103 | −0.014 |
| unfrozen, baseline GAN | −1.72 | +0.060 | −0.067 | −0.003 |
| unfrozen, improved GAN | −2.38 | +0.066 | −0.081 | −0.008 |

Malignant recall does rise consistently — by up to +0.075 — and on a cancer-detection task that is
the recall that matters clinically. It is tempting to read this as the model learning malignant
features better. **The balanced-accuracy column shows it is not.** In every configuration the gain in
malignant recall is offset almost exactly by a loss in benign recall, leaving balanced accuracy flat
or slightly negative. In case counts, the unfrozen baseline-GAN configuration catches **9 more
cancers** but misclassifies **14 more benign** cases.

This is a decision-threshold shift, not improved discrimination, and it has a mundane cause: adding
300 synthetic malignant images to a 231:147 benign-majority training set moves the class prior. It
requires nothing to have been learned from the synthetic images. The contrast with §6.4 is the point —
unfreezing improved malignant recall *for free*, augmentation only traded for it.

We note this trade-off may still be operationally desirable: in screening, a missed malignancy costs
more than a false positive, so a more sensitive operating point can be preferred even at equal
balanced accuracy. But that is a statement about which point on the curve one wants, not evidence that
augmentation improved the model — and the same shift is obtainable for free by class weighting or
threshold adjustment, with no generator required.

## 6.6 Finding 3: the generator improvements trade quality ceiling for stability

We trained both architectures to 900 epochs and evaluated **every** saved checkpoint (18 per
architecture) with KID against the real training set.

**Stability improved unambiguously**, reproducing the COVID Track A result:

| Window | D_loss sd | G_loss sd |
|---|---|---|
| Baseline, full 300 epochs | 0.351 | 0.984 |
| Improved, epochs 201–300 | **0.031** | **0.025** |
| Improved, epochs 351–600 | **0.026** | **0.018** |

**Generation quality, however, did not.** KID across training (lower is better):

| epoch | 250 | 400 | 500 | 600 | 650 | 700 | 750 | 900 |
|---|---|---|---|---|---|---|---|---|
| Baseline | 0.164 | 0.170 | 0.158 | 0.134 | **0.121** | 0.419 | 0.437 | 0.404 |
| Improved | 0.390 | 0.283 | 0.220 | 0.203 | 0.259 | 0.206 | **0.194** | 0.264 |

Two opposing facts follow. **The baseline architecture reaches a higher ceiling** — its best
checkpoint (0.121) beats the improved architecture's best (0.194) with 350 fewer epochs, and it wins
12 of 18 matched-epoch comparisons. **But it is not safe to train unattended:** at epoch 684 it
diverged catastrophically, KID rising 245% in 50 epochs and never recovering. The loss trace shows
the mechanism — D_loss fell 1.17 → 0.38 while G_loss rose 2.32 → 7.81, the discriminator overpowering
the generator until its gradients died. The improved architecture held D ≈ 1.60 / G ≈ 0.90 flat from
epoch 200 to 900 and never diverged.

This is exactly the mechanism spectral normalization implements. Bounding the discriminator's
Lipschitz constant *weakens* D by construction: a weaker discriminator teaches the generator less,
costing peak quality, but can never overpower it. **Lower ceiling, no floor.** Against a
pre-registered criterion (median KID over the final six checkpoints) the improved architecture wins,
0.248 to 0.411; on best-ever checkpoint the baseline wins, 0.121 to 0.194. Both are true, and they
answer different questions: monitor quality and stop early, take the baseline; train to a fixed budget
and take what you get, take the improved architecture.

Methodologically, adjacent checkpoints 50 epochs apart differ by up to 245% in KID against error bars
of ±0.015 — so comparing one checkpoint per architecture, as we initially did, measures where training
happened to be sampled rather than the architecture.

## 6.7 Finding 4: generation quality is decoupled from augmentation value

DDSM permits the cleanest possible test of the decoupling claim, with **no architecture confound** —
the same generator at two training lengths:

| Baseline generator | KID | Unfrozen CNN-SA accuracy |
|---|---|---|
| epoch 300 | 0.315 | 68.96 |
| epoch 650 | **0.121** | 68.61 |

**A 2.6× improvement in generation quality changed downstream accuracy by −0.35 points — that is, not
at all.** This is the DDSM counterpart to the COVID Stage 1 result where FID halved with no downstream
movement, and it is stronger evidence because only one variable changed.

Consistently, no DDSM generator produced transferable pathology. Training solely on a synthetic pool
and testing on real data never cleared the 61.11% floor meaningfully (60.05%, 59.79%, 65.08% for the
baseline, improved-300 and improved-600 pools), and a real-versus-synthetic probe separated **every**
pool from real data with 100% held-out accuracy.

We are careful not to overclaim here. On DDSM the rankings of quality and utility happen to *agree*
across architectures — the improved generator is worse on KID and worse downstream. We therefore **do
not** claim DDSM reproduces the COVID Reversal Paradox, in which a strictly superior generator
degraded the classifier; on DDSM no generator was strictly superior. What DDSM establishes is the
weaker but still decisive claim: **generator quality metrics carry no information about augmentation
value in either direction.**

## 6.8 Cross-dataset summary

| Finding | COVID-CXR | CBIS-DDSM |
|---|---|---|
| Naive paper setup fails to lift accuracy | ✅ | ✅ |
| Unfreezing the encoder raises the baseline | ✅ | ✅ (+3.88, genuine — balanced acc +0.052) |
| Synthetic pool carries no transferable pathology (naive GAN) | ✅ | ✅ |
| **Unfreezing rescues the augmentation lift** | ✅ (+2.19) | ❌ **(−1.72)** |
| Generator quality decoupled from augmentation value | ✅ | ✅ (strongest evidence) |
| Improved architecture stabilises GAN training | ✅ | ✅ (10–30× lower loss variance; no divergence) |
| Improved architecture raises generation quality | ✅ (FID 273 → 121) | ❌ (baseline ceiling is better) |

DDSM replicates four findings, refines one, and contradicts two. The contradictions are the
informative part: augmentation's success on COVID is not a general property of the method, and the
generator changes that worked on chest X-rays bought stability rather than quality on mammography.

---

# Corrections needed elsewhere

**Abstract**, final paragraph — replace the DDSM sentences with:

> The results reproduced our core diagnostic findings: the naive paper setup failed to improve
> accuracy, unfreezing the encoder was again the only intervention that reliably improved the model
> (+3.88%, with malignant recall rising 0.111 at no cost to benign recall), and generator quality was
> again decoupled from downstream value, with a 2.6× KID improvement producing no accuracy change.
> Critically, DDSM also **bounds** our central claim: augmentation did not help under *any*
> configuration tested, including one built from the best generator we produced. Where augmentation
> did shift malignant recall upward, balanced accuracy stayed flat — a decision-threshold shift rather
> than improved discrimination. Unfreezing is therefore necessary but not sufficient. Establishing
> this also required quantifying a measurement problem that invalidated several of our own earlier
> DDSM comparisons: uncontrolled synthetic-pool sampling contributes up to 5 accuracy points,
> exceeding every effect we sought to measure.

**§7.1 What Worked** — scope "capacity expansion rescued the augmentation lift" to COVID. On DDSM
capacity raised the baseline but produced no lift.

**§7.2 What Did Not Work** — add: *On CBIS-DDSM, augmentation failed to produce a lift under every
configuration tested, including an unfrozen classifier and our best-quality generator. Where malignant
recall improved, balanced accuracy did not — the gain was a threshold shift, obtainable by class
weighting alone. Ample headroom was not sufficient to rescue augmentation.*

**§7.3 The Dual-Knob Framework** — the framework survives and gains a boundary condition:

> *Headroom is necessary but not sufficient. DDSM had abundant headroom (69.93% against a 61.11%
> floor) and augmentation still failed, because the generator never produced class-discriminative
> samples: trained alone, its pools could not beat the majority-class floor. A third precondition —
> generator transfer quality above the majority floor — must hold before capacity and headroom can
> matter. This also supplies a cheap go/no-go test: run the transfer probe first; if it does not clear
> the floor, no downstream configuration will rescue the augmentation.*

Add to the decoupling paragraph: *Confirmed on DDSM under the cleanest available control — the same
generator architecture at two training lengths, where a 2.6× KID improvement produced a −0.35 point
accuracy change.*
