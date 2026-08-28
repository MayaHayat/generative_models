"""
Report figures for the CBIS-DDSM section.

    python make_figures.py --out-dir figures

Produces two PDFs (vector, for LaTeX) plus PNG previews:

  fig_ddsm_augmentation  -- the DDSM counterpart to the COVID report's Figure 3:
                            CNN-AD vs CNN-SA under both classifier capacities.
  fig_ddsm_kid_curve     -- KID across training for both generator architectures,
                            which Table 11 currently carries as eight columns of
                            numbers.

Every value below is transcribed from Weights & Biases with its run ID in the
comment, so each bar traces back to the run that produced it. Update NUMBERS
after new runs land; nothing else needs touching.

Palette: categorical slots 1-3 of the project's reference palette, validated for
colour-vision deficiency (worst adjacent pair dE 9.2 deutan / 27.6 normal). The
aqua slot falls below 3:1 contrast against the surface, so every bar carries a
visible value label -- identity is never colour-alone.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- palette ---
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e3e2dd"

MAJORITY_FLOOR = 61.11  # always-predict-benign on the 378-image test set

# ---------------------------------------------------------------- numbers ---
# accuracy (%), mean of three runs; run IDs are the W&B runs averaged
NUMBERS = {
    "frozen": {
        "AD (real only)":     66.05,  # nmg0hffv, d9c1262z, 1wnr289v
        "SA (baseline GAN)":  63.93,  # hi1nckfl, xkwhrvj7, kx1x41u5
        "SA (improved GAN)":  62.87,  # ytpur0z5, x5dq1v8h, zbbxp5ea
    },
    "unfrozen (uf=2)": {
        "AD (real only)":     69.93,  # bdb3jrf7, vne7xvub, zkt37znw
        "SA (baseline GAN)":  68.61,  # 2gzdjtvj, fx8ijyvc, mirxvw6q
        "SA (improved GAN)":  67.55,  # 15aab19e, wz3t79no, yy0zfud8
    },
}

# KID per checkpoint, run dasjho3b (36 pools, n=600 each)
KID_EPOCHS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]
KID_BASELINE = [.369, .376, .216, .180, .164, .315, .249, .170, .172, .158, .163, .134, .121, .419, .437, .435, .393, .404]
KID_IMPROVED = [.414, .327, .273, .403, .390, .339, .385, .283, .276, .220, .236, .203, .259, .206, .194, .237, .308, .264]


def _style(ax):
    """Recessive chrome: solid hairline grid on the value axis only, no box."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    ax.set_axisbelow(True)


def fig_augmentation(out_dir: Path):
    groups = list(NUMBERS)
    series = list(NUMBERS[groups[0]])
    colors = [BLUE, ORANGE, AQUA]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width, gap = 0.24, 0.02
    for i, (name, color) in enumerate(zip(series, colors)):
        xs = [g + (i - 1) * (width + gap) for g in range(len(groups))]
        vals = [NUMBERS[g][name] for g in groups]
        ax.bar(xs, vals, width, label=name, color=color, linewidth=0)
        for x, v in zip(xs, vals):                       # visible label on every bar
            ax.text(x, v + 0.18, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, color=INK)

    ax.axhline(MAJORITY_FLOOR, color=INK_MUTED, linewidth=1.2, zorder=1)
    ax.text(1.46, MAJORITY_FLOOR,
            "  majority-class\n  floor {:.2f}".format(MAJORITY_FLOOR),
            ha="left", va="center", fontsize=8.5, color=INK_MUTED)

    ax.set_xlim(-0.45, 1.92)                     # room at the right for the floor label
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g.capitalize() for g in groups], fontsize=10.5, color=INK)
    ax.set_ylabel("test accuracy (%)", fontsize=10, color=INK_2)
    ax.set_ylim(MAJORITY_FLOOR - 1.6, 72.0)   # baseline = the no-information line
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    _style(ax)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_2,
              loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_ddsm_augmentation.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig_ddsm_augmentation.pdf / .png")


def fig_kid_curve(out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(KID_EPOCHS, KID_BASELINE, color=BLUE, linewidth=2,
            marker="o", markersize=5, label="baseline architecture")
    ax.plot(KID_EPOCHS, KID_IMPROVED, color=ORANGE, linewidth=2,
            marker="s", markersize=5, label="improved (spectral norm + kernel 4)")

    # selective direct labels: each architecture's best, and the divergence
    ax.annotate(f"best {min(KID_BASELINE):.3f}\n(ep 650)",
                xy=(650, 0.121), xytext=(560, 0.055),
                fontsize=8.5, color=INK_2, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.8))
    ax.annotate("diverges at ep 684\n(+245% in 50 epochs)",
                xy=(700, 0.419), xytext=(770, 0.47),
                fontsize=8.5, color=INK_2, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.8))
    ax.annotate(f"best {min(KID_IMPROVED):.3f}\n(ep 750)",
                xy=(750, 0.194), xytext=(660, 0.145),
                fontsize=8.5, color=INK_2, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.8))

    ax.set_xlabel("training epoch", fontsize=10, color=INK_2)
    ax.set_ylabel("KID  (lower is better)", fontsize=10, color=INK_2)
    ax.set_xlim(0, 950)
    ax.set_ylim(0, 0.52)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    _style(ax)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_2, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_ddsm_kid_curve.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig_ddsm_kid_curve.pdf / .png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_augmentation(out)
    fig_kid_curve(out)
    print(f"\nfigures in {out.resolve()}")
