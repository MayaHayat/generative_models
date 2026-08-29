"""Evaluation helpers matching Sec. IV of the paper: per-class precision /
recall / F1 / support, macro & weighted averages, specificity, and a
confusion-matrix plot -- reproducing Table 1, Fig. 6 and Fig. 7."""
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def classification_table(y_true: Sequence[int], y_pred: Sequence[int],
                          class_names: Sequence[str]) -> str:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                             average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                average="weighted", zero_division=0)
    accuracy = float((y_true == y_pred).mean())

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificities = []
    for i in labels:
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    lines = [f"{'class':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}{'specificity':>13}"]
    for i, name in enumerate(class_names):
        lines.append(
            f"{name:<12}{precision[i]:>10.2f}{recall[i]:>10.2f}{f1[i]:>10.2f}"
            f"{support[i]:>10d}{specificities[i]:>13.2f}"
        )
    lines.append(
        f"{'macro avg':<12}{macro[0]:>10.2f}{macro[1]:>10.2f}{macro[2]:>10.2f}{sum(support):>10d}"
    )
    lines.append(
        f"{'weighted avg':<12}{weighted[0]:>10.2f}{weighted[1]:>10.2f}{weighted[2]:>10.2f}{sum(support):>10d}"
    )
    total = int(cm.sum())
    correct = int(np.trace(cm))

    lines.append("")
    lines.append("confusion matrix (rows = true, cols = predicted):")
    lines.append(f"{'':<12}" + "".join(f"{name:>10}" for name in class_names))
    for i, name in enumerate(class_names):
        lines.append(f"{name:<12}" + "".join(f"{cm[i, j]:>10d}" for j in labels))

    lines.append("")
    lines.append(f"{'class':<12}{'TP':>8}{'FP':>8}{'FN':>8}{'TN':>8}")
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - cm[i, i])
        fn = int(cm[i, :].sum() - cm[i, i])
        tn = total - tp - fp - fn
        lines.append(f"{name:<12}{tp:>8}{fp:>8}{fn:>8}{tn:>8}")

    lines.append("")
    lines.append(f"accuracy: {accuracy:.2%}  (= {correct}/{total} correct)")
    return "\n".join(lines)


def plot_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int],
                           class_names: Sequence[str], title: str, out_path: str):
    import matplotlib.pyplot as plt

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels, class_names)
    ax.set_yticks(labels, class_names)
    ax.set_xlabel("predicted label")
    ax.set_ylabel("true label")
    ax.set_title(title)
    for i in labels:
        for j in labels:
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pca(features: np.ndarray, labels: np.ndarray, sources: Sequence[str],
             class_names: Sequence[str], out_path: str):
    """PCA scatter of penultimate-layer CNN features, colored by class and
    real/synthetic origin -- reproduces Fig. 5."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    coords = PCA(n_components=2).fit_transform(features)
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls_idx, cls_name in enumerate(class_names):
        for source in ("real", "synthetic"):
            mask = (labels == cls_idx) & (np.asarray(sources) == source)
            if not mask.any():
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.6,
                       label=f"{cls_name} ({source})")
    ax.legend(fontsize=8)
    ax.set_title("PCA of CNN penultimate-layer features")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
