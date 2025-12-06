# analyze_dataset_split.py
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import configs
from plot_style import set_confmat_style, apply_axes_style


def _count_images_in_dir(root: Path, cls: str) -> int:
    """
    Zählt .png und .jpg in root/<cls>.
    """
    cls_dir = root / cls
    if not cls_dir.exists():
        print(f"[WARN] Ordner fehlt: {cls_dir}")
        return 0

    n = sum(1 for _ in cls_dir.glob("*.png")) + sum(1 for _ in cls_dir.glob("*.jpg"))
    return n


def collect_split_counts() -> pd.DataFrame:
    """
    Zählt Bilder pro Klasse und Split (train/val/test).

    Rückgabe-DataFrame:
        index  = Klassen
        columns = ["train", "val", "test", "total"]
    """
    classes = list(configs.CLASSES)

    train_root = configs.DATA_TRAIN
    val_root = getattr(configs, "DATA_VAL", Path("data/ecg_val"))
    test_root = getattr(configs, "DATA_TEST", Path("data/ecg_test"))

    rows = []
    for cls in classes:
        n_train = _count_images_in_dir(train_root, cls)
        n_val = _count_images_in_dir(val_root, cls)
        n_test = _count_images_in_dir(test_root, cls)
        total = n_train + n_val + n_test

        rows.append(
            {
                "class": cls,
                "train": n_train,
                "val": n_val,
                "test": n_test,
                "total": total,
            }
        )

    df = pd.DataFrame(rows).set_index("class")
    return df


def plot_class_split_bars(df: pd.DataFrame) -> None:
    """
    Balkendiagramm: pro Klasse Train/Val/Test-Anzahl.
    """
    set_confmat_style()

    out_dir = configs.THESIS_DIR / "dataset_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = list(df.index)
    x = np.arange(len(classes), dtype=float)
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x - width, df["train"], width=width, label="Train")
    ax.bar(x, df["val"], width=width, label="Val")
    ax.bar(x + width, df["test"], width=width, label="Test")

    apply_axes_style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")

    ax.set_ylabel("Anzahl Bilder")
    ax.set_xlabel("Klasse")
    ax.set_title("Verteilung der ECG-Klassen nach Split")

    ax.legend(title="Split")

    fig.tight_layout()
    png_path = out_dir / "class_split_bars.png"
    pdf_path = out_dir / "class_split_bars.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {png_path}")


def plot_split_donut(df: pd.DataFrame) -> None:
    """
    Donut-Chart: globaler Anteil Train / Val / Test (über alle Klassen).
    """
    set_confmat_style()

    out_dir = configs.THESIS_DIR / "dataset_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    totals = {
        "Train": int(df["train"].sum()),
        "Val": int(df["val"].sum()),
        "Test": int(df["test"].sum()),
    }

    labels = list(totals.keys())
    values = list(totals.values())
    total_n = sum(values)
    if total_n == 0:
        print("[ERR] Keine Bilder gefunden – Donut-Plot wird übersprungen.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    wedges, _, autotexts = ax.pie(
        values,
        labels=None,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        pctdistance=0.8,
    )

    # Loch in der Mitte → Donut
    centre_circle = plt.Circle((0, 0), 0.55, fc="white")
    fig.gca().add_artist(centre_circle)

    ax.set_title("Dataset Split: Train / Val / Test")

    # Legende mit absoluten Zahlen
    legend_labels = [
        f"{lbl} (n={val})" for lbl, val in zip(labels, values)
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Splits",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    fig.tight_layout()
    png_path = out_dir / "split_donut.png"
    pdf_path = out_dir / "split_donut.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {png_path}")


def main() -> None:
    df = collect_split_counts()
    print("\n[INFO] Bilder pro Klasse und Split:")
    print(df)

    plot_class_split_bars(df)
    plot_split_donut(df)


if __name__ == "__main__":
    main()
