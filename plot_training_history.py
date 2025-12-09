# starten mit: python plot_training_history.py
# erstellt Trainingsverlauf-Plots für alle Modelle und eine Vergleichsübersicht der Validation Loss
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from src import configs
from plot_style import set_confmat_style, apply_axes_style


def _check_required_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Benötigte Spalten {missing} fehlen in {name}. "
            f"Verfügbare Spalten: {list(df.columns)}"
        )


def _history_path(model_name: str) -> Path:
    return configs.METRICS_DIR / model_name / "history.csv"


def _get_out_dir() -> Path:
    out_dir = configs.THESIS_DIR / "training_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_history(model_name: str) -> bool:
    hist_path = _history_path(model_name)

    if not hist_path.exists():
        print(f"[WARN] History nicht gefunden: {hist_path}")
        return False

    df = pd.read_csv(hist_path)

    _check_required_columns(df, ["epoch", "train_loss", "val_loss"], hist_path.name)

    has_acc = "train_acc" in df.columns and "val_acc" in df.columns

    set_confmat_style()

    if has_acc:
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

        ax_loss.plot(df["epoch"], df["train_loss"], marker="o", label="Training loss")
        ax_loss.plot(df["epoch"], df["val_loss"], marker="o", label="Validation loss")
        apply_axes_style(ax_loss)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()

        ax_acc.plot(df["epoch"], df["train_acc"], marker="o", label="Training accuracy")
        ax_acc.plot(df["epoch"], df["val_acc"], marker="o", label="Validation accuracy")
        apply_axes_style(ax_acc)
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()

        fig.suptitle(f"Training history – {model_name}", y=1.02)
        fig.tight_layout()

    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["epoch"], df["train_loss"], marker="o", label="Training loss")
        ax.plot(df["epoch"], df["val_loss"], marker="o", label="Validation loss")
        apply_axes_style(ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training history – {model_name}")
        ax.legend()
        fig.tight_layout()

    out_dir = _get_out_dir()
    png_path = out_dir / f"{model_name}_training_curve.png"
    pdf_path = out_dir / f"{model_name}_training_curve.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")

    print(f"[OK] Saved PNG: {png_path}")
    print(f"[OK] Saved PDF: {pdf_path}")

    plt.close(fig)
    return True


def plot_val_loss_summary(models_with_history: List[str]) -> None:
    if not models_with_history:
        print(
            "[WARN] Keine Modelle mit vorhandener History – Summary wird übersprungen."
        )
        return

    set_confmat_style()

    fig, ax = plt.subplots(figsize=(7, 5))
    all_vals = []

    for model_name in models_with_history:
        hist_path = _history_path(model_name)
        df = pd.read_csv(hist_path)
        _check_required_columns(df, ["epoch", "val_loss"], hist_path.name)

        ax.plot(
            df["epoch"],
            df["val_loss"],
            marker="o",
            linewidth=1.8,
            label=model_name,
        )

        all_vals.extend(df["val_loss"].tolist())

    ax.set_yscale("log")
    positive_vals = [v for v in all_vals if v > 0]
    if positive_vals:
        ymin = min(positive_vals)
        ymax = max(positive_vals)
        ax.set_ylim(ymin * 0.8, ymax * 1.2)

    apply_axes_style(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss (log scale)")
    ax.set_title("Comparison of models – validation loss")

    ax.legend(title="Model", loc="best")
    fig.tight_layout()

    out_dir = _get_out_dir()
    png_path = out_dir / "all_models_val_loss.png"
    pdf_path = out_dir / "all_models_val_loss.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")

    print(f"[OK] Saved summary PNG (log): {png_path}")
    print(f"[OK] Saved summary PDF (log): {pdf_path}")

    plt.close(fig)


def main() -> None:
    models_with_history: List[str] = []

    for model_name in configs.MODEL_NAMES:
        ok = plot_history(model_name)
        if ok:
            models_with_history.append(model_name)
    if models_with_history:
        plot_val_loss_summary(models_with_history)


if __name__ == "__main__":
    main()
