# plot_confidence_histogram.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src import configs
from plot_style import set_confmat_style


def _get_label_cols(df: pd.DataFrame):
    """Ermittelt Spaltennamen für True/Pred-Label."""
    if "true_label" in df.columns and "pred_label" in df.columns:
        return "true_label", "pred_label"
    if "true" in df.columns and "pred" in df.columns:
        return "true", "pred"
    return None, None


def plot_confidence_hist(pred_path: Path):
    # Modellname aus dem Ordnernamen ableiten (…/metrics/resnet50/predictions.csv)
    model_name = pred_path.parent.name

    df = pd.read_csv(pred_path)

    true_col, pred_col = _get_label_cols(df)
    if true_col is None:
        print(f"[WARN] Keine passenden Label-Spalten in {pred_path}, skip.")
        return

    # Prüfen, ob es überhaupt prob_* Spalten gibt
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        print(f"[WARN] Keine prob_* Spalten in {pred_path}, skip.")
        return

    true_probs = []
    is_correct = []

    for _, row in df.iterrows():
        true_label = row[true_col]
        col_name = f"prob_{true_label}"
        if col_name not in df.columns:
            continue
        true_probs.append(row[col_name])
        is_correct.append(row[pred_col] == true_label)

    if not true_probs:
        print(f"[WARN] Keine gültigen Wahrscheinlichkeiten in {pred_path}")
        return

    true_probs = pd.Series(true_probs, dtype=float)
    is_correct = pd.Series(is_correct, dtype=bool)

    probs_correct = true_probs[is_correct]
    probs_wrong   = true_probs[~is_correct]

    # einfache Accuracy für den Titel
    acc = is_correct.mean() if len(is_correct) > 0 else float("nan")

    set_confmat_style()  # einheitlicher Style
    fig, ax = plt.subplots(figsize=(7, 4))

    bins = np.linspace(0.0, 1.0, 21)

    ax.hist(probs_correct, bins=bins, alpha=0.6, color="grey", label=f"korrekt (n={probs_correct.size})")
    ax.hist(probs_wrong,   bins=bins, alpha=0.6, color="red",  label=f"falsch (n={probs_wrong.size})")

    ax.set_xlabel("Modell-Konfidenz für wahre Klasse")
    ax.set_ylabel("Anzahl Beispiele")
    ax.set_title(f"Verteilung der Konfidenz – {model_name} (Accuracy = {acc:.3f})")
    ax.legend()

    # Beispiel-Threshold (z.B. 0.5)
    ax.axvline(0.5, color="blue", linestyle="--", linewidth=1)
    ymax = ax.get_ylim()[1]
    ax.text(0.5, ymax * 0.95, "Threshold 0.5", ha="center", va="top",
            color="blue", fontsize=9)

    fig.tight_layout()

    out_dir = configs.THESIS_DIR / "confidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{model_name}_confidence_hist.png"
    pdf_path = out_dir / f"{model_name}_confidence_hist.pdf"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"[OK] Speicherung: {png_path}")


def main():
    metrics_dir = configs.METRICS_DIR

    # Wir erwarten: outputs/metrics/<model>/predictions.csv
    model_dirs = [d for d in metrics_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        print("[ERR] Keine Modellordner in", metrics_dir)
        return

    any_found = False
    for mdir in sorted(model_dirs):
        pred_path = mdir / "predictions_with_probs.csv"
        if not pred_path.exists():
            pred_path = mdir / "predictions.csv"
        if not pred_path.exists():
            continue

        any_found = True
        plot_confidence_hist(pred_path)

    if not any_found:
        print("[ERR] Keine predictions.csv in den Modellordnern gefunden.")


if __name__ == "__main__":
    main()
