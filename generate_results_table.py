# generate_results_table.py
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

METRICS_DIR = Path("outputs/metrics")
OUT_FILE = METRICS_DIR / "model_performance.csv"


def parse_classification_report(txt_path: Path):
    text = txt_path.read_text(encoding="utf-8")

    # Accuracy
    m = re.search(r"Accuracy:\s*([0-9.]+)", text)
    acc = float(m.group(1)) if m else None

    macro_f1 = None
    weighted_f1 = None

    # F1-Werte aus den Zeilen "macro avg" und "weighted avg"
    lines = text.strip().splitlines()
    for line in lines:
        if line.strip().startswith("macro avg"):
            parts = line.split()
            macro_f1 = float(parts[-2])
        if line.strip().startswith("weighted avg"):
            parts = line.split()
            weighted_f1 = float(parts[-2])

    return acc, macro_f1, weighted_f1


def load_history_summary(model_name: str):
    """
    Liest outputs/metrics/<model>_history.csv und gibt eine kleine
    Zusammenfassung zurück (finale Werte + bestes val_acc).
    Falls keine History existiert: alles None.
    """
    hist_path = METRICS_DIR / model_name / "history.csv"
    if not hist_path.exists():
        return {
            "final_train_loss": None,
            "final_val_loss": None,
            "final_train_acc": None,
            "final_val_acc": None,
            "best_val_acc": None,
            "epoch_best_val": None,
        }

    df = pd.read_csv(hist_path)

    last = df.iloc[-1]
    final_train_loss = float(last["train_loss"])
    final_val_loss   = float(last["val_loss"])
    final_train_acc  = float(last["train_acc"])
    final_val_acc    = float(last["val_acc"])

    best_idx = df["val_acc"].idxmax()
    best_row = df.loc[best_idx]
    best_val_acc = float(best_row["val_acc"])
    epoch_best_val = int(best_row["epoch"])

    return {
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "best_val_acc": best_val_acc,
        "epoch_best_val": epoch_best_val,
    }


def save_table_png(df: pd.DataFrame, out_path: Path):
    """
    Speichert das DataFrame als gut lesbare Tabelle als PNG.
    Höhe wird dynamisch an die Zeilenzahl angepasst.
    """
    df_rounded = df.copy()
    for col in df_rounded.select_dtypes(include="number").columns:
        df_rounded[col] = df_rounded[col].round(3)

    n_rows, n_cols = df_rounded.shape

    row_height = 0.4
    fig_height = max(3, n_rows * row_height)
    fig_width = max(7, n_cols * 1.1)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df_rounded.values,
        colLabels=df_rounded.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = []
    for model_dir in METRICS_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        rpt = model_dir / "classification_report.txt"
        if not rpt.exists():
            continue

        model_name = model_dir.name
        acc, macro_f1, weighted_f1 = parse_classification_report(rpt)
        hist_summary = load_history_summary(model_name)

        row = {
            "model": model_name,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }
        row.update(hist_summary)
        rows.append(row)

    if not rows:
        print("[ERROR] Keine classification_report.txt gefunden.")
        return

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)

    # CSV unter outputs/metrics
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print("[OK] Modell-Performance gespeichert unter:", OUT_FILE)
    print(df)

    # zusätzlich CSV + PNG für die Thesis
    thesis_dir = Path("outputs/thesis_figures/metrics")
    thesis_dir.mkdir(parents=True, exist_ok=True)

    thesis_csv = thesis_dir / "model_performance.csv"
    thesis_png = thesis_dir / "model_performance_table.png"

    df.to_csv(thesis_csv, index=False)
    save_table_png(df, thesis_png)

    print("[OK] Thesis-CSV:", thesis_csv)
    print("[OK] Tabellen-PNG:", thesis_png)


if __name__ == "__main__":
    main()
