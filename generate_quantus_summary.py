# starten mit python generate_quantus_summary.py
# erstellt eine zusammenfassende CSV-Tabelle und ein PNG-Bild mit den Quantus-Metriken aller Methoden

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import math


def save_table_png(df: pd.DataFrame, out_path: Path):
    display_rename = {
        "model": "Model",
        "method": "Method",
        "faithfulness_corr": "Faithfulness corr.",
        "max_sens": "Max sensitivity",
        "mprt": "MPRT",
    }
    df_display = df.rename(
        columns={k: v for k, v in display_rename.items() if k in df.columns}
    )

    df_rounded = df_display.copy()
    for col in df_rounded.select_dtypes(include="number").columns:
        df_rounded[col] = df_rounded[col].round(3)

    n_rows, n_cols = df_rounded.shape

    row_height = 0.4
    fig_height = max(3, n_rows * row_height)
    fig_width = max(6, n_cols * 1.0)

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
    in_dir = Path("outputs/metrics/quantus_raw")

    metrics_out_dir = Path("outputs/metrics")
    metrics_out_dir.mkdir(parents=True, exist_ok=True)
    out_csv_metrics = metrics_out_dir / "quantus_summary.csv"

    thesis_quantus_dir = Path("outputs/thesis_figures/quantus")
    thesis_quantus_dir.mkdir(parents=True, exist_ok=True)
    out_csv_thesis = thesis_quantus_dir / "quantus_summary.csv"
    out_png_thesis = thesis_quantus_dir / "quantus_summary_table.png"

    csvs = sorted(in_dir.glob("*.csv"))
    if not csvs:
        print("[ERROR] Keine quantus-*.csv gefunden in", in_dir)
        return

    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)

    df.to_csv(out_csv_metrics, index=False)
    df.to_csv(out_csv_thesis, index=False)

    print("[OK] Summary CSV gespeichert unter:")
    print("   -", out_csv_metrics)
    print("   -", out_csv_thesis)

    save_table_png(df, out_png_thesis)
    print("[OK] Tabellen-PNG für Thesis gespeichert unter:")
    print("   -", out_png_thesis)


if __name__ == "__main__":
    main()
