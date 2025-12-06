# generate_quantus_summary.py
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import math


def save_table_png(df: pd.DataFrame, out_path: Path):
    """
    Speichert das DataFrame als gut lesbare Tabelle als PNG.
    Höhe der Figur wird dynamisch an die Anzahl der Zeilen angepasst.
    """
    # etwas runden, damit die Tabelle cleaner aussieht
    df_rounded = df.copy()
    for col in df_rounded.select_dtypes(include="number").columns:
        df_rounded[col] = df_rounded[col].round(3)

    n_rows, n_cols = df_rounded.shape

    # Dynamische Fig-Größe: Breite fix, Höhe pro Zeile
    row_height = 0.4  # probier ggf. 0.45 oder 0.5, wenn es dir zu eng ist
    fig_height = max(3, n_rows * row_height)  # mindestens 3 inch
    fig_width = max(6, n_cols * 1.0)          # Breite abhängig von Spalten

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df_rounded.values,
        colLabels=df_rounded.columns,
        loc="center",
        cellLoc="center"
    )

    # Schriftgröße & Zeilenabstand anpassen
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)  # x-Skalierung, y-Skalierung

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    in_dir = Path("outputs/metrics/quantus_raw")

    # 1) CSV-Gesamttabelle erstellen
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

    # CSV an zwei Stellen speichern
    df.to_csv(out_csv_metrics, index=False)
    df.to_csv(out_csv_thesis, index=False)

    print("[OK] Summary CSV gespeichert unter:")
    print("   -", out_csv_metrics)
    print("   -", out_csv_thesis)

    # 2) Schönes PNG für die Thesis
    save_table_png(df, out_png_thesis)
    print("[OK] Tabellen-PNG für Thesis gespeichert unter:")
    print("   -", out_png_thesis)


if __name__ == "__main__":
    main()
