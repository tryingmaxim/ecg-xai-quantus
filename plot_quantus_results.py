# plot_quantus_results.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import set_confmat_style, apply_axes_style, get_color
from src import configs


def _get_metrics_dir() -> Path:
    """Basis-Verzeichnis für Metriken (aus configs oder Fallback)."""
    return getattr(configs, "METRICS_DIR", Path("outputs/metrics"))


def _get_thesis_xai_dir() -> Path:
    """Verzeichnis für XAI-Figuren der Thesis (aus configs oder Fallback)."""
    thesis_base = getattr(configs, "THESIS_DIR", Path("outputs/thesis_figures"))
    xai_dir = Path(thesis_base) / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)
    return xai_dir


def _check_required_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    """Stellt sicher, dass alle benötigten Spalten vorhanden sind."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Benötigte Spalten {missing} fehlen in {name}. "
            f"Verfügbare Spalten: {list(df.columns)}"
        )


def plot_quantus_results() -> None:
    """
    Erstellt Balkenplots für die drei Quantus-Metriken:

    - faithfulness_corr: höhere Werte = bessere Faithfulness
    - max_sens        : kleinere Werte = robustere Erklärungen
    - mprt            : je nach Interpretation (im Text erläutern)

    Es werden nur Modelle geplottet, für die es für die jeweilige Metrik
    überhaupt Werte gibt (mindestens eine Methode mit non-NaN).
    """
    set_confmat_style()

    metrics_dir = _get_metrics_dir()
    quantus_path = metrics_dir / "quantus_summary.csv"

    if not quantus_path.exists():
        raise FileNotFoundError(
            f"{quantus_path} nicht gefunden. "
            f"Bitte zuerst generate_quantus_summary.py ausführen."
        )

    df = pd.read_csv(quantus_path)
    _check_required_columns(
        df,
        ["model", "method", "faithfulness_corr", "max_sens", "mprt"],
        "quantus_summary.csv",
    )

    if df.empty:
        raise ValueError("quantus_summary.csv enthält keine Daten.")

    # Definiere zu plottende Metriken: (Spaltenname, Y-Achsentext, Dateiname)
    plots: List[Tuple[str, str, str]] = [
        (
            "faithfulness_corr",
            "Faithfulness der Erklärungen\n(FaithfulnessCorrelation, höher = besser)",
            "faithfulness.png",
        ),
        (
            "max_sens",
            "MaxSensitivity\n(kleiner = robustere Erklärungen)",
            "robustness.png",
        ),
        (
            "mprt",
            "Randomisationstest (MPRT)",
            "mprt.png",
        ),
    ]

    out_dir = metrics_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    thesis_dir = _get_thesis_xai_dir()

    all_models = sorted(df["model"].unique())
    methods = sorted(df["method"].unique())

    for metric, y_label, fname in plots:
        if metric not in df.columns:
            print(f"[WARN] Metrik '{metric}' nicht in DataFrame – Plot wird übersprungen.")
            continue

        # 🔹 Nur Modelle behalten, bei denen es für diese Metrik überhaupt Werte gibt
        has_value = (
            df.groupby("model")[metric]
            .apply(lambda s: s.notna().any())
            .reindex(all_models)
        )
        models = [m for m, ok in has_value.items() if ok]
        if not models:
            print(f"[WARN] Keine Modelle mit gültigen Werten für '{metric}' – Plot wird übersprungen.")
            continue

        x = np.arange(len(models), dtype=float)
        width = 0.8 / max(len(methods), 1)

        fig, ax = plt.subplots(figsize=(8, 5))

        # Bestes Modell/Methoden-Kombi nur für Faithfulness highlighten
        best_row = None
        if metric == "faithfulness_corr":
            df_valid = df.dropna(subset=[metric])
            if not df_valid.empty:
                best_idx = df_valid[metric].idxmax()
                best_row = df_valid.loc[best_idx]

        for i, method in enumerate(methods):
            sub = df[df["method"] == method].set_index("model")
            sub = sub.reindex(models)

            offset = (i - (len(methods) - 1) / 2.0) * width
            heights = sub[metric].values

            ax.bar(
                x + offset,
                heights,
                width=width,
                label=method,
                color=get_color(method),
            )

            # ★ Highlight auf besten Balken (nur für Faithfulness)
            if best_row is not None and metric == "faithfulness_corr":
                for j, model in enumerate(models):
                    if (
                        model == best_row["model"]
                        and method == best_row["method"]
                        and not pd.isna(heights[j])
                    ):
                        y_val = float(heights[j])
                        x_pos = float(x[j] + offset)
                        # kleiner Offset relativ zur Max-Höhe, damit der Stern nicht abgeschnitten wird
                        max_height = np.nanmax(heights) if np.isfinite(np.nanmax(heights)) else 1.0
                        ax.text(
                            x_pos,
                            y_val + 0.03 * max_height,
                            "★",
                            ha="center",
                            va="bottom",
                            fontsize=14,
                            color="black",
                        )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")

        ax.set_ylabel(y_label)
        ax.set_title(f"Quantus Metrik: {metric}")

        apply_axes_style(ax)
        ax.axhline(0, color="black", linewidth=0.8)

        # 🔹 Legende außerhalb platzieren, damit sie keine Balken überdeckt
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.subplots_adjust(right=0.78)  # Platz rechts für Legende
            ax.legend(
                handles,
                labels,
                title="XAI-Methode",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
            )

        # Layout: genug Platz für lange Y-Labels, X-Ticks und Legende
        fig.tight_layout()

        png_path = out_dir / fname
        fig.savefig(png_path, dpi=300, bbox_inches="tight")

        pdf_path = thesis_dir / fname.replace(".png", ".pdf")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")

        plt.close(fig)
        print(f"[OK] Saved PNG: {png_path}")
        print(f"[OK] Saved PDF: {pdf_path}")


def main() -> None:
    plot_quantus_results()


if __name__ == "__main__":
    main()
