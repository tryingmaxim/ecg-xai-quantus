# plot_quantus_scatter.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
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


def plot_quantus_scatter_grid() -> None:
    """
    Erzeugt ein 2×2-Scatter-Grid für Quantus-Metriken:

    (a) Faithfulness vs. Robustheit (MaxSensitivity)
    (b) Faithfulness vs. Randomisation (MPRT)
    (c) Robustheit vs. Randomisation
    (d) Accuracy vs. Faithfulness

    Punkte sind farblich nach XAI-Methode kodiert.
    NaN-Werte (z.B. MPRT bei Modellen ohne Berechnung) werden pro Subplot entfernt.
    """
    set_confmat_style()

    metrics_dir = _get_metrics_dir()
    quantus_path = metrics_dir / "quantus_summary.csv"
    perf_path = metrics_dir / "model_performance.csv"

    if not quantus_path.exists():
        raise FileNotFoundError(
            f"{quantus_path} nicht gefunden. "
            f"Bitte zuerst generate_quantus_summary.py ausführen."
        )
    if not perf_path.exists():
        raise FileNotFoundError(
            f"{perf_path} nicht gefunden. "
            f"Bitte zuerst generate_results_table.py ausführen."
        )

    df_q = pd.read_csv(quantus_path)
    df_p = pd.read_csv(perf_path)

    _check_required_columns(
        df_q,
        ["model", "method", "faithfulness_corr", "max_sens", "mprt"],
        "quantus_summary.csv",
    )
    _check_required_columns(
        df_p,
        ["model", "accuracy"],
        "model_performance.csv",
    )

    df = df_q.merge(df_p, on="model", how="left")

    out_dir = metrics_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    thesis_dir = _get_thesis_xai_dir()

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.subplots_adjust(hspace=0.30, wspace=0.25)

    # Helper für die vier Subplots: (ax, x_col, y_col, x_label, y_label, title)
    configs_scatter: List[Tuple] = [
        (
            axes[0, 0],
            "faithfulness_corr",
            "max_sens",
            "FaithfulnessCorrelation",
            "MaxSensitivity\n(kleiner = robustere Erklärungen)",
            "(a) Faithfulness vs. Robustheit",
        ),
        (
            axes[0, 1],
            "faithfulness_corr",
            "mprt",
            "FaithfulnessCorrelation",
            "MPRT",
            "(b) Faithfulness vs. Randomisation",
        ),
        (
            axes[1, 0],
            "max_sens",
            "mprt",
            "MaxSensitivity\n(kleiner = robuster)",
            "MPRT",
            "(c) Robustheit vs. Randomisation",
        ),
        (
            axes[1, 1],
            "accuracy",
            "faithfulness_corr",
            "Genauigkeit (Accuracy)",
            "FaithfulnessCorrelation",
            "(d) Genauigkeit vs. Faithfulness",
        ),
    ]

    methods = sorted(df["method"].unique())

    for ax, x_col, y_col, x_label, y_label, title in configs_scatter:
        # Pro Methode plottet man nur Zeilen mit gültigen Werten für x und y
        for method in methods:
            sub = df[df["method"] == method]
            sub = sub.dropna(subset=[x_col, y_col])
            if sub.empty:
                continue

            ax.scatter(
                sub[x_col],
                sub[y_col],
                label=method,
                color=get_color(method),
                alpha=0.8,
                edgecolor="black",
                linewidth=0.4,
                s=40,
            )

        apply_axes_style(ax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        # Nulllinie sinnvoll bei MPRT/Faithfulness
        if y_col in ("faithfulness_corr", "mprt"):
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)

    # gemeinsame Legende (nur aus dem ersten Achsenobjekt)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(methods),
            bbox_to_anchor=(0.5, 0.02),
            title="XAI-Methode",
            frameon=False,
        )

    # Platz für Legende unten lassen
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = out_dir / "quantus_scatter_grid.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    pdf_path = thesis_dir / "quantus_scatter_grid.pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"[OK] Saved PNG: {out_path}")
    print(f"[OK] Saved PDF: {pdf_path}")


def main() -> None:
    plot_quantus_scatter_grid()


if __name__ == "__main__":
    main()
