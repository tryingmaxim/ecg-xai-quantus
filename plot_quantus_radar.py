# plot_quantus_radar.py
from __future__ import annotations

from math import pi
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import set_confmat_style, get_color
from src import configs


def _get_metrics_dir() -> Path:
    """Liefert das Basis-Metrics-Verzeichnis (aus configs oder Fallback)."""
    return getattr(configs, "METRICS_DIR", Path("outputs/metrics"))


def _get_thesis_xai_dir() -> Path:
    """Liefert das Thesis-XAI-Verzeichnis (aus configs oder Fallback)."""
    thesis_base = getattr(configs, "THESIS_DIR", Path("outputs/thesis_figures"))
    xai_dir = Path(thesis_base) / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)
    return xai_dir


def _safe_normalize(series: pd.Series) -> pd.Series:
    """
    Normiert eine numerische Series auf [0, 1].
    Falls alle Werte gleich sind oder nur NaNs vorliegen, wird 0.5 zurückgegeben.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        return pd.Series(0.5, index=s.index)

    s_min = s.min()
    s_max = s.max()
    denom = s_max - s_min

    if denom <= 0:
        # alle Werte gleich
        return pd.Series(0.5, index=s.index)

    return (s - s_min) / denom


def _check_required_columns(df: pd.DataFrame, cols: List[str], df_name: str) -> None:
    """Stellt sicher, dass alle benötigten Spalten vorhanden sind."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Benötigte Spalten {missing} fehlen in {df_name}. "
            f"Verfügbare Spalten: {list(df.columns)}"
        )


def plot_quantus_radar() -> None:
    """
    Erzeugt pro Modell einen Radar-Plot mit drei Quantus-Metriken:

    - Faithfulness (normierte faithfulness_corr, höher = besser)
    - Robustheit (1 − normierte max_sens, höher = robustere Erklärungen)
    - Randomisation (normierte mprt, Richtung abhängig von Interpretation)

    Pro Modell werden die verschiedenen XAI-Methoden als Polygone eingezeichnet.
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

    # Normierung über alle Modelle & Methoden hinweg
    faith_norm = _safe_normalize(df["faithfulness_corr"])
    maxsens_norm = _safe_normalize(df["max_sens"])
    mprt_norm = _safe_normalize(df["mprt"])

    # neue Spalten für Radar (semantisch sauber)
    df = df.copy()
    df["faith_norm"] = faith_norm                 # höher = bessere Faithfulness
    df["robust_norm"] = 1.0 - maxsens_norm       # höher = robustere Erklärungen
    df["mprt_norm"] = mprt_norm                  # höhere Werte = "mehr Effekt" im Randomisation-Test

    # Reihenfolge & Labels im Radar
    metrics = ["faith_norm", "robust_norm", "mprt_norm"]
    labels = ["Faithfulness", "Robustheit (1−MaxSens)", "Randomisation"]

    out_dir = metrics_dir / "radar"
    out_dir.mkdir(parents=True, exist_ok=True)

    thesis_dir = _get_thesis_xai_dir()

    # Winkel für die Achsen
    num_vars = len(metrics)
    angles = [n / num_vars * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Kreis schließen

    # Pro Modell ein Radar
    for model in sorted(df["model"].unique()):
        subset = df[df["model"] == model]
        if subset.empty:
            continue

        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)

        # Hintergrund/Style
        ax.set_facecolor("white")
        ax.grid(color="lightgray", alpha=0.4, linewidth=0.7)
        ax.spines["polar"].set_color("black")
        ax.spines["polar"].set_linewidth(1.0)
        ax.tick_params(labelsize=9)

        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])
        ax.yaxis.grid(True, color="lightgray", alpha=0.4, linewidth=0.7)

        # Linien/Flächen pro XAI-Methode
        for method in sorted(subset["method"].unique()):
            row = subset[subset["method"] == method].iloc[0]
            values = [row[m] for m in metrics]
            values += values[:1]  # Kreis schließen

            color = get_color(method)

            ax.plot(
                angles,
                values,
                label=method,
                color=color,
                linewidth=2.5,
            )
            ax.fill(
                angles,
                values,
                color=color,
                alpha=0.22,
            )

        # Achsenbeschriftung
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)

        plt.title(f"XAI Evaluation Radar – {model}", pad=15)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.25, 1.10),
            frameon=False,
            title="XAI-Methode",
        )

        fig.tight_layout()

        png_path = out_dir / f"radar_{model}.png"
        pdf_path = thesis_dir / f"radar_{model}.pdf"

        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")

        plt.close(fig)
        print(f"[OK] Saved PNG: {png_path}")
        print(f"[OK] Saved PDF: {pdf_path}")


def main() -> None:
    plot_quantus_radar()


if __name__ == "__main__":
    main()
