# plot_performance_vs_xai.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import set_confmat_style, apply_axes_style, get_color
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
    Falls alle Werte gleich sind oder NaNs vorliegen, wird eine Konstante zurückgegeben.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        # alles NaN -> konstante 0.5
        return pd.Series(0.5, index=s.index)

    s_min = s.min()
    s_max = s.max()
    denom = s_max - s_min

    if denom <= 0:
        # alle Werte gleich -> alles 0.5
        return pd.Series(0.5, index=s.index)

    return (s - s_min) / denom


def _check_required_columns(df: pd.DataFrame, cols: list[str], df_name: str) -> None:
    """Stellt sicher, dass alle benötigten Spalten vorhanden sind."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Benötigte Spalten {missing} fehlen in {df_name}. "
            f"Verfügbare Spalten: {list(df.columns)}"
        )


def plot_performance_vs_xai(
    quantus_path: Optional[Path] = None,
    perf_path: Optional[Path] = None,
) -> None:
    """
    Erzeugt einen Bubble-Scatterplot:
    x-Achse: Modell-Accuracy
    y-Achse: FaithfulnessCorrelation (Quantus)
    Farbe: XAI-Methode
    Bubble-Größe: (1 - normalisierte MaxSensitivity) → kleinere Sensitivity = größere Blase.
    """

    set_confmat_style()

    metrics_dir = _get_metrics_dir()
    quantus_path = quantus_path or (metrics_dir / "quantus_summary.csv")
    perf_path = perf_path or (metrics_dir / "model_performance.csv")

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

    # Minimal benötigte Spalten prüfen
    _check_required_columns(df_q, ["model", "method", "faithfulness_corr", "max_sens"], "quantus_summary.csv")
    _check_required_columns(df_p, ["model", "accuracy"], "model_performance.csv")

    # Mergen
    df = df_q.merge(df_p, on="model", how="left")

    # Zeilen mit fehlender Accuracy oder Faithfulness rausfiltern
    df = df.dropna(subset=["accuracy", "faithfulness_corr", "max_sens"])
    if df.empty:
        raise ValueError(
            "Keine gültigen Zeilen nach dem Zusammenführen von Quantus- und Performance-Daten."
        )

    out_dir = metrics_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Bubble-Size basierend auf MaxSensitivity
    max_sens_norm = _safe_normalize(df["max_sens"])
    # Größere Blasen bei geringerer Sensitivity (robustere Erklärungen)
    sizes = (1.0 - max_sens_norm) * 600.0 + 100.0

    # Scatter pro Methode (für Legende)
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        if sub.empty:
            continue

        method_color = get_color(method)

        ax.scatter(
            sub["accuracy"],
            sub["faithfulness_corr"],
            s=sizes.loc[sub.index],
            color=method_color,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            label=method,
        )

    # Modelle beschriften (leicht versetzt, um Überlappung etwas zu reduzieren)
    for _, row in df.iterrows():
        ax.text(
            float(row["accuracy"]) + 0.002,
            float(row["faithfulness_corr"]) + 0.002,
            str(row["model"]),
            fontsize=8,
        )

    apply_axes_style(ax)
    ax.set_xlabel("Genauigkeit (Accuracy)")
    ax.set_ylabel("FaithfulnessCorrelation")
    ax.set_title(
        "Trade-off: Modellgüte vs. Erklärqualität\n"
        "(Blasengröße ≈ Robustheit der Erklärungen, 1 − MaxSensitivity)"
    )

    # Legende sortiert (alphabetisch nach Methode)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        labels_handles = sorted(zip(labels, handles), key=lambda x: x[0])
        labels, handles = zip(*labels_handles)
        ax.legend(handles, labels, title="XAI-Methode", loc="best")

    fig.tight_layout()

    # Speichern
    out_path = out_dir / "performance_vs_xai.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    thesis_dir = _get_thesis_xai_dir()
    thesis_pdf = thesis_dir / "performance_vs_xai.pdf"
    fig.savefig(thesis_pdf, bbox_inches="tight")

    plt.close(fig)

    print(f"[OK] Saved PNG : {out_path}")
    print(f"[OK] Saved PDF : {thesis_pdf}")


def main() -> None:
    """Entry-Point für CLI-Nutzung."""
    plot_performance_vs_xai()


if __name__ == "__main__":
    main()
