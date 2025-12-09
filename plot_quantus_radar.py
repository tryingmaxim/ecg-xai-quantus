# starten mit: python plot_quantus_radar.py
#  erstellt Radar-Plots für die Quantus-Metriken verschiedener XAI-Methoden
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
    return getattr(configs, "METRICS_DIR", Path("outputs/metrics"))


def _get_thesis_xai_dir() -> Path:
    thesis_base = getattr(configs, "THESIS_DIR", Path("outputs/thesis_figures"))
    xai_dir = Path(thesis_base) / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)
    return xai_dir


def _safe_normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        return pd.Series(0.5, index=s.index)

    s_min = s.min()
    s_max = s.max()
    denom = s_max - s_min

    if denom <= 0:
        return pd.Series(0.5, index=s.index)

    return (s - s_min) / denom


def _check_required_columns(df: pd.DataFrame, cols: List[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Benötigte Spalten {missing} fehlen in {df_name}. "
            f"Verfügbare Spalten: {list(df.columns)}"
        )


def plot_quantus_radar() -> None:
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

    faith_norm = _safe_normalize(df["faithfulness_corr"])
    maxsens_norm = _safe_normalize(df["max_sens"])
    mprt_norm = _safe_normalize(df["mprt"])

    df = df.copy()
    df["faith_norm"] = faith_norm
    df["robust_norm"] = 1.0 - maxsens_norm
    df["mprt_norm"] = mprt_norm

    metrics = ["faith_norm", "robust_norm", "mprt_norm"]
    labels = ["Faithfulness", "Robustness (1−MaxSensitivity)", "Randomisation (MPRT)"]

    out_dir = metrics_dir / "radar"
    out_dir.mkdir(parents=True, exist_ok=True)

    thesis_dir = _get_thesis_xai_dir()

    num_vars = len(metrics)
    angles = [n / num_vars * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    for model in sorted(df["model"].unique()):
        subset = df[df["model"] == model]
        if subset.empty:
            continue

        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)

        ax.set_facecolor("white")
        ax.grid(color="lightgray", alpha=0.4, linewidth=0.7)
        ax.spines["polar"].set_color("black")
        ax.spines["polar"].set_linewidth(1.0)
        ax.tick_params(labelsize=9)

        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])
        ax.yaxis.grid(True, color="lightgray", alpha=0.4, linewidth=0.7)

        for method in sorted(subset["method"].unique()):
            row = subset[subset["method"] == method].iloc[0]
            values = [row[m] for m in metrics]
            values += values[:1]

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

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)

        plt.title(f"XAI evaluation radar – {model}", pad=15)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.25, 1.10),
            frameon=False,
            title="XAI method",
        )

        fig.tight_layout()

        png_path = out_dir / f"radar_{model}.png"
        pdf_path = thesis_dir / f"radar_{model}.pdf"

        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")

        plt.close(fig)
        print(f"[OK] Saved PNG: {png_path}")
        print(f"[OK] Saved PDF: {pdf_path}")


def main() -> None:
    plot_quantus_radar()


if __name__ == "__main__":
    main()
