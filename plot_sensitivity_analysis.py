# starten mit: python plot_sensitivity_analysis.py
# erstellt Trend-Plot für MaxSensitivity verschiedener Modelle und XAI-Methoden
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import set_confmat_style, apply_axes_style, get_color
from src import configs


def _get_metrics_dir() -> Path:
    return getattr(configs, "METRICS_DIR", Path("outputs/metrics"))


def _get_thesis_xai_dir() -> Path:
    thesis_base = getattr(configs, "THESIS_DIR", Path("outputs/thesis_figures"))
    xai_dir = Path(thesis_base) / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)
    return xai_dir


def _check_required_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Benötigte Spalten {missing} fehlen in {name}. "
            f"Verfügbare Spalten: {list(df.columns)}"
        )


def _pretty_method_name(method: str) -> str:
    mapping = {
        "gradcam": "Grad-CAM",
        "gradcam++": "Grad-CAM++",
        "ig": "Integrated Gradients",
        "lime": "LIME",
    }
    return mapping.get(method, method)


def plot_sensitivity_trend() -> None:
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
        ["model", "method", "max_sens"],
        "quantus_summary.csv",
    )

    if df.empty:
        raise ValueError("quantus_summary.csv enthält keine Daten.")

    has_value = (
        df.groupby("model")["max_sens"].apply(lambda s: s.notna().any()).sort_index()
    )
    models = [m for m, ok in has_value.items() if ok]
    if not models:
        raise ValueError("Keine Modelle mit gültigen MaxSensitivity-Werten gefunden.")

    methods = sorted(df["method"].unique())

    out_dir = metrics_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    thesis_dir = _get_thesis_xai_dir()

    x = np.arange(len(models), dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4))

    for method in methods:
        sub = df[df["method"] == method].set_index("model")
        sub = sub.reindex(models)

        y = sub["max_sens"].values

        if np.all(np.isnan(y)):
            continue

        ax.plot(
            x,
            y,
            marker="o",
            linestyle="-",
            linewidth=1.8,
            label=_pretty_method_name(method),
            color=get_color(method),
        )

    apply_axes_style(ax)
    ax.set_ylabel("MaxSensitivity\n(lower = more robust explanations)")
    ax.set_xlabel("Model")
    ax.set_title("Trend in explanation robustness (MaxSensitivity)")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")

    ax.legend(title="XAI method", loc="best")

    fig.tight_layout()

    png_path = out_dir / "sensitivity_trend.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    pdf_path = thesis_dir / "sensitivity_trend.pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"[OK] Saved PNG: {png_path}")
    print(f"[OK] Saved PDF: {pdf_path}")


def main() -> None:
    plot_sensitivity_trend()


if __name__ == "__main__":
    main()
