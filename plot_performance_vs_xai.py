# starten mit: python plot_performance_vs_xai.py
# erstellt ein Streudiagramm, das die Modellgüte (Accuracy) gegen die Erklärungsqualität (FaithfulnessCorrelation) darstellt
from __future__ import annotations

from pathlib import Path
from typing import Optional

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


def _check_required_columns(df: pd.DataFrame, cols: list[str], df_name: str) -> None:
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

    if "model" not in df_p.columns and "Model" in df_p.columns:
        rename_map = {"Model": "model"}
        if "Test Accuracy" in df_p.columns:
            rename_map["Test Accuracy"] = "accuracy"
        df_p = df_p.rename(columns=rename_map)

    _check_required_columns(
        df_q,
        ["model", "method", "faithfulness_corr", "max_sens"],
        "quantus_summary.csv",
    )
    _check_required_columns(df_p, ["model", "accuracy"], "model_performance.csv")

    df = df_q.merge(df_p, on="model", how="left")
    df = df.dropna(subset=["accuracy", "faithfulness_corr", "max_sens"])
    if df.empty:
        raise ValueError(
            "Keine gültigen Zeilen nach dem Zusammenführen von Quantus- "
            "und Performance-Daten."
        )

    out_dir = metrics_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(18, 8))

    max_sens_norm = _safe_normalize(df["max_sens"])
    sizes = (1.0 - max_sens_norm) * 600.0 + 100.0

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

    for _, row in df.iterrows():
        ax.text(
            float(row["accuracy"]) + 0.002,
            float(row["faithfulness_corr"]) + 0.002,
            str(row["model"]),
            fontsize=8,
        )

    apply_axes_style(ax)

    x_min, x_max = df["accuracy"].min(), df["accuracy"].max()
    x_min_plot = max(0.24, x_min - 0.02)
    x_max_plot = min(1.02, x_max + 0.02)
    ax.set_xlim(x_min_plot, x_max_plot)

    x_ticks = np.arange(
        np.floor(x_min_plot * 100) / 100.0,
        np.ceil(x_max_plot * 100) / 100.0 + 1e-9,
        0.02,
    )
    ax.set_xticks(x_ticks)
    x_tick_labels = [f"{t:.2f}" if i % 2 == 0 else "" for i, t in enumerate(x_ticks)]
    ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")

    y_min, y_max = df["faithfulness_corr"].min(), df["faithfulness_corr"].max()
    y_min_plot = y_min - 0.02
    y_max_plot = y_max + 0.02
    ax.set_ylim(y_min_plot, y_max_plot)

    y_ticks = np.arange(
        np.floor(y_min_plot * 100) / 100.0,
        np.ceil(y_max_plot * 100) / 100.0 + 1e-9,
        0.02,
    )
    ax.set_yticks(y_ticks)
    y_tick_labels = [f"{t:.2f}" if i % 2 == 0 else "" for i, t in enumerate(y_ticks)]
    ax.set_yticklabels(y_tick_labels)

    ax.axhline(0.0, color="black", linewidth=0.8)

    ax.set_xlabel("Accuracy (test set)")
    ax.set_ylabel("FaithfulnessCorrelation")
    ax.set_title(
        "Trade-off: model performance vs. explanation quality\n"
        "(Bubble size ≈ explanation robustness, 1 − MaxSensitivity)"
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        labels_handles = sorted(zip(labels, handles), key=lambda x: x[0])
        labels, handles = zip(*labels_handles)
        ax.legend(handles, labels, title="XAI method", loc="upper left")

    fig.tight_layout()

    out_path = out_dir / "performance_vs_xai.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    thesis_dir = _get_thesis_xai_dir()
    thesis_pdf = thesis_dir / "performance_vs_xai.pdf"
    fig.savefig(thesis_pdf, dpi=300, bbox_inches="tight")

    plt.close(fig)

    print(f"[OK] Saved PNG : {out_path}")
    print(f"[OK] Saved PDF : {thesis_pdf}")


def main() -> None:
    plot_performance_vs_xai()


if __name__ == "__main__":
    main()
