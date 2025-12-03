# plot_quantus_radar.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path

def main():
    df = pd.read_csv("outputs/metrics/quantus_summary.csv")

    metrics = ["faithfulness_corr", "max_sens", "mprt"]
    labels = ["Faithfulness", "Robustness", "Randomisation"]

    df_norm = df.copy()
    for m in metrics:
        mn, mx = df_norm[m].min(), df_norm[m].max()
        df_norm[m] = (df_norm[m] - mn) / (mx - mn + 1e-9)

    out_dir = Path("outputs/metrics/radar")
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in df_norm["model"].unique():
        subset = df_norm[df_norm["model"] == model]

        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)

        angles = [n / len(metrics) * 2 * pi for n in range(len(metrics))]
        angles += angles[:1]

        for method in subset["method"].unique():
            row = subset[subset["method"] == method].iloc[0]
            values = row[metrics].tolist() + row[metrics].tolist()[:1]

            ax.plot(angles, values, label=method)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)

        plt.title(f"XAI Evaluation Radar – {model}")
        plt.legend(loc="lower left", bbox_to_anchor=(0.0, -0.15))
        plt.tight_layout()

        out_path = out_dir / f"radar_{model}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print("[OK] Saved:", out_path)

if __name__ == "__main__":
    main()
