# plot_quantus_results.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    df = pd.read_csv("outputs/metrics/quantus_summary.csv")

    plots = [
        ("faithfulness_corr", "Faithfulness (FaithfulnessCorrelation)", "faithfulness.png"),
        ("max_sens", "Robustness (MaxSensitivity)", "robustness.png"),
        ("mprt", "Randomisation (MPRT)", "mprt.png"),
    ]

    out_dir = Path("outputs/metrics/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric, title, fname in plots:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df, x="model", y=metric, hue="method")
        plt.title(title)
        plt.ylabel(metric)
        plt.tight_layout()
        out_path = out_dir / fname
        plt.savefig(out_path, dpi=300)
        plt.close()
        print("[OK] Saved:", out_path)

if __name__ == "__main__":
    main()
