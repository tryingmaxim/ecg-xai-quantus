# starten mit python analyze_dataset_distribution.py
# erstellt eine Donut-Plot-Grafik, die die Verteilung der Klassen im ECG-Trainingsdatensatz zeigt
from pathlib import Path
import matplotlib.pyplot as plt

DATA_TRAIN = Path("data/ecg_train")


CLASSES = ["Abnormal", "HistoryMI", "MI", "Normal"]


def count_images_per_class():
    counts = {}
    for cls in CLASSES:
        cls_dir = DATA_TRAIN / cls
        if not cls_dir.exists():
            print(f"[WARN] Ordner fehlt: {cls_dir}")
            counts[cls] = 0
            continue

        n = sum(1 for p in cls_dir.glob("*.jpg")) + sum(
            1 for p in cls_dir.glob("*.png")
        )
        counts[cls] = n
    return counts


def plot_donut(counts):
    labels = list(counts.keys())
    values = list(counts.values())
    total = sum(values)
    if total == 0:
        print("[ERR] Keine Bilder gefunden.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        pctdistance=0.8,
    )
    centre_circle = plt.Circle((0, 0), 0.50, fc="white")
    fig.gca().add_artist(centre_circle)

    ax.set_title("Class distribution of ECG training data")

    legend_labels = [f"{cls} (n={counts[cls]})" for cls in labels]
    ax.legend(
        wedges,
        legend_labels,
        title="Classes",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    fig.tight_layout()
    out_path = Path("outputs/thesis_figures/dataset_distribution.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"[OK] Speicherung: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    counts = count_images_per_class()
    print(counts)
    plot_donut(counts)
