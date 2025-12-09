# starten mit: python visualize_ecg_examples.py
# erstellt Beispiel-Gitter für alle Klassen im ECG-Datensatz
from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from src import configs
from plot_style import set_confmat_style, apply_axes_style


def _find_images_for_class(cls: str, n_examples: int) -> list[Path]:
    img_dir = configs.DATA_TRAIN / cls
    if not img_dir.exists():
        raise FileNotFoundError(f"Train-Verzeichnis für Klasse fehlt: {img_dir}")

    files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))

    if not files:
        print(f"[WARN] Keine Bilder für Klasse '{cls}' in {img_dir}")
        return []

    return files[:n_examples]


def make_grid_for_class(cls: str, n_examples: int = 10):
    set_confmat_style()

    files = _find_images_for_class(cls, n_examples=n_examples)
    if not files:
        return

    n = len(files)
    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))
    axes = axes.flatten()

    for ax, f in zip(axes, files):
        try:
            img = Image.open(f).convert("RGB")
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"ERR\n{f.name}", ha="center", va="center")
            print(f"[WARN] Fehler beim Laden: {f} – {e}")

        ax.set_title(f.name, fontsize=8)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Example ECG images – class: {cls}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir = configs.THESIS_DIR / "ecg_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"examples_{cls}.png"
    pdf_path = out_dir / f"examples_{cls}.pdf"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300)

    plt.close()
    print(f"[OK] Saved PNG: {png_path}")
    print(f"[OK] Saved PDF: {pdf_path}")


def main():
    for cls in configs.CLASSES:
        make_grid_for_class(cls)


if __name__ == "__main__":
    main()
