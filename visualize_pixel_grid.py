# visualize_pixel_grid.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

from src import configs
from plot_style import set_confmat_style


def visualize_pixel_grid(
    img_path: Path,
    size: int = 28,
    cmap: str = "gray"
) -> None:
    """
    Visualisiert ein ECG als 28×28-Pixelbild + zugehöriges Pixelraster.
    Links: skaliertes ECG-Bild
    Rechts: Pixelwerte (jeweils als Zahl in Rot dargestellt)
    """

    set_confmat_style()

    if not img_path.exists():
        raise FileNotFoundError(f"Bild nicht gefunden: {img_path}")

    # === 1. Bild laden ===
    img = Image.open(img_path).convert("L")

    # === 2. Auf 28×28 skalieren ===
    tfms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),   # Werte in [0,1]
    ])
    tensor = tfms(img)[0]       # shape (H, W)
    pixels = tensor.numpy()     # float32, Werte ∈ [0,1]

    # Für Anzeige linkes Bild als 0–255 Graustufen
    img_28 = Image.fromarray(np.uint8(pixels * 255), mode="L")

    # === 3. Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ---- Links: skaliertes ECG ----
    axes[0].imshow(img_28, cmap=cmap, vmin=0, vmax=255)
    axes[0].set_title(f"ECG ({size}×{size} Input)")
    axes[0].axis("off")

    # ---- Rechts: Pixel Grid ----
    axes[1].imshow(pixels, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title(f"Pixelwerte ({size}×{size})")
    axes[1].axis("off")

    # Zahlen einzeichnen (immer ROT)
    for i in range(size):
        for j in range(size):
            val = pixels[i, j]
            axes[1].text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="red",       # konstant rot
            )

    fig.tight_layout()

    # === 4. Speichern ===
    out_dir = configs.THESIS_DIR / "pixel_grids"
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"pixel_grid_{img_path.stem}_{size}x{size}"
    png_path = out_dir / f"{fname}.png"
    pdf_path = out_dir / f"{fname}.pdf"

    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=400, bbox_inches="tight")

    plt.close(fig)

    print(f"[OK] Saved PNG: {png_path}")
    print(f"[OK] Saved PDF: {pdf_path}")


def main():
    """
    Beispiel: erstes Testbild der ersten Klasse laden.
    """
    sample_class = configs.CLASSES[0]
    cls_dir = configs.DATA_TEST / sample_class

    img_files = sorted(list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg")))
    if not img_files:
        raise FileNotFoundError(f"Keine Testbilder in: {cls_dir}")

    visualize_pixel_grid(img_files[0], size=28)


if __name__ == "__main__":
    main()
