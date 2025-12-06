# visualize_super_grid_all.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

from src import configs
from plot_style import set_confmat_style


# -----------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------

def make_model_input_and_pixels(img_pil: Image.Image):
    """
    Wendet die gleiche Resize-/Graustufen-Transformation an wie das Modell
    und gibt zurück:
      - model_img: PIL-Image (IMG_SIZE×IMG_SIZE, 1 Kanal)
      - pixels: numpy-Array mit Werten in [0,1], shape (H,W)
    """
    tfms = transforms.Compose([
        transforms.Resize((configs.IMG_SIZE, configs.IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    tensor = tfms(img_pil)[0]          # (H, W)
    pixels = tensor.numpy()
    model_img = Image.fromarray(np.uint8(pixels * 255), mode="L")
    return model_img, pixels


def load_explanation(model_name: str, method: str, index: str):
    """
    Lädt eine Heatmap (falls vorhanden) aus outputs/explanations/<model>/<method>/<index>.
    """
    p = configs.EXPL_DIR / model_name / method / index
    if not p.exists():
        print(f"[WARN] Heatmap fehlt: {p}")
        return None
    return Image.open(p)


# -----------------------------------------------------------
# Super-Grid pro Modell
# -----------------------------------------------------------

def make_super_grid_for_model(model_name: str, img_path: Path, index: str = "000.png"):
    """
    Erzeugt ein 2×3 Super-Grid für ein Modell:

    Reihe 1:
      - Original ECG
      - CNN-Input (IMG_SIZE×IMG_SIZE)
      - CNN-Input (Pixelgrid, Werte aus [0,1], nur jeder k-te Pixel beschriftet)

    Reihe 2:
      - GradCAM
      - GradCAM++
      - Integrated Gradients
    """
    set_confmat_style()

    print(f"\n[INFO] Erzeuge Super-Grid für: {model_name}")

    if not img_path.exists():
        print(f"[ERROR] Eingangsbild fehlt: {img_path}")
        return

    # Original-ECG laden
    orig = Image.open(img_path).convert("RGB")

    # Modellinput + echte Pixelwerte erzeugen (IMG_SIZE×IMG_SIZE)
    model_img, model_pixels = make_model_input_and_pixels(orig)
    H, W = model_pixels.shape

    # Heatmaps laden (berechnet auf demselben Modellinput)
    gradcam   = load_explanation(model_name, "gradcam",   index)
    gradcampp = load_explanation(model_name, "gradcam++", index)
    ig        = load_explanation(model_name, "ig",        index)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # ====== Reihe 1: Original / CNN-Input / Pixelgrid ======

    # Original
    axes[0, 0].imshow(orig)
    axes[0, 0].set_title("Original ECG")
    axes[0, 0].axis("off")

    # CNN-Input (IMG_SIZE×IMG_SIZE)
    axes[0, 1].imshow(model_img, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"CNN-Input ({configs.IMG_SIZE}×{configs.IMG_SIZE})")
    axes[0, 1].axis("off")

    # CNN-Input als Pixel-Heatmap
    axes[0, 2].imshow(model_pixels, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Pixelwerte des CNN-Inputs")
    axes[0, 2].axis("off")

    # Nur jeden k-ten Pixel beschriften (sonst völlig unlesbar)
    # Ziel: ca. 28×28 Labels → stride ≈ IMG_SIZE / 28
    target_labels = 28
    stride = max(1, configs.IMG_SIZE // target_labels)

    for i in range(0, H, stride):
        for j in range(0, W, stride):
            val = model_pixels[i, j]
            axes[0, 2].text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color="red",
                fontsize=4,
            )

    # ====== Reihe 2: GradCAM / GradCAM++ / IG ======

    if gradcam is not None:
        axes[1, 0].imshow(gradcam)
        axes[1, 0].set_title("GradCAM")
    axes[1, 0].axis("off")

    if gradcampp is not None:
        axes[1, 1].imshow(gradcampp)
        axes[1, 1].set_title("GradCAM++")
    axes[1, 1].axis("off")

    if ig is not None:
        axes[1, 2].imshow(ig)
        axes[1, 2].set_title("Integrated Gradients")
    axes[1, 2].axis("off")

    fig.suptitle(f"Super-Grid – {model_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # Speichern
    out_dir = configs.THESIS_DIR / "super_grids"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"super_grid_{model_name}.png"
    pdf_path = out_dir / f"super_grid_{model_name}.pdf"

    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=400, bbox_inches="tight")

    plt.close(fig)
    print(f"[OK] Gespeichert: {png_path}")


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():
    """
    Nimmt ein Beispiel-ECG aus dem Testset und erzeugt
    ein Super-Grid für alle Modelle mit vorhandenen Heatmaps.
    """
    # Beispielbild: erste Datei der ersten Klasse im Testset
    sample_class = configs.CLASSES[0]
    cls_dir = configs.DATA_TEST / sample_class
    img_files = sorted(list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg")))
    if not img_files:
        raise FileNotFoundError(f"Keine Testbilder in: {cls_dir}")

    img_path = img_files[0]
    print(f"[INFO] Verwende Testbild: {img_path}")

    # Nur Modelle nutzen, für die es Erklärungen gibt
    models = [m for m in configs.MODEL_NAMES if (configs.EXPL_DIR / m).exists()]

    for model_name in models:
        make_super_grid_for_model(model_name, img_path, index="000.png")


if __name__ == "__main__":
    main()
