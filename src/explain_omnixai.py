# src/explain_omnixai.py

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image as PilImage

from src import configs
from .model_def import build_model

# KORREKTE OmniXAI-Imports
from omnixai.data.image import Image as OmniImage
from omnixai.explainers.vision.specific.gradcam.gradcam import GradCAM, GradCAMPlus
from omnixai.explainers.vision.specific.ig import IntegratedGradientImage
DEBUG = True

# --------------------- Checkpoint Laden -----------------------------------

def load_checkpoint(ckpt_path, device):
    blob = torch.load(ckpt_path, map_location="cpu")

    if isinstance(blob, dict) and "classes" in blob:
        classes = list(blob["classes"])
    else:
        classes = list(configs.CLASSES)

    # Modellname sauber extrahieren
    stem = Path(ckpt_path).stem
    model_name = stem.replace("_best", "")

    model = build_model(model_name, num_classes=len(classes))

    state = blob["state_dict"] if isinstance(blob, dict) else blob
    clean = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean, strict=False)
    model.to(device).eval()
    return model, classes, model_name


# ---------------------- Transforms ---------------------------------------

_TFM = T.Compose([
    T.Resize((configs.IMG_SIZE, configs.IMG_SIZE)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def make_preprocess(device):
    def preprocess_fn(batch: OmniImage):
        tensors = []
        for img in batch:
            pil = img.to_pil()
            t = _TFM(pil)
            tensors.append(t)
        return torch.stack(tensors).to(device)

    return preprocess_fn


# ---------------------- Letzte Conv finden --------------------------------

def find_last_conv(model):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


# ---------------------- Heatmap Overlay ----------------------------------

def save_overlay(original_pil, heatmap, out_path):
    import cv2
    orig = np.array(original_pil.convert("RGB"))

    # Heatmap normalisieren
    heatmap = heatmap.astype(np.float32)
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap = (heatmap * 255).astype(np.uint8)

    # Größe anpassen
    heatmap = PilImage.fromarray(heatmap).resize((orig.shape[1], orig.shape[0]))
    heatmap = np.array(heatmap)

    # Colormap + Overlay
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[:, :, ::-1]
    overlay = (0.55 * orig + 0.45 * colored).clip(0, 255).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PilImage.fromarray(overlay).save(out_path)


# ---------------------- CLI ----------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--method", required=True, choices=["gradcam", "gradcam++", "ig"])
    ap.add_argument("--data_dir", default="data/ecg_test")
    ap.add_argument("--limit", type=int, default=10)
    return ap.parse_args()


# ---------------------- MAIN ----------------------------------------------

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model, classes, model_name = load_checkpoint(args.ckpt, device)
    preprocess_fn = make_preprocess(device)
    last_conv = find_last_conv(model)

    # Explainer auswählen
    if args.method == "gradcam":
        explainer = GradCAM(model=model, target_layer=last_conv,
                            preprocess_function=preprocess_fn, mode="classification")
    elif args.method == "gradcam++":
        explainer = GradCAMPlus(model=model, target_layer=last_conv,
                                preprocess_function=preprocess_fn, mode="classification")
    else:
        explainer = IntegratedGradientImage(model=model,
                                            preprocess_function=preprocess_fn,
                                            mode="classification")

    # Bilder einlesen
    data_dir = Path(args.data_dir)
    img_paths = sorted(list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.png")))
    img_paths = img_paths[:args.limit]

    if not img_paths:
        print("[ERROR] Keine Bilder gefunden.")
        return

    print(f"[INFO] Erkläre {len(img_paths)} Bilder...")

    out_root = configs.EXPL_DIR / model_name / args.method
    out_root.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(img_paths):
        pil = PilImage.open(path).convert("RGB")
        omni = OmniImage(pil, batched=False)

        explanation = explainer.explain(omni)

        # HEATMAP aus dict extrahieren (KORREKT!)
        explan = explanation.get_explanations()
        print("\nDEBUG: OUTPUT VON get_explanations():")
        print(explan)
        print("\n")

        explanation_data = explan[0]

        # Finaler Heatmap-Schlüssel (deine OmniXAI-Version nutzt überall "scores")
        if "scores" in explanation_data:
            heat = explanation_data["scores"]

        elif "data" in explanation_data:
            heat = explanation_data["data"]

        elif "importances" in explanation_data:
            heat = explanation_data["importances"]

        elif isinstance(explanation_data, np.ndarray):
            heat = explanation_data

        else:
            raise ValueError(f"Unbekanntes Heatmap-Format: {explanation_data.keys()}")



        out_path = out_root / f"{i:03d}.png"
        save_overlay(pil, heat, out_path)

        print(f"[OK] {args.method} -> {out_path}")

    print(f"[FINISHED] Alle Erklärungen gespeichert unter: {out_root}")


if __name__ == "__main__":
    main()
