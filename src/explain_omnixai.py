import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image as PilImage

from src import configs
from .model_def import build_model

from omnixai.data.image import Image as OmniImage
from omnixai.explainers.vision.specific.gradcam.gradcam import GradCAM, GradCAMPlus
from omnixai.explainers.vision.specific.ig import IntegratedGradientImage
from omnixai.explainers.vision.agnostic.lime import LimeImage


def disable_inplace_relu(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False


def load_checkpoint(ckpt_path: Path, device: torch.device):
    blob = torch.load(ckpt_path, map_location="cpu")

    if isinstance(blob, dict) and "classes" in blob:
        classes = list(blob["classes"])
    else:
        classes = list(configs.CLASSES)

    stem = Path(ckpt_path).stem
    model_name = stem.replace("_best", "")

    model = build_model(model_name, num_classes=len(classes), pretrained=configs.PRETRAINED)

    state = blob["state_dict"] if isinstance(blob, dict) else blob
    clean = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean, strict=False)

    disable_inplace_relu(model)
    model.to(device).eval()
    return model, classes, model_name


_TFM = T.Compose(
    [
        T.Resize((configs.IMG_SIZE, configs.IMG_SIZE)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=configs.IMAGENET_MEAN, std=configs.IMAGENET_STD)
    ]
)


def make_preprocess(device: torch.device):
    def preprocess_fn(batch: OmniImage):
        tensors = []
        for img in batch:
            pil = img.to_pil()
            t = _TFM(pil)
            tensors.append(t)
        return torch.stack(tensors).to(device)

    return preprocess_fn


def make_predict_function(model: nn.Module, preprocess_fn):
    def predict(batch: OmniImage):
        model.eval()
        with torch.no_grad():
            x = preprocess_fn(batch)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    return predict


def find_last_conv(model: nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


def save_overlay(original_pil: PilImage.Image, heatmap: np.ndarray, out_path: Path):
    import cv2

    orig = np.array(original_pil.convert("RGB"))

    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap = (heatmap * 255).astype(np.uint8)

    heatmap = PilImage.fromarray(heatmap).resize((orig.shape[1], orig.shape[0]))
    heatmap = np.array(heatmap)

    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[:, :, ::-1]
    overlay = (0.55 * orig + 0.45 * colored).clip(0, 255).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PilImage.fromarray(overlay).save(out_path)


def to_2d_heatmap(heat: np.ndarray) -> np.ndarray:
    heat = np.asarray(heat)

    if heat.ndim == 3:
        if heat.shape[-1] == 1:
            heat = heat[..., 0]
        elif heat.shape[0] == 1:
            heat = heat[0]
        else:
            if heat.shape[-1] in (3, 4):
                heat = heat.mean(axis=-1)
            else:
                heat = heat.mean(axis=0)

    if heat.ndim != 2:
        raise ValueError(
            f"Heatmap konnte nicht auf 2D gebracht werden, shape={heat.shape}"
        )

    heat = heat.astype(np.float32)
    return heat


def normalise_01(hm: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    hm = hm.astype(np.float32)
    hm = hm - hm.min()
    mx = hm.max()
    if mx > eps:
        hm = hm / mx
    return hm


def save_heatmap_only(
    heat: np.ndarray,
    out_npy: Path,
    out_png: Path | None,
    target_size: tuple[int, int],
):
    hm2d = to_2d_heatmap(heat)
    hm2d = normalise_01(hm2d)

    hm_img = PilImage.fromarray((hm2d * 255).astype(np.uint8))
    hm_img = hm_img.resize(target_size)
    hm2d_resized = np.array(hm_img).astype(np.float32) / 255.0

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, hm2d_resized.astype(np.float32))

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        PilImage.fromarray((hm2d_resized * 255).astype(np.uint8)).save(out_png)


def save_overlay_from_heatmap(
    original_pil: PilImage.Image, heat: np.ndarray, out_path: Path
):
    import cv2

    orig = np.array(original_pil.convert("RGB"))
    hm2d = normalise_01(to_2d_heatmap(heat))

    heat_u8 = (hm2d * 255).astype(np.uint8)
    heat_u8 = PilImage.fromarray(heat_u8).resize((orig.shape[1], orig.shape[0]))
    heat_u8 = np.array(heat_u8)

    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)[:, :, ::-1]
    overlay = (0.55 * orig + 0.45 * colored).clip(0, 255).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PilImage.fromarray(overlay).save(out_path)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--method",
        required=True,
        choices=["gradcam", "gradcam++", "ig", "lime"],
    )
    ap.add_argument("--data_dir", default="data/ecg_test")
    ap.add_argument("--limit", type=int, default=10)
    return ap.parse_args()


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model, classes, model_name = load_checkpoint(Path(args.ckpt), device)
    preprocess_fn = make_preprocess(device)

    data_dir = Path(args.data_dir)
    img_paths = sorted(list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.png")))
    img_paths = img_paths[: args.limit]

    if not img_paths:
        print("[ERROR] Keine Bilder gefunden.")
        return

    print(f"[INFO] Erkläre {len(img_paths)} Bilder mit {args.method}...")

    last_conv = find_last_conv(model)
    print(f"[DEBUG] Last conv layer: {last_conv}")
    
    if args.method == "gradcam":
        explainer = GradCAM(
            model=model,
            target_layer=last_conv,
            preprocess_function=preprocess_fn,
            mode="classification",
        )
    elif args.method == "gradcam++":
        explainer = GradCAMPlus(
            model=model,
            target_layer=last_conv,
            preprocess_function=preprocess_fn,
            mode="classification",
        )
    elif args.method == "ig":
        explainer = IntegratedGradientImage(
            model=model,
            preprocess_function=preprocess_fn,
            mode="classification",
        )
    else:
        predict_fn = make_predict_function(model, preprocess_fn)
        explainer = LimeImage(predict_function=predict_fn, mode="classification")

    out_root = configs.EXPL_DIR / model_name / args.method
    out_root.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(img_paths):
        pil = PilImage.open(path).convert("RGB")
        omni = OmniImage(pil, batched=False)

        if args.method == "lime":
            explanation = explainer.explain(omni, hide_color=0, num_samples=800)
        else:
            explanation = explainer.explain(omni)

        explan_list = explanation.get_explanations()
        explanation_data = explan_list[0]

        heat = None
        if isinstance(explanation_data, np.ndarray):
            heat = explanation_data
        elif "scores" in explanation_data:
            heat = explanation_data["scores"]
        elif "data" in explanation_data:
            heat = explanation_data["data"]
        elif "importances" in explanation_data:
            heat = explanation_data["importances"]
        elif "masks" in explanation_data:
            masks = explanation_data["masks"]
            heat = masks[0]

        if heat is None:
            raise ValueError(
                f"Unbekanntes Heatmap-Format für Methode {args.method}: "
                f"Keys = {getattr(explanation_data, 'keys', lambda: [])()}"
            )

        overlay_dir = out_root / "overlay"
        heatmap_dir = out_root / "heatmap"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        overlay_path = overlay_dir / f"{i:03d}.png"
        heatmap_npy = heatmap_dir / f"{i:03d}.npy"
        heatmap_png = heatmap_dir / f"{i:03d}.png"

        save_heatmap_only(
            heat=heat,
            out_npy=heatmap_npy,
            out_png=heatmap_png,
            target_size=pil.size,
        )

        save_overlay_from_heatmap(pil, heat, overlay_path)

        print(f"[OK] {args.method} -> {overlay_path} (+ heatmap-only: {heatmap_npy})")

    print(f"[FINISHED] Alle Erklärungen gespeichert unter: {out_root}")


if __name__ == "__main__":
    main()
