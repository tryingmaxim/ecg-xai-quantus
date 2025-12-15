# starten mit: python visualize_umap_features.py
# visualisiert UMAP der Feature-Repräsentationen für alle Modelle
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import umap

from src import configs
from src.model_def import build_model
from plot_style import set_confmat_style


BATCH_SIZE = 32


def get_device() -> torch.device:
    use_gpu = getattr(configs, "USE_GPU", True)
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_test_loader() -> Tuple[DataLoader, list[str]]:
    tfms = transforms.Compose(
        [
            transforms.Resize((configs.IMG_SIZE, configs.IMG_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=configs.IMAGENET_MEAN, std=configs.IMAGENET_STD),
        ]
    )

    data_root = getattr(configs, "DATA_TEST", Path("data/ecg_test"))
    ds = datasets.ImageFolder(str(data_root), transform=tfms)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    return loader, ds.classes


def load_model(
    model_name: str, num_classes: int, device: torch.device
) -> torch.nn.Module:
    ckpt_path = configs.CKPT_DIR / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {ckpt_path}")

    blob = torch.load(ckpt_path, map_location=device)
    if isinstance(blob, dict):
        state = blob.get("state_dict", blob)
    else:
        state = blob

    model = build_model(
        model_name, num_classes=num_classes, pretrained=configs.PRETRAINED
    )
    clean_state = {
        (k.replace("module.", "") if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    model.load_state_dict(clean_state, strict=False)
    model.to(device)
    model.eval()
    return model


def get_feature_extractor(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    name = model_name.lower()

    if name.startswith("resnet"):
        return torch.nn.Sequential(*list(model.children())[:-1])

    if name.startswith("densenet"):
        return torch.nn.Sequential(
            model.features,
            torch.nn.ReLU(inplace=False),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

    if name.startswith("vgg"):
        return torch.nn.Sequential(
            model.features,
            torch.nn.AdaptiveAvgPool2d((7, 7)),
        )

    if name.startswith("mobilenet"):
        return torch.nn.Sequential(
            model.features,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

    if name.startswith("efficientnet"):
        return torch.nn.Sequential(
            model.features,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
    return torch.nn.Sequential(*list(model.children())[:-1])


def extract_features(model, loader, device, model_name: str):
    feature_extractor = get_feature_extractor(model, model_name).to(device)
    feature_extractor.eval()

    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            f = feature_extractor(x)
            f = f.view(f.size(0), -1)
            feats.append(f.cpu())
            labels.append(y)

    return torch.cat(feats).numpy(), torch.cat(labels).numpy()



def plot_umap(feats, labels, class_names, model_name: str):
    set_confmat_style()

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
    )
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)
    emb = reducer.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(6, 6))
    num_classes = len(class_names)
    cmap = plt.cm.tab10

    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=10,
            color=cmap(cls_idx % 10),
            label=cls_name,
            alpha=0.8,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"UMAP of feature representations – {model_name}")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()

    out_dir = configs.THESIS_DIR / "umap_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{model_name}_umap_features.png"
    pdf_path = out_dir / f"{model_name}_umap_features.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")

    print(f"[OK] Speicherung PNG: {png_path}")
    print(f"[OK] Speicherung PDF: {pdf_path}")
    plt.close(fig)


def main():
    device = get_device()
    loader, class_names = make_test_loader()

    for model_name in configs.MODEL_NAMES:
        try:
            print(f"\n[INFO] UMAP für Modell: {model_name}")
            model = load_model(model_name, num_classes=len(class_names), device=device)
            feats, labels = extract_features(model, loader, device, model_name)
            plot_umap(feats, labels, class_names, model_name)
        except FileNotFoundError as e:
            print(f"[WARN] Überspringe {model_name}: {e}")


if __name__ == "__main__":
    main()
