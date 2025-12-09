# Bewertung des besten Modells auf einem Test-Set: Accuracy/Klassifikationsreport/Konfusionsmatrix/predictions.csv
# Aufruf:
#   python -m src.eval --data_dir data/ecg_test --ckpt outputs/checkpoints/resnet50_best.pt --out_dir outputs/metrics
#   oder gleich alles mit python run_all_eval.py ausführen

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.model_def import build_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=str,
        default="data/test",
        help="Pfad zum Test-Ordner (ImageFolder-Struktur)",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default="outputs/checkpoints/best.pt",
        help="Pfad zum gespeicherten Modell-Checkpoint (.pt)",
    )
    ap.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Bildgröße (Transform, kann Checkpoint überschreiben)",
    )
    ap.add_argument("--batch", type=int, default=64, help="Batch Size")
    ap.add_argument("--num_workers", type=int, default=2, help="DataLoader Worker")
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda oder cpu",
    )
    ap.add_argument(
        "--out_dir", type=str, default="outputs/metrics", help="Ausgabeordner"
    )
    return ap.parse_args()


def build_loader(data_dir: str, img_size: int, batch: int, num_workers: int):
    tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    ds = datasets.ImageFolder(data_dir, tfms)
    loader = DataLoader(
        ds, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return ds, loader


def load_checkpoint_and_model(
    ckpt_path: str, ds_classes, img_size_cli: int, device: str
):
    blob = torch.load(ckpt_path, map_location="cpu")
    if (
        isinstance(blob, dict)
        and "classes" in blob
        and isinstance(blob["classes"], (list, tuple))
    ):
        classes = list(blob["classes"])
    else:
        classes = list(ds_classes)
    if isinstance(blob, dict) and "model_name" in blob:
        model_name = blob["model_name"]
    else:
        model_name = "resnet18"

    model = build_model(model_name, num_classes=len(classes))

    if isinstance(blob, dict) and "state_dict" in blob:
        state = blob["state_dict"]
        img_size_ckpt = blob.get("img_size", None)
    else:
        state = blob
        img_size_ckpt = None

    clean_state = {
        (k.replace("module.", "") if k.startswith("module.") else k): v
        for k, v in state.items()
    }

    try:
        model.load_state_dict(clean_state, strict=True)
    except Exception:
        model.load_state_dict(clean_state, strict=False)

    model.eval().to(device)

    img_size = img_size_cli if img_size_cli is not None else (img_size_ckpt or 224)
    return model, classes, img_size, model_name


def save_confusion_png(
    cm: np.ndarray, class_names, out_png: str, model_name: str = None
):
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
        im = ax.imshow(cm, interpolation="nearest", cmap="Reds")
        if model_name is None:
            title = "Confusion Matrix"
        else:
            title = f"Confusion Matrix - {model_name}"
        ax.set_title(title, fontsize=12)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        thresh = cm.max() / 2.0 if cm.size else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]
                color = "white" if value > thresh else "black"
                ax.text(
                    j,
                    i,
                    int(value),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )

        ax.set_ylabel("True label", fontsize=10)
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_aspect("equal")

        fig.tight_layout()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(
            f"[WARN] Konnte Konfusionsmatrix-PNG nicht speichern ({e}). CSV wurde dennoch geschrieben."
        )


def main():
    args = parse_args()
    ds, loader = build_loader(
        args.data_dir, args.img_size, args.batch, args.num_workers
    )
    model, ckpt_classes, _, model_name = load_checkpoint_and_model(
        args.ckpt, ds.classes, args.img_size, args.device
    )

    ds_idx2name = {i: n for i, n in enumerate(ds.classes)}
    ckpt_name2idx = {n: i for i, n in enumerate(ckpt_classes)}

    def remap_true_indices(y_true_indices: np.ndarray) -> np.ndarray:
        names = [ds_idx2name[i] for i in y_true_indices]
        return np.array([ckpt_name2idx[n] for n in names], dtype=int)

    y_true_idx_batches = []
    y_pred_idx_batches = []
    prob_batches = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(args.device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            y_true_idx_batches.append(y.numpy())
            y_pred_idx_batches.append(logits.argmax(dim=1).cpu().numpy())
            prob_batches.append(probs.cpu().numpy())

    y_true_idx = (
        np.concatenate(y_true_idx_batches) if y_true_idx_batches else np.array([])
    )
    y_pred_idx = (
        np.concatenate(y_pred_idx_batches) if y_pred_idx_batches else np.array([])
    )
    probs_all = (
        np.concatenate(prob_batches, axis=0)
        if prob_batches
        else np.empty((0, len(ckpt_classes)))
    )

    model_out_dir = os.path.join(args.out_dir, model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    if y_true_idx.size == 0:
        print("[FEHLER] Keine Testdaten gefunden. Prüfe --data_dir und Ordnerstruktur.")
        return

    y_true_remap = remap_true_indices(y_true_idx)

    acc = accuracy_score(y_true_remap, y_pred_idx)
    cm = confusion_matrix(
        y_true_remap, y_pred_idx, labels=list(range(len(ckpt_classes)))
    )
    rep = classification_report(
        y_true_remap,
        y_pred_idx,
        target_names=ckpt_classes,
        digits=3,
        zero_division=0,
    )

    paths = [p for (p, _) in ds.samples]

    rows = []
    for i, p in enumerate(paths):
        true_name_ds = ds_idx2name[y_true_idx[i]]
        true_name = ckpt_classes[ckpt_name2idx[true_name_ds]]
        pred_name = ckpt_classes[y_pred_idx[i]]

        row = {
            "path": p.replace("\\", "/"),
            "true": true_name,
            "pred": pred_name,
            "correct": bool(true_name == pred_name),
        }
        for cls_idx, cls_name in enumerate(ckpt_classes):
            row[f"prob_{cls_name}"] = float(probs_all[i, cls_idx])

        rows.append(row)

    pred_csv = os.path.join(model_out_dir, "predictions.csv")
    pd.DataFrame(rows).to_csv(pred_csv, index=False, encoding="utf-8")
    print(f"  - {pred_csv}")
    cm_csv = os.path.join(model_out_dir, "confusion_matrix.csv")
    rep_txt = os.path.join(model_out_dir, "classification_report.txt")
    cm_png = os.path.join(model_out_dir, "confusion_matrix.png")

    pd.DataFrame(cm, index=ckpt_classes, columns=ckpt_classes).to_csv(
        cm_csv, encoding="utf-8"
    )
    with open(rep_txt, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{rep}")
    save_confusion_png(cm, ckpt_classes, cm_png, model_name=model_name)

    print(f"Accuracy: {acc:.4f}")
    print(f"Saved:\n  - {cm_csv}\n  - {rep_txt}\n  - {cm_png}")


if __name__ == "__main__":
    main()
