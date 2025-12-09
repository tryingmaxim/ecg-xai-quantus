# mit python run_all_quantus.py ausführen (um alle Modelle und Methoden zu testen)
# Dieses Skript lädt Heatmaps aus Ordnern, nur von Resnet-Modellen werden MPRT-Metriken berechnet.
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import pandas as pd
import quantus

from src import configs
from src.model_def import build_model


def load_images_with_labels(data_dir, img_size, limit=None):
    data_dir = Path(data_dir)
    paths = sorted(list(data_dir.rglob("*.png")) + list(data_dir.rglob("*.jpg")))
    if limit is not None:
        paths = paths[:limit]

    if not paths:
        raise RuntimeError(f"Keine Bilder in {data_dir} gefunden.")

    class_to_idx = {name: idx for idx, name in enumerate(configs.CLASSES)}

    imgs = []
    labels = []
    for p in paths:
        cls_name = p.parent.name
        if cls_name not in class_to_idx:
            raise RuntimeError(
                f"Ordnername '{cls_name}' nicht in configs.CLASSES: "
                f"{list(class_to_idx.keys())}"
            )
        label = class_to_idx[cls_name]
        labels.append(label)

        img = Image.open(p).convert("RGB")
        img = img.resize((img_size, img_size))
        arr = np.array(img).astype(np.float32) / 255.0
        imgs.append(arr.transpose(2, 0, 1))

    X = np.stack(imgs)
    y = np.array(labels, dtype=np.int64)
    return X, y, paths


def load_heatmaps(hm_dir, img_size, limit=None):
    hm_dir = Path(hm_dir)
    paths = sorted(list(hm_dir.glob("*.png")) + list(hm_dir.glob("*.jpg")))
    if limit is not None:
        paths = paths[:limit]

    if not paths:
        raise RuntimeError(f"Keine Heatmaps in {hm_dir} gefunden.")

    hmaps = []
    for p in paths:
        img = Image.open(p).convert("L")
        img = img.resize((img_size, img_size))
        arr = np.array(img).astype(np.float32) / 255.0
        hmaps.append(arr[None, ...])

    H = np.stack(hmaps)
    return H, paths


def load_model(model_name, num_classes, ckpt_path, device):
    """Lädt dein CNN aus dem Checkpoint."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {ckpt_path}")

    blob = torch.load(ckpt_path, map_location="cpu")
    if isinstance(blob, dict):
        state = blob.get("state_dict", blob)
    else:
        state = blob

    model = build_model(model_name, num_classes=num_classes)
    clean_state = {
        (k.replace("module.", "") if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    model.load_state_dict(clean_state, strict=False)
    model.to(device).eval()
    return model


def map_method_to_quantus(method: str):
    return "IntegratedGradients", {"n_steps": 50}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--method",
        required=True,
        choices=["gradcam", "gradcam++", "ig", "lime"],
    )
    ap.add_argument("--data_dir", default="data/ecg_test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = configs.IMG_SIZE
    num_classes = len(configs.CLASSES)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] IMG_SIZE: {img_size}")

    hm_dir = Path("outputs/explanations") / args.model / args.method
    if not hm_dir.exists():
        print(f"[ERROR] Heatmap-Ordner fehlt: {hm_dir}")
        return

    print(f"[INFO] Lade Heatmaps aus {hm_dir} ...")
    H, hm_paths = load_heatmaps(hm_dir, img_size=img_size, limit=args.limit)
    n_h = len(H)
    print(f"[INFO] Anzahl Heatmaps: {n_h}")

    print(f"[INFO] Lade Testbilder aus {args.data_dir} ...")
    X, Y, img_paths = load_images_with_labels(
        args.data_dir, img_size=img_size, limit=args.limit
    )
    n_x = len(X)
    print(f"[INFO] Anzahl Bilder: {n_x}")

    n = min(n_h, n_x)
    if n == 0:
        print("[ERROR] Keine Samples für Quantus.")
        return
    if n_h != n_x:
        print(
            f"[WARN] Heatmaps != Bilder ({n_h} vs {n_x}). "
            f"Nutze nur die ersten {n} Paare."
        )

    X = X[:n]
    Y = Y[:n]
    H = H[:n]

    ckpt_path = f"outputs/checkpoints/{args.model}_best.pt"
    print(f"[INFO] Lade Modell: {ckpt_path}")
    model = load_model(
        args.model, num_classes=num_classes, ckpt_path=ckpt_path, device=device
    )

    quantus_method_name, extra_explain_kwargs = map_method_to_quantus(args.method)
    explain_kwargs = {
        "method": quantus_method_name,
        "normalise": True,
        "abs": False,
        "batch_size": args.batch_size,
        "device": device,
    }
    explain_kwargs.update(extra_explain_kwargs)

    print(f"[INFO] Quantus.explain-Methode: {quantus_method_name}")
    print(f"[INFO] explain_kwargs: {explain_kwargs}")

    metrics = {
        "faithfulness_corr": quantus.FaithfulnessCorrelation(
            nr_runs=10,
            subset_size=50,
            disable_warnings=True,
        ),
        "max_sens": quantus.MaxSensitivity(
            nr_samples=10,
            lower_bound=0.2,
            norm_numerator=quantus.norm_func.fro_norm,
            norm_denominator=quantus.norm_func.fro_norm,
            similarity_func=quantus.similarity_func.difference,
            abs=True,
            normalise=True,
            disable_warnings=True,
        ),
    }

    ENABLE_MPRT_MODELS = {"resnet18", "resnet34", "resnet50"}

    if args.model in ENABLE_MPRT_MODELS and args.method != "lime":
        metrics["mprt"] = quantus.MPRT(
            return_average_correlation=True,
            disable_warnings=True,
        )

    scores = {}
    for name, metric in metrics.items():
        print(f"[INFO] Running metric: {name} ...")
        vals = metric(
            model=model,
            x_batch=X,
            y_batch=Y,
            a_batch=H,
            device=device,
            explain_func=quantus.explain,
            explain_func_kwargs=explain_kwargs,
        )
        if isinstance(vals, dict):
            vals = list(vals.values())
        scores[name] = float(np.mean(vals))
        print(f"[INFO] {name}: mean = {scores[name]:.6f}")

    if "mprt" not in scores:
        scores["mprt"] = float("nan")

    out_dir = Path("outputs/metrics/quantus_raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"quantus_{args.model}_{args.method}.csv"

    df = pd.DataFrame(
        [
            {
                "model": args.model,
                "method": args.method,
                "faithfulness_corr": scores["faithfulness_corr"],
                "max_sens": scores["max_sens"],
                "mprt": scores["mprt"],
            }
        ]
    )
    df.to_csv(out_file, index=False)

    print("\n[RESULTS]")
    print(df)
    print(f"[OK] Quantus-Ergebnisse gespeichert in: {out_file}")


if __name__ == "__main__":
    main()
