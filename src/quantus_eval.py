# veraltet

import os, argparse, numpy as np, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src import configs
from src.model_def import build_model


TFM = transforms.Compose(
    [
        transforms.Resize((configs.IMG_SIZE, configs.IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def build_loader(data_dir: str, batch: int, num_workers: int, limit: int = None):
    ds = datasets.ImageFolder(data_dir, TFM)
    if limit is not None and limit < len(ds):
        ds = torch.utils.data.Subset(ds, list(range(limit)))
    loader = DataLoader(
        ds, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return ds, loader


def load_model(model_name: str, device: str):
    ckpt_path = f"outputs/checkpoints/{model_name}_best.pt"
    blob = torch.load(ckpt_path, map_location="cpu")

    if isinstance(blob, dict):
        classes = list(blob.get("classes", configs.CLASSES))
        state = blob["state_dict"]
    else:
        classes = list(configs.CLASSES)
        state = blob

    model = build_model(model_name, num_classes=len(classes))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model, classes


def make_gradcam_explainer(model):
    import torch.nn as nn

    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m

    class GradCAM:
        def __init__(self, model, target):
            self.model = model
            self.target = target
            self.acts = None
            self.grads = None

            def fwd_hook(_, __, out):
                self.acts = out.detach()

            def bwd_hook(_, grad_in, grad_out):
                self.grads = grad_out[0].detach()

            self.h1 = target.register_forward_hook(fwd_hook)
            self.h2 = target.register_backward_hook(bwd_hook)

        def __call__(self, x):
            logits = self.model(x)
            cls = logits.argmax(1)
            self.model.zero_grad()
            picked = logits.gather(1, cls.view(-1, 1)).sum()
            picked.backward()

            w = self.grads.mean(dim=(2, 3), keepdim=True)
            cam = (w * self.acts).sum(dim=1, keepdim=True).relu()

            B = cam.size(0)
            for i in range(B):
                c = cam[i, 0]
                c -= c.min()
                if c.max() > 0:
                    c /= c.max()
                cam[i, 0] = c

            return cam

    return GradCAM(model, last_conv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/ecg_test")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--method", type=str, default="gradcam", choices=["gradcam"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--limit", type=int, default=144)
    ap.add_argument("--out_dir", type=str, default="outputs/quantus")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model, classes = load_model(args.model, args.device)
    explainer = make_gradcam_explainer(model)

    ds, loader = build_loader(
        args.data_dir, args.batch, args.num_workers, limit=args.limit
    )

    Xs, Ys, Hs = [], [], []

    for x, y in loader:
        x = x.to(args.device)
        x.requires_grad_(True)
        h = explainer(x)
        Xs.append(x.detach().cpu())
        Ys.append(y.cpu())
        Hs.append(h.detach().cpu())

    X = torch.cat(Xs).numpy()
    Y = torch.cat(Ys).numpy()
    H = torch.cat(Hs).numpy()[:, 0, :, :]

    from quantus import (
        FaithfulnessCorrelation,
        MonotonicityCorrelation,
        MaxSensitivity,
        ModelParameterRandomisation,
    )

    metrics = {
        "faithfulness": FaithfulnessCorrelation(similarity_func="spearmanr"),
        "monotonicity": MonotonicityCorrelation(similarity_func="spearmanr"),
        "max_sens": MaxSensitivity(nr_samples=10, perturbations_kwargs={"std": 0.1}),
        "model_rand": ModelParameterRandomisation(nr_models=3, subset_ratio=0.5),
    }

    scores = {}
    for name, metric in metrics.items():
        vals = metric(model=model, x_batch=X, y_batch=Y, a_batch=H, device=args.device)
        scores[name] = float(np.mean(vals))

    import pandas as pd

    df = pd.DataFrame([{"model": args.model, "method": args.method, **scores}])
    out_csv = os.path.join(args.out_dir, f"quantus_{args.model}.csv")
    df.to_csv(out_csv, index=False)

    print(df)
    print(f"[OK] saved → {out_csv}")


if __name__ == "__main__":
    main()
