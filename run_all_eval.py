# run_all_eval.py
import os

models = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "densenet121",
    "densenet169",
    "vgg16_bn",
    "efficientnet_b0",
    "efficientnet_b1",
    "mobilenet_v2",
]

DATA_DIR = "data/ecg_test"

for m in models:
    ckpt = f"outputs/checkpoints/{m}_best.pt"
    cmd = (
        f"python -m src.eval "
        f"--data_dir {DATA_DIR} "
        f"--ckpt {ckpt}"
    )
    print(f"\n=== EVAL {m} ===")
    os.system(cmd)
