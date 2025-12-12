# starten mit: python run_all_xai.py
# führt OmniXAI-Erklärungen für alle Modelle und XAI-Methoden durch
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

METHODS = ["gradcam", "gradcam++", "ig", "lime"]

DATA_DIR = "data/ecg_test"
LIMIT = 144

for m in models:
    ckpt = f"outputs/checkpoints/{m}_best.pt"
    for method in METHODS:
        cmd = (
            f"python -m src.explain_omnixai "
            f"--ckpt {ckpt} "
            f"--method {method} "
            f"--data_dir {DATA_DIR} "
            f"--limit {LIMIT}"
        )
        print("\n===== RUNNING:", cmd)
        os.system(cmd)
