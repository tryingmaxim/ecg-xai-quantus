# starten mit: python run_all_quantus.py
# führt Quantus-Evaluationen für alle Modelle und XAI-Methoden durch, außer ResNet101, weil zu hohe Laufzeit
import os

models = [
    "resnet18",
    "resnet34",
    "resnet50",
    # "resnet101",
    "densenet121",
    "densenet169",
    "vgg16_bn",
    "efficientnet_b0",
    "efficientnet_b1",
    "mobilenet_v2",
]

methods = ["gradcam", "gradcam++", "ig", "lime"] 

LIMIT = 144
BATCH_SIZE = 4

for m in models:
    for method in methods:
        cmd = (
            f"python -m src.quantus_from_heatmaps "
            f"--model {m} "
            f"--method {method} "
            f"--data_dir data/ecg_test_flat "
            f"--limit {LIMIT} "
            f"--batch_size {BATCH_SIZE}"
        )
        print("\n=== RUN:", cmd)
        os.system(cmd)
