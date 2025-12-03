# run_all_quantus.py
import os

models = [
    "resnet18",
    "resnet34",
    "resnet50",
    #"resnet101",
    "densenet121",
    "densenet169",
    "vgg16_bn",
    "efficientnet_b0",
    "efficientnet_b1",
    "mobilenet_v2",
]

methods = ["gradcam", "gradcam++", "ig"]

LIMIT = 10        # 10 Samples pro Modell/Methode
BATCH_SIZE = 32   # GPU-freundlich

for m in models:
    for method in methods:
        cmd = (
            f"python -m src.quantus_from_heatmaps "
            f"--model {m} "
            f"--method {method} "
            f"--data_dir data/ecg_test "
            f"--limit {LIMIT} "
            f"--batch_size {BATCH_SIZE}"
        )
        print("\n=== RUN:", cmd)
        os.system(cmd)
