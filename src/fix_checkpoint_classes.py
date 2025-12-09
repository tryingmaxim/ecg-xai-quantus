# Setzt die Klassenreihenfolge im bestehenden Checkpoint auf alphabetische ImageFolder-Order
# (hier: ["Abnormal", "HistoryMI", "MI", "Normal"])

import torch

CKPT = "outputs/checkpoints/best.pt"
NEW_CLASSES = ["Abnormal", "HistoryMI", "MI", "Normal"]

ckpt = torch.load(CKPT, map_location="cpu")
if isinstance(ckpt, dict):
    ckpt["classes"] = list(NEW_CLASSES)
    torch.save(ckpt, CKPT)
    print("[CHECK] Checkpoint-Klassenreihenfolge korrigiert:", NEW_CLASSES)
else:
    torch.save({"state_dict": ckpt, "classes": list(NEW_CLASSES)}, CKPT)
    print(
        "[CHECK] Checkpoint in Dict-Format umgewandelt und Klassen gesetzt:",
        NEW_CLASSES,
    )
