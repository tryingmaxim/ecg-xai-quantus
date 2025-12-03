import torch, torchvision
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda" if torch.cuda.is_available() else "cpu")
y = torch.mm(x, x.t())  # kurze Matmul zum Aufwärmen
print("Test tensor shape:", y.shape)
