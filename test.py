import torch


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


print(f"[debug] torch version: {torch.__version__}")
print(f"[debug] cuda available: {torch.cuda.is_available()}")
print(f"[debug] mps backend exists: {hasattr(torch.backends, 'mps')}")
if hasattr(torch.backends, "mps"):
    print(f"[debug] mps built: {torch.backends.mps.is_built()}")
    print(f"[debug] mps available: {torch.backends.mps.is_available()}")

device = resolve_device()
print(f"[debug] selected device: {device}")

x = torch.ones(5, device=device)
y = x * 2
print(f"张量 y 的内容: {y}")
print(f"张量 y 的设备: {y.device}")
