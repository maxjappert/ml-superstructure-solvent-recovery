import matplotlib.pyplot as plt
import torch

from config import eps


def plot_training(train_losses, val_losses):
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_dicts_strings(a: dict[str, str], b: dict[str, str],
                            name_a: str = "A", name_b: str = "B") -> str:
    keys = list(a)  # assumes same keys; use a.keys() | b.keys() if not guaranteed
    kw = max(len(k) for k in keys)
    cw = max(len(name_a), len(name_b), 9)  # 9 fits a formatted float

    lines = [f"{'':<{kw}}  {name_a:>{cw}}  {name_b:>{cw}}"]
    for k in keys:
        lines.append(
            f"{k:<{kw}}  {a[k]:>{cw}}  {b[k]:>{cw}}"
        )

    lines.append('\n')
    return "\n".join(lines)

def z_score(tensor: torch.Tensor) -> torch.Tensor:
    t = tensor.float()
    return ((t - t.mean()) / t.std().clamp_min(eps)).to(tensor.dtype)
