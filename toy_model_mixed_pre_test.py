import torch
from torch import nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print(f"fc1 output dtype: {x.dtype}")
        x = self.ln(x)
        print(f"LayerNorm output dtype: {x.dtype}")
        x = self.fc2(x)
        return x


def print_param_precisions(model: nn.Module, title: str) -> None:
    print(title)
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")


def main() -> None:
    device = torch.device("cuda")

    model = ToyModel(in_features=8, out_features=4).to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(32, 8, device=device, dtype=torch.float32)
    target = torch.randn(32, 4, device=device, dtype=torch.float32)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
        print_param_precisions(model, "Parameter precision within autocast:")
        print(f"Logits dtype: {logits.dtype}")
        loss = nn.functional.mse_loss(logits, target)
        print(f"Loss dtype: {loss.dtype}")

    loss.backward()
    print(f"Gradient dtype for fc1 weight: {model.fc1.weight.grad.dtype}")
    optimizer.step()


if __name__ == "__main__":
    main()


