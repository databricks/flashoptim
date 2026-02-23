"""Train a CNN on MNIST using FlashOptim.

This example demonstrates:
  - ``cast_model`` to convert a model to bf16 (keeping normalization in fp32)
  - ``FlashAdamW`` as a drop-in replacement for ``torch.optim.AdamW``
  - ``master_weight_bits`` precision control
  - ``enable_gradient_release`` for reduced peak memory
  - ``get_fp32_model_state_dict`` / ``set_fp32_model_state_dict`` for
    full-precision checkpoint round-tripping

Usage::

    python train_mnist.py                         # default: 24-bit master weights
    python train_mnist.py --master-weight-bits 32  # full fp32 master weights
    python train_mnist.py --gradient-release       # enable gradient release

Requirements: ``torch >= 2.7`` and ``flashoptim``.  No other dependencies beyond stdlib.
"""

import argparse
import gzip
import struct
import tempfile
import urllib.request
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashoptim import FlashAdamW, cast_model, enable_gradient_release

# ── MNIST data loading (no torchvision dependency) ──────────────────────

_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(filename: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / filename
    if not path.exists():
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(f"{_MIRROR}/{filename}", path)
    return path


def _read_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(
        n, 1, rows, cols
    )


def _read_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        f.read(8)  # skip header
        data = f.read()
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).long()


def load_mnist(
    data_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Download MNIST (if needed) and return normalised tensors on *device*."""
    paths = {k: _download(v, data_dir) for k, v in _FILES.items()}
    train_x = _read_images(paths["train_images"]).to(device, dtype) / 255.0
    train_y = _read_labels(paths["train_labels"]).to(device)
    test_x = _read_images(paths["test_images"]).to(device, dtype) / 255.0
    test_y = _read_labels(paths["test_labels"]).to(device)
    return train_x, train_y, test_x, test_y


# ── Model ───────────────────────────────────────────────────────────────


class MNISTNet(nn.Module):
    """Small CNN: two conv blocks followed by two linear layers."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ── Training / evaluation ──────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> float:
    model.train()
    n = images.size(0)
    perm = torch.randperm(n, device=images.device)
    images, labels = images[perm], labels[perm]

    total_loss = 0.0
    num_batches = 0
    for i in range(0, n, batch_size):
        logits = model(images[i : i + batch_size])
        loss = F.cross_entropy(logits, labels[i : i + batch_size])
        loss.backward()
        # step() and zero_grad() are intentional no-ops when gradient release
        # is active — parameters are updated by backward hooks instead.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> float:
    model.eval()
    correct = 0
    n = images.size(0)
    for i in range(0, n, batch_size):
        preds = model(images[i : i + batch_size]).argmax(1)
        correct += (preds == labels[i : i + batch_size]).sum().item()
    return correct / n


# ── Checkpoint round-trip ───────────────────────────────────────────────


def verify_checkpoint_roundtrip(
    model: nn.Module,
    optimizer: FlashAdamW,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    batch_size: int,
) -> None:
    """Save a full-precision checkpoint, reload it, and verify correctness."""
    acc_before = evaluate(model, test_x, test_y, batch_size)

    # Export fp32 state dict and save to disk
    fp32_sd = optimizer.get_fp32_model_state_dict(model)

    # Verify all exported tensors are fp32
    dtypes = {v.dtype for v in fp32_sd.values() if isinstance(v, torch.Tensor)}
    non_fp32 = dtypes - {torch.float32}
    assert not non_fp32, f"Expected all fp32, got {non_fp32}"
    print(f"  Exported {len(fp32_sd)} tensors, all fp32.")

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(fp32_sd, f.name)
        loaded_sd = torch.load(f.name, weights_only=True)

    # Reload into the model (recomputes error-correction bits from fp32)
    optimizer.set_fp32_model_state_dict(model, loaded_sd)

    acc_after = evaluate(model, test_x, test_y, batch_size)
    print(f"  Accuracy before save: {acc_before:.2%}")
    print(f"  Accuracy after load:  {acc_after:.2%}")
    assert acc_before == acc_after, (
        f"Accuracy changed after checkpoint reload: {acc_before} -> {acc_after}"
    )


# ── CLI ─────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    """Train a small CNN on MNIST with FlashOptim."""
    device = torch.device("cuda")
    dtype = torch.bfloat16

    epochs: int = args.epochs
    batch_size: int = args.batch_size
    lr: float = args.lr
    data_dir = Path(args.data_dir)
    gradient_release: bool = args.gradient_release

    # Treat 0 as None
    mwb: int | None = None if args.master_weight_bits == 0 else args.master_weight_bits

    # ── Data ────────────────────────────────────────────────────────────
    print("Loading MNIST...")
    train_x, train_y, test_x, test_y = load_mnist(data_dir, device, dtype)

    # ── Model ───────────────────────────────────────────────────────────
    model = MNISTNet().to(device)
    # cast_model keeps BatchNorm layers in fp32 automatically
    cast_model(model, dtype=dtype)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # ── Optimizer ───────────────────────────────────────────────────────
    optimizer = FlashAdamW(model.parameters(), lr=lr, master_weight_bits=mwb)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Optimizer: FlashAdamW  master_weight_bits={mwb}  lr={lr}")
    print(f"Schedule:  CosineAnnealing  T_max={epochs}")

    # ── Gradient release (optional) ─────────────────────────────────────
    gr_handle = None
    if gradient_release:
        gr_handle = enable_gradient_release(model, optimizer)
        print("Gradient release: enabled")
    print()

    # ── Training loop ───────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, optimizer, train_x, train_y, batch_size)
        scheduler.step()
        acc = evaluate(model, test_x, test_y, batch_size)
        cur_lr = scheduler.get_last_lr()[0]
        print(
            f"  Epoch {epoch:>2}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}  lr={cur_lr:.2e}"
        )

    # ── Final evaluation ────────────────────────────────────────────────
    final_acc = evaluate(model, test_x, test_y, batch_size)
    print(f"\nTest accuracy: {final_acc:.2%}")

    # ── Checkpoint round-trip demo ──────────────────────────────────────
    if mwb is not None:
        print("\nVerifying checkpoint round-trip...")
        if gr_handle is not None:
            gr_handle.remove()
        verify_checkpoint_roundtrip(model, optimizer, test_x, test_y, batch_size)

    if gr_handle is not None and gr_handle.active:
        gr_handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a small CNN on MNIST with FlashOptim."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate.")
    parser.add_argument(
        "--master-weight-bits",
        type=int,
        default=24,
        help="Master weight precision: 24, 32, or 0 for None.",
    )
    parser.add_argument(
        "--gradient-release", action="store_true", help="Enable gradient release."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/mnist",
        help="Directory to cache MNIST data.",
    )
    main(parser.parse_args())
