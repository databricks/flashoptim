<img src="https://raw.githubusercontent.com/databricks/flashoptim/refs/heads/assets/imgs/flashoptim_ithaca_italic.png" alt="FlashOptim" width="650">

This is the official implementation of [FlashOptim: Optimizers for Memory Efficient Training](https://arxiv.org/abs/2602.23349)

By [Jose Javier Gonzalez Ortiz](https://x.com/jjgort), [Abhay Gupta](https://x.com/gupta__abhay), [Christopher Rinard](https://x.com/ChrisRinard), and [Davis Blalock](https://x.com/davisblalock).

[![CI](https://img.shields.io/github/actions/workflow/status/databricks/flashoptim/ci.yaml?branch=main&label=ci)](https://github.com/databricks/flashoptim/actions/workflows/ci.yaml)
[![PyPI](https://img.shields.io/pypi/v/flashoptim)](https://pypi.org/project/flashoptim/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D2.7-ee4c2c)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.23349-b31b1b.svg)](https://arxiv.org/abs/2602.23349)

## TL;DR

FlashOptim is a library implementing drop-in replacements for PyTorch optimizers that substantially reduces training memory by **shrinking the footprint of** optimizer states, master weights, and gradients.

For example, for finetuning an 8B model, FlashOptim requires 35% less peak memory and produces checkpoints that are 57% smaller.

<p align="center">
  <img src="https://raw.githubusercontent.com/databricks/flashoptim/refs/heads/assets/imgs/finetuning_memory_breakdown.png" width="48%" alt="Memory breakdown comparing a regular optimizer vs FlashOptim">
  <img src="https://raw.githubusercontent.com/databricks/flashoptim/refs/heads/assets/imgs/convergence_finetuning_adamw.png" width="48%" alt="Convergence comparison between regular AdamW and FlashOptim">
</p>

Despite operating in reduced precision, FlashOptim does not affect model convergence.

## 1. Quickstart

To get started you can install flashoptim:

```shell
$ pip install flashoptim
```

Once installed, you can import `FlashSGD`, `FlashSGDW`, `FlashAdam`, `FlashAdamW` and `FlashLion`, which follow the standard PyTorch optimizer API. For example, to use `FlashAdamW`:

```python
import torch
from torch import nn

from flashoptim import FlashAdamW, cast_model

model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10)).cuda()
# cast parameters to bf16
cast_model(model, dtype=torch.bfloat16)

# master_weight_bits=24 (default) means we have 24-bit parameter semantics
optimizer = FlashAdamW(model.parameters(), lr=1e-3)

x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
loss = model(x).sum()
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

That's it! You are now training with 50% less per-parameter memory! For more details on the API and advanced features, keep reading.

## 2. Key Features

- **Memory Savings**. By splitting the weight representation and quantizing the optimizer states, FlashOptim reduces per-parameter memory (e.g. 57% for Adam) and peak training memory without degrading convergence.
- **Fused Triton Kernels**. All compression operations are fused into the update kernel, introducing no practical overhead.
- **Gradient Release**. Optionally, parameters can be updated as soon as the gradients are computed, further reducing peak memory.
- **Compressed Checkpoints**. Checkpoints can optionally be stored using quantized optimizer states, producing >50% space savings.
- **PyTorch API**. The optimizers follow the standard `torch.optim.Optimizer` interface.

## 3. Installation

FlashOptim can be installed using `pip` or `uv`. Note that FlashOptim is only supported on Linux systems with NVIDIA CUDA GPUs.

```bash
# install stable version
pip install flashoptim

# install latest version from source
pip install git+https://github.com/databricks/flashoptim.git

# or install it locally in editable mode for development
git clone https://github.com/databricks/flashoptim.git
cd flashoptim
pip install -e .
```

## 4. Usage

> [!NOTE]
> The first optimizer step will be slower than subsequent steps due to Triton kernel JIT compilation. This is a one-time cost per kernel configuration.

### Specifying Precision

The `master_weight_bits` parameter controls the width of the master weights maintained by the optimizer. By default, master weights are 24-bit, narrower than fp32, which saves memory. When training in bf16/fp16, the downcasting is fused into the update kernel, so no separate cast step is needed:

```python
from flashoptim import FlashAdamW

# Default: 24-bit master weights (bf16 param + 8-bit correction term)
optimizer = FlashAdamW(model.parameters(), lr=1e-3)

# 32-bit master weights (bf16 param + 16-bit correction term)
optimizer = FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=32)

# No master weight correction; parameters stay at native precision
optimizer = FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=None)
```

The exact behavior depends on the dtype of the parameters passed to the optimizer:

- **bf16/fp16 parameters**: Optimizer states (moments) are quantized to 8-bit. The `master_weight_bits` setting controls master weight precision and fuses the downcasting into the update kernel:
    - `master_weight_bits=24` (default): 8-bit correction terms for 24-bit master weights, narrower than fp32 while preserving convergence
    - `master_weight_bits=32`: 16-bit correction terms for full 32-bit master weight semantics
    - `master_weight_bits=None`: no master weight correction; optimizer states are still quantized, but parameters stay at their native precision
- **fp32 parameters**: Optimizer states (moments) are quantized to 8-bit to reduce memory. Parameters are already full precision, so `master_weight_bits` is not applicable.

To cast a model's parameters and buffers to bf16, use the `cast_model` helper. `cast_model` uses keyword matching against module names to determine which layers to keep in full precision, by default, normalization layers are kept in fp32 for training stability. It also registers forward pre-hooks on fp32 modules to automatically upcast their inputs during the forward pass:

```python
from flashoptim import cast_model

# Cast all parameters to bf16 (normalization layers kept in fp32 by default)
cast_model(model, dtype=torch.bfloat16)

# Keep specific layers (e.g., the output head) in fp32
cast_model(model, dtype=torch.bfloat16, full_precision_keywords=["lm_head", "head"])
```

> [!NOTE]
> Keywords are matched against dot-separated name segments, so `"head"` matches `model.head.weight` but not `model.header.weight`.

### Weight Decay

FlashOptim follows PyTorch's convention of separating L2 regularization from decoupled weight decay via separate classes:

| Optimizer | Weight Decay Style | PyTorch Equivalent |
|-----------|-------------------|-------------------|
| `FlashAdam` | L2 regularization (coupled) | `torch.optim.Adam` |
| `FlashAdamW` | Decoupled | `torch.optim.AdamW` |
| `FlashSGD` | L2 regularization (coupled) | `torch.optim.SGD` |
| `FlashSGDW` | Decoupled | - |
| `FlashLion` | Decoupled | - |

For decoupled optimizers (`FlashAdamW`, `FlashSGDW`, `FlashLion`), weight decay is applied as a multiplicative factor on the parameters, matching PyTorch's `AdamW` semantics:

$$\theta_t \leftarrow \theta_{t-1} \cdot (1 - \eta_t \cdot \lambda)$$

This means `FlashAdamW(params, lr=1e-3, weight_decay=0.01)` is equivalent to `torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)`.

#### Fully LR-Decoupled Weight Decay

Setting `decouple_lr=True` enables **fully LR-decoupled weight decay**, where $\lambda$ is the *absolute* per-step decay rate, scaled only by the LR ratio to track the schedule:

$$\theta_t \leftarrow \theta_{t-1} \cdot \left(1 - \lambda \cdot \frac{\eta_t}{\eta_0}\right)$$

At initialization $\eta_t = \eta_0$, so the effective decay is simply $\lambda$. This means you should use much smaller `weight_decay` values than with PyTorch. For example, if you were using `torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)` (effective decay $10^{-3} \times 0.01 = 10^{-5}$), the equivalent FlashOptim call is `FlashAdamW(params, lr=1e-3, weight_decay=1e-5, decouple_lr=True)`.

The LR-decoupled formulation ensures that weight decay remains stable regardless of learning rate schedule changes. See [Loshchilov & Hutter (2019)](https://arxiv.org/abs/1711.05101) and [Schaipp (2024)](https://fabian-sp.github.io/posts/2024/02/decoupling/) for more details on decoupling LR and WD magnitudes.


### Loading & Saving Models

FlashOptim represents full-precision parameters using two components:

- **Low precision parameters**. These are stored as `nn.Module` tensors.
- **Error correction terms**. These are stored as optimizer state tensors under the `"error_bits"` key in `optimizer.state[param]`.


FlashOptim provides methods for exporting and importing full-precision (FP32) checkpoints. For loading, the model must have been initialized with the desired precision (e.g. via `cast_model`).

```python
import torch
import torchvision

from flashoptim import FlashAdamW, cast_model

model = torchvision.models.resnet18().cuda()
cast_model(model, dtype=torch.bfloat16, full_precision_keywords=["fc"])
optimizer = FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=24)

# ... training ...

# Save: reconstruct fp32 from bf16 + error bits
fp32_state_dict = optimizer.get_fp32_model_state_dict(model)
torch.save(fp32_state_dict, "checkpoint.pt")

# Load: restore fp32 weights into a bf16 model (error bits recomputed automatically)
fp32_state_dict = torch.load("checkpoint.pt", weights_only=True)
optimizer.set_fp32_model_state_dict(model, fp32_state_dict)
```

### Compressed Checkpoints

By default, optimizer state dicts are saved with states cast to bf16, which is already smaller than fp32. For additional savings, set `compress_state_dict=True` when constructing the optimizer to quantize states to int8, producing checkpoints ~50% smaller than bf16:

```python
# Default: state_dict() saves states as bf16
optimizer = FlashAdamW(model.parameters(), lr=1e-3)
torch.save(optimizer.state_dict(), "checkpoint_bf16.pt")

# Compressed: state_dict() saves states as quantized int8
optimizer = FlashAdamW(model.parameters(), lr=1e-3, compress_state_dict=True)
torch.save(optimizer.state_dict(), "checkpoint_int8.pt")
```

> [!NOTE]
> Compressed state dicts are **not** loadable by vanilla PyTorch optimizers. They can only be loaded back by FlashOptim optimizers using `optimizer.load_state_dict()`.

### Distributed Training

FlashOptim is compatible with data parallelism strategies including [DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [FSDP2](https://pytorch.org/docs/stable/distributed.fsdp.html). Wrap or shard your model as usual, then pass the resulting parameters to the optimizer:

> [!WARNING]
> FlashOptim does **not** support FSDP1 (`FullyShardedDataParallel`) due to design limitations in how FSDP1 manages parameter and optimizer state sharding. Please use FSDP2 (`fully_shard`) instead.

```python
# DDP
model = DDP(model, device_ids=[device.index])
optimizer = FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=24)

# FSDP2
fully_shard(model)
optimizer = FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=24)
```

### Gradient Release

FlashOptim supports gradient release, which updates parameters during the backward pass as soon as gradients are computed, further reducing memory usage. Gradient release is implemented via post-backward hooks and needs to be enabled explicitly:

```python
from flashoptim import FlashAdamW, enable_gradient_release

optimizer = FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=24)
handle = enable_gradient_release(model, optimizer)

for x, y in dataloader:
    loss = loss_fn(model(x), y)
    loss.backward()
    # step() and zero_grad() are no-ops while gradient release is active;
    # parameters are updated during backward and gradients are freed immediately
    optimizer.step()
    optimizer.zero_grad()

# Call handle.remove() to restore normal optimizer behavior
handle.remove()
```

Gradient release is compatible with single-GPU training and FSDP2 (`fully_shard`).

**Limitations**. Since the parameters are updated during the backward pass and gradients are freed immediately, gradient release is incompatible with:

- **DDP**. DDP uses custom communication hooks and buffers that cannot be easily instrumented.
- **Microbatch Accumulation**. Gradient release steps parameters immediately as gradients arrive, so gradients cannot be accumulated.
- **Gradient Clipping**. Global gradient clipping (e.g. `torch.nn.utils.clip_grad_norm_`) cannot be applied.
- **Gradient Scaling**. `torch.amp.GradScaler` is not supported with gradient release.

### Numerics Checking

When training in reduced precision, a learning rate that is too small relative to the parameter magnitudes can produce updates that round to zero, silently stalling training.
Setting `check_numerics=True` detects this: at each step FlashOptim verifies that `lr` is large enough to actually change the largest values in every tensor (given the parameter dtype and `master_weight_bits`).
This is useful as a sanity check during early training to catch silent stalls caused by updates that round to zero.

## 5. Compatibility

| Requirement | Details |
|-------------|---------|
| **Hardware** | NVIDIA GPUs with CUDA support  |
| **OS** | Linux |
| **Python** | ≥3.9 |
| **PyTorch** | ≥2.7 |
| **Triton** | ≥2.0 |
| **Distributed** | DDP and FSDP2 supported; FSDP1 **not** supported |
| **Precision** | bf16, fp16, and fp32 parameters |

## 6. Contributing

For contributing to FlashOptim, please see our [contributing guidelines](CONTRIBUTING.md).

## 7. Citation

If you use FlashOptim in your research, please cite our paper:

```bibtex
@article{gonzalezblalock2026flashoptim,
  title={FlashOptim: Optimizers for Memory Efficient Training},
  author={Gonzalez Ortiz, Jose Javier and Gupta, Abhay and Rinard, Christopher and Blalock, Davis},
  journal={arXiv preprint arXiv:2602.23349},
  year={2026}
}
```
