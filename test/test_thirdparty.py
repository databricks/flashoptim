"""Test cast_model + forward + backward + FlashAdamW step on third-party models.

Run explicitly with::

    python -m pytest test/test_thirdparty.py -m thirdparty
"""

import gc

import pytest
import torch
from _pytest.mark.structures import ParameterSet

from flashoptim import FlashAdamW, cast_model

DEVICE = "cuda"
DTYPE = torch.bfloat16


# -- Helpers -------------------------------------------------------------------


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def _img(b, c, h, w):
    return torch.randn(b, c, h, w, device=DEVICE, dtype=DTYPE)


def _tok(b, s):
    return torch.randint(0, 1000, (b, s), device=DEVICE)


def _extract_loss(out):
    """Get a scalar loss from various model output formats."""
    if isinstance(out, dict):
        if "logits" in out:
            return out["logits"].float().sum()
        if "loss" in out and out["loss"] is not None:
            return out["loss"].float()
        tensors = [
            v
            for v in out.values()
            if isinstance(v, torch.Tensor) and v.is_floating_point()
        ]
        if tensors:
            return sum(t.float().sum() for t in tensors)
    if isinstance(out, (tuple, list)):
        first = out[0]
        if isinstance(first, torch.Tensor):
            return first.float().sum()
        if isinstance(first, dict):
            return sum(
                v.float().sum() for v in first.values() if isinstance(v, torch.Tensor)
            )
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state.float().sum()
    if hasattr(out, "sample"):
        return out.sample.float().sum()
    return out.float().sum()


def _run_test(build_fn, input_fn, forward_fn=None):
    """Build, cast, forward, backward, optimizer step."""
    model = build_fn().to(DEVICE)
    cast_model(model, dtype=DTYPE)

    inp = input_fn()
    if forward_fn is not None:
        out = forward_fn(model, inp)
    elif isinstance(inp, dict):
        out = model(**inp)
    else:
        out = model(inp)

    loss = _extract_loss(out)
    loss.backward()

    optimizer = FlashAdamW(model.parameters(), lr=1e-4, master_weight_bits=24)
    optimizer.step()
    optimizer.zero_grad()

    del model, optimizer


# -- Model registries ----------------------------------------------------------

MODELS: list[pytest.param] = []


def _add(library: str, entries: list):
    """Append model entries for a library, handling both tuples and pytest.param."""
    for entry in entries:
        if isinstance(entry, ParameterSet):
            MODELS.append(
                pytest.param(
                    library,
                    *entry.values,
                    marks=entry.marks,
                    id=f"{library}/{entry.values[0]}",
                )
            )
        else:
            name, build_fn, input_fn, forward_fn = entry
            MODELS.append(
                pytest.param(
                    library,
                    name,
                    build_fn,
                    input_fn,
                    forward_fn,
                    id=f"{library}/{name}",
                )
            )


# --- timm ---


def _register_timm():
    import timm

    names = [
        "resnet50",  # BatchNorm int buffers (num_batches_tracked)
        "vit_base_patch16_224",  # position embeddings, multi-head attention
        "swin_tiny_patch4_window7_224",  # relative_position_index int buffer (original bug)
        "mixer_b16_224",  # pure MLP — no conv, no attention
    ]
    _add(
        "timm",
        [
            (
                n,
                lambda _n=n: timm.create_model(_n, pretrained=False, num_classes=10),
                lambda: _img(2, 3, 224, 224),
                None,
            )
            for n in names
        ],
    )


# --- transformers ---


def _register_transformers():
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        CLIPTextModel,
        SamModel,
        WhisperForConditionalGeneration,
    )

    def _build(model_id, cls, **config_overrides):
        def fn():
            cfg = AutoConfig.from_pretrained(model_id)
            for k, v in config_overrides.items():
                setattr(cfg, k, v)
            return cls.from_config(cfg)

        return fn

    def _enc_input():
        return {
            "input_ids": _tok(2, 64),
            "attention_mask": torch.ones(2, 64, device=DEVICE, dtype=torch.long),
        }

    def _dec_input():
        return {"input_ids": _tok(2, 64)}

    def _build_whisper():
        cfg = AutoConfig.from_pretrained("openai/whisper-tiny")
        return WhisperForConditionalGeneration(cfg)

    def _whisper_input():
        cfg = AutoConfig.from_pretrained("openai/whisper-tiny")
        return {
            "input_features": torch.randn(2, 80, 3000, device=DEVICE, dtype=DTYPE),
            "decoder_input_ids": torch.tensor(
                [[cfg.decoder_start_token_id]] * 2, device=DEVICE
            ),
        }

    def _build_dinov2():
        cfg = AutoConfig.from_pretrained("facebook/dinov2-small")
        return AutoModel.from_config(cfg)

    def _dinov2_input():
        return {"pixel_values": _img(2, 3, 224, 224)}

    def _build_sam():
        cfg = AutoConfig.from_pretrained("facebook/sam-vit-base")
        return SamModel(cfg)

    def _sam_input():
        return {
            "pixel_values": _img(1, 3, 1024, 1024),
            "input_points": torch.tensor(
                [[[[512.0, 512.0]]]], device=DEVICE, dtype=DTYPE
            ),
            "input_labels": torch.tensor([[[1]]], device=DEVICE, dtype=torch.long),
        }

    def _build_qwen25():
        cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
        cfg.dtype = "float32"
        return AutoModelForCausalLM.from_config(cfg)

    def _build_clip_text():
        cfg = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")
        return CLIPTextModel(cfg.text_config)

    def _clip_text_input():
        return {"input_ids": torch.randint(0, 49408, (2, 77), device=DEVICE)}

    def _build_qwen35():
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B")
        # Trim to one DeltaNet cycle (3 linear_attention + 1 full_attention)
        cfg.text_config.num_hidden_layers = 4
        cfg.text_config.layer_types = cfg.text_config.layer_types[:4]
        cfg.vision_config.depth = 2
        return AutoModel.from_config(cfg)

    _add(
        "transformers",
        [
            # Encoder + position_ids/token_type_ids int buffers
            (
                "bert-base-uncased",
                _build("bert-base-uncased", AutoModel),
                _enc_input,
                None,
            ),
            # Decoder-only causal LM
            ("gpt2", _build("gpt2", AutoModelForCausalLM), _dec_input, None),
            # GemmaRMSNorm, rotary embeddings (trimmed: 26 → 2 layers)
            (
                "google/gemma-2-2b",
                _build(
                    "google/gemma-2-2b",
                    AutoModelForCausalLM,
                    num_hidden_layers=2,
                ),
                _dec_input,
                None,
            ),
            # Custom dtype config
            ("Qwen/Qwen2.5-0.5B", _build_qwen25, _dec_input, None),
            # Encoder-decoder + audio modality
            ("openai/whisper-tiny", _build_whisper, _whisper_input, None),
            # Vision transformer via AutoModel
            ("facebook/dinov2-small", _build_dinov2, _dinov2_input, None),
            # Multi-modal, mixed-dtype inputs (float points + int labels)
            ("facebook/sam-vit-base", _build_sam, _sam_input, None),
            # Text-only CLIP encoder
            ("CLIPTextModel", _build_clip_text, _clip_text_input, None),
            # Pure SSM — no attention (selective state spaces)
            (
                "state-spaces/mamba-130m-hf",
                _build("state-spaces/mamba-130m-hf", AutoModelForCausalLM),
                _dec_input,
                None,
            ),
            # Gated DeltaNet hybrid: 3:1 linear/full attention (trimmed: 24 → 4 layers)
            ("Qwen/Qwen3.5-0.8B", _build_qwen35, _dec_input, None),
            # MoE expert routing + gating (trimmed: 48 → 2 layers, 128 → 8 experts)
            (
                "Qwen/Qwen3-30B-A3B",
                _build(
                    "Qwen/Qwen3-30B-A3B",
                    AutoModelForCausalLM,
                    num_hidden_layers=2,
                    num_experts=8,
                    num_experts_per_tok=2,
                ),
                _dec_input,
                None,
            ),
        ],
    )


# --- diffusers ---


def _register_diffusers():
    from diffusers import UNet2DConditionModel

    _add(
        "diffusers",
        [
            # Cross-attention + timestep conditioning
            (
                "UNet2DConditionModel",
                lambda: UNet2DConditionModel(
                    sample_size=32,
                    in_channels=4,
                    out_channels=4,
                    layers_per_block=1,
                    block_out_channels=(32, 64, 64, 64),
                    cross_attention_dim=64,
                ),
                lambda: {
                    "sample": _img(2, 4, 32, 32),
                    "timestep": torch.tensor([10, 20], device=DEVICE),
                    "encoder_hidden_states": torch.randn(
                        2, 8, 64, device=DEVICE, dtype=DTYPE
                    ),
                },
                None,
            ),
        ],
    )


# --- open_clip ---


def _register_open_clip():
    import open_clip

    def _build(arch):
        def fn():
            model, _, _ = open_clip.create_model_and_transforms(arch, pretrained="")
            return model

        return fn

    def _clip_forward(mdl, inp):
        mdl.train()
        return mdl(inp["image"], inp["text"])

    def _clip_input():
        return {
            "image": _img(2, 3, 224, 224),
            "text": torch.randint(0, 49408, (2, 77), device=DEVICE),
        }

    _add(
        "open_clip",
        # Multimodal image+text, custom forward
        [("ViT-B-32", _build("ViT-B-32"), _clip_input, _clip_forward)],
    )


# -- Register all models at import time ----------------------------------------

_REGISTRATIONS = [
    _register_timm,
    _register_transformers,
    _register_diffusers,
    _register_open_clip,
]

for _reg in _REGISTRATIONS:
    _reg()


# -- Test function -------------------------------------------------------------


@pytest.mark.thirdparty
@pytest.mark.gpu
@pytest.mark.parametrize("library, model_name, build_fn, input_fn, forward_fn", MODELS)
def test_cast_model_thirdparty(library, model_name, build_fn, input_fn, forward_fn):
    try:
        _run_test(build_fn, input_fn, forward_fn)
    finally:
        _cleanup()
