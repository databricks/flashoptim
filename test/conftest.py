# Copyright 2026 Databricks AI Research authors

import os
import tempfile
from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@pytest.fixture(autouse=True, scope="session")
def _allow_ineffective_master_weight_bits():
    os.environ["FLASHOPTIM_ALLOW_INEFFECTIVE_MASTER_WEIGHT_BITS"] = "1"
    yield
    os.environ.pop("FLASHOPTIM_ALLOW_INEFFECTIVE_MASTER_WEIGHT_BITS", None)


def pytest_collection_modifyitems(config, items):
    """Auto-deselect thirdparty tests unless explicitly requested via -m."""
    marker_expr = config.getoption("-m", default="")
    if "thirdparty" in marker_expr:
        return
    remaining = []
    deselected = []
    for item in items:
        if "thirdparty" in item.keywords:
            deselected.append(item)
        else:
            remaining.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = remaining


def pytest_sessionstart(session):
    """Check for CUDA availability before running any tests."""
    if not torch.cuda.is_available():
        pytest.exit(
            "CUDA is not available. The flashoptim test suite requires CUDA to run.",
            returncode=1,
        )


# ============================================================================
# DDP Test Infrastructure
# ============================================================================


def _run_ddp_test(
    rank: int,
    world_size: int,
    init_file: str,
    test_fn: Callable,
    test_args: tuple,
    test_kwargs: dict,
) -> None:
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=init_file,
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank)
        test_fn(rank, world_size, *test_args, **test_kwargs)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture(scope="function")
def ddp_world_size():
    """Configure world size for DDP tests."""
    return 2


@pytest.fixture(scope="function")
def ddp_runner(ddp_world_size):
    """Provide a function to run tests with DDP using mp.spawn.

    Usage::

        def test_something(ddp_runner):
            def _test_impl(rank, world_size):
                ...

            ddp_runner(_test_impl)
    """

    def runner(test_fn: Callable, *args, **kwargs) -> None:
        fd, init_file_path = tempfile.mkstemp()
        os.close(fd)

        try:
            init_file = f"file://{init_file_path}"
            mp.spawn(
                _run_ddp_test,
                args=(ddp_world_size, init_file, test_fn, args, kwargs),
                nprocs=ddp_world_size,
                join=True,
            )
        finally:
            if os.path.exists(init_file_path):
                os.unlink(init_file_path)

    yield runner


# FSDP2 tests reuse the same mp.spawn-based distributed testing infrastructure
fsdp2_runner = ddp_runner
