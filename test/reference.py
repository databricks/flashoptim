"""Reference optimizer implementations used only in tests.

ReferenceAdamW uses standard decoupled weight decay by default:
    p *= (1 - lr * weight_decay)

When decouple_lr=True, uses fully LR-decoupled weight decay:
    decay_factor = lr / initial_lr
    p *= (1 - decay_factor * weight_decay)

ReferenceLion uses standard decoupled weight decay by default:
    p *= (1 - lr * weight_decay)

When decouple_lr=True, uses fully LR-decoupled weight decay:
    decay_factor = lr / initial_lr
    p *= (1 - decay_factor * weight_decay)

ReferenceSGD uses coupled weight decay (L2 regularization) to match FlashSGD.

ReferenceSGDW uses standard decoupled weight decay by default:
    p *= (1 - lr * weight_decay)

When decouple_lr=True, uses fully LR-decoupled weight decay:
    decay_factor = lr / initial_lr
    p *= (1 - decay_factor * weight_decay)
"""

import math

import torch
from torch import optim


class ReferenceAdamW(optim.Optimizer):
    """AdamW with decoupled weight decay.

    By default uses standard decoupled weight decay:
        p *= (1 - lr * weight_decay)

    When decouple_lr=True, uses fully LR-decoupled weight decay:
        p *= (1 - (lr / initial_lr) * weight_decay)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decouple_lr: bool = False,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        self._decouple_lr = decouple_lr
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in filter(
                lambda p: p.grad is not None and p.requires_grad,
                group["params"],
            ):
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                if group["weight_decay"] != 0:
                    if self._decouple_lr:
                        decay_factor = group["lr"] / group["initial_lr"]
                    else:
                        decay_factor = group["lr"]
                    p.mul_(1 - decay_factor * group["weight_decay"])

                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                step_size = group["lr"] / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])
                p.addcdiv_(exp_avg, denom, value=-step_size)


class ReferenceSGD(optim.Optimizer):
    """SGD with momentum and L2 regularisation (coupled weight decay).

    Weight decay is applied as: grad += weight_decay * param
    This matches torch.optim.SGD and FlashSGD.

    For decoupled weight decay, use :class:`ReferenceSGDW`.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        self._decouple_lr = False
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "decoupled": False,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in filter(
                lambda p: p.grad is not None and p.requires_grad,
                group["params"],
            ):
                state = self.state[p]

                if group["decoupled"]:
                    if group["weight_decay"] != 0:
                        if self._decouple_lr:
                            decay_factor = group["lr"] / group["initial_lr"]
                        else:
                            decay_factor = group["lr"]
                        p.mul_(1 - decay_factor * group["weight_decay"])

                    if group["momentum"] != 0:
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        buf = state["momentum_buffer"]
                        buf.mul_(group["momentum"]).add_(p.grad)
                        if group["nesterov"]:
                            p.add_(p.grad + buf * group["momentum"], alpha=-group["lr"])
                        else:
                            p.add_(buf, alpha=-group["lr"])
                    else:
                        p.add_(p.grad, alpha=-group["lr"])
                else:
                    grad = p.grad
                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])

                    if group["momentum"] != 0:
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        buf = state["momentum_buffer"]
                        buf.mul_(group["momentum"]).add_(grad)
                        if group["nesterov"]:
                            p.add_(grad + buf * group["momentum"], alpha=-group["lr"])
                        else:
                            p.add_(buf, alpha=-group["lr"])
                    else:
                        p.add_(grad, alpha=-group["lr"])


class ReferenceSGDW(ReferenceSGD):
    """SGD with momentum and decoupled weight decay.

    By default uses standard decoupled weight decay:
        p *= (1 - lr * weight_decay)

    When decouple_lr=True, uses fully LR-decoupled weight decay:
        p *= (1 - (lr/initial_lr) * weight_decay)
    """

    def __init__(
        self,
        params,
        *,
        weight_decay: float = 0.0,
        decouple_lr: bool = False,
        **kwargs,
    ):
        super().__init__(params, weight_decay=weight_decay, **kwargs)
        self._decouple_lr = decouple_lr
        for group in self.param_groups:
            group["decoupled"] = True


class ReferenceLion(optim.Optimizer):
    """Reference Lion optimizer implementation with decoupled weight decay."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        decouple_lr: bool = False,
    ):
        self._decouple_lr = decouple_lr
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]

    @staticmethod
    def lionw(
        p: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        lr: float,
        initial_lr: float,
        wd: float,
        beta1: float,
        beta2: float,
        decouple_lr: bool = False,
    ) -> None:
        if wd != 0:
            if decouple_lr:
                decay_factor = (lr / initial_lr) if initial_lr else 1.0
            else:
                decay_factor = lr
            p.data.mul_(1 - decay_factor * wd)
        update = exp_avg.lerp(grad, 1 - beta1).sign_()
        p.add_(update, alpha=-lr)
        exp_avg.lerp_(grad, 1 - beta2)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in filter(
                lambda p: p.grad is not None and p.requires_grad,
                group["params"],
            ):
                grad, lr, initial_lr, wd, beta1, beta2, state = (
                    p.grad,
                    group["lr"],
                    group["initial_lr"],
                    group["weight_decay"],
                    *group["betas"],
                    self.state[p],
                )

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                self.lionw(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    initial_lr,
                    wd,
                    beta1,
                    beta2,
                    self._decouple_lr,
                )
