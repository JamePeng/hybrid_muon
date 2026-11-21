"""Hybrid Muon/AdamW for dense Qwen-style language models."""

import math

import torch

DEFAULT_ADAMW_LR = 1.5e-5
DEFAULT_ADAMW_WEIGHT_DECAY = 0.01
MUON_CONFIG = {
    "lr": 5e-5,
    "weight_decay": 0.005,
    "momentum": 0.95,
    "scale_mode": "native",
    "update_multiplier": 1.0,
    "paper_scale": 0.2,
    "ns_steps": 5,
}


def zeropower_via_newtonschulz5(matrix, steps):
    """KellerJordan/Muon quintic NS coefficients; local, non-compiled implementation.

    Algorithm reference: https://github.com/KellerJordan/Muon (MIT).
    """
    if matrix.ndim != 2:
        raise ValueError("Muon requires a complete two-dimensional matrix")
    x = matrix.bfloat16()
    transpose = matrix.shape[0] > matrix.shape[1]
    if transpose:
        x = x.mT
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a = x @ x.mT
        b = -4.7750 * a + 2.0315 * a @ a
        x = 3.4445 * x + b @ x
    return x.mT if transpose else x


class HybridMuonAdamW(torch.optim.Optimizer):
    """Whole-matrix Muon plus torch AdamW, with standard unified checkpoints.

    DDP synchronizes gradients; every rank updates all matrices and owns all states.
    No rank-dependent optimizer sharding or parameter all_gather is used here.
    """

    MUON_SUFFIXES = tuple(
        f"{module}.{projection}.weight"
        for module, projections in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("linear_attn", ("in_proj_qkv", "in_proj_z", "out_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        )
        for projection in projections
    )

    @classmethod
    def parameter_group(cls, name, ndim):
        """Classify a trainable parameter by its model name and rank."""
        if ndim == 2 and ".layers." in name and name.endswith(cls.MUON_SUFFIXES):
            return "muon"
        if ndim <= 1 or any(s in name for s in ("norm", "bias", "embed_tokens", "lm_head")):
            return "adamw_no_decay"
        return "adamw_decay"

    def __init__(self, model, muon_config=None, adamw_lr=None, adamw_weight_decay=None):
        cfg = MUON_CONFIG.copy()
        if muon_config is not None:
            unknown = set(muon_config) - cfg.keys()
            if unknown:
                raise ValueError(f"Unknown Muon options: {sorted(unknown)}")
            cfg.update(muon_config)
        if (
            cfg["scale_mode"] not in ("native", "paper")
            or not isinstance(cfg["ns_steps"], int)
            or cfg["ns_steps"] < 1
            or not all(
                math.isfinite(cfg[key])
                for key in (
                    "lr",
                    "weight_decay",
                    "momentum",
                    "update_multiplier",
                    "paper_scale",
                )
            )
            or cfg["update_multiplier"] <= 0
            or cfg["paper_scale"] <= 0
            or not 0 <= cfg["momentum"] < 1
            or cfg["lr"] < 0
            or cfg["weight_decay"] < 0
        ):
            raise ValueError("Invalid Muon configuration")
        buckets = {name: [] for name in ("muon", "adamw_decay", "adamw_no_decay")}
        names = {name: [] for name in buckets}
        for name, param in model.named_parameters():  # tied weights deduplicated
            if not param.requires_grad:
                continue
            group = self.parameter_group(name, param.ndim)
            buckets[group].append(param)
            names[group].append(name)
        if not buckets["muon"]:
            raise ValueError("No trainable two-dimensional projection matched the Muon rules. Check the parameter names.")
        lr = DEFAULT_ADAMW_LR if adamw_lr is None else adamw_lr
        wd = (
            DEFAULT_ADAMW_WEIGHT_DECAY
            if adamw_weight_decay is None
            else adamw_weight_decay
        )
        if not all(math.isfinite(v) and v >= 0 for v in (lr, wd)):
            raise ValueError(
                "AdamW learning rate and weight decay must be finite and nonnegative"
            )
        groups = [
            dict(params=buckets["muon"], name="muon", param_names=names["muon"], **cfg)
        ]
        for name in ("adamw_decay", "adamw_no_decay"):
            if buckets[name]:
                groups.append(
                    {
                        "params": buckets[name],
                        "name": name,
                        "param_names": names[name],
                        "lr": lr,
                        "weight_decay": wd if name == "adamw_decay" else 0.0,
                    }
                )
        super().__init__(groups, {})
        self._bind_adamw()

    def _bind_adamw(self):
        # Share actual group dictionaries: Trainer scheduler changes both LRs.
        groups = [g for g in self.param_groups if g["name"] != "muon"]
        self.adamw = (
            torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8) if groups else None
        )
        if self.adamw is not None:
            self.adamw.state = self.state

    def load_state_dict(self, state_dict):
        # Prevent silent state reassignment when grouping/order changes on resume.
        for old, new in zip(state_dict["param_groups"], self.param_groups):
            if (
                old.get("param_names") != new["param_names"]
                or old.get("name") != new["name"]
            ):
                raise ValueError("Checkpoint parameter groups do not match the current model")
        super().load_state_dict(state_dict)
        self._bind_adamw()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["name"] != "muon":
                continue
            for param in group["params"]:
                if param.ndim != 2:
                    raise RuntimeError("Muon requires full matrices on each rank. Use unsharded parameters.")
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon requires dense gradients")
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                buffer = state["momentum_buffer"]
                buffer.lerp_(grad, 1 - group["momentum"])
                update = zeropower_via_newtonschulz5(
                    grad.lerp(buffer, group["momentum"]), group["ns_steps"]
                )
                if group["scale_mode"] == "native":
                    update = update * max(1, param.shape[0] / param.shape[1]) ** 0.5
                else:
                    update = update.float() * (
                        group["paper_scale"] * max(param.shape) ** 0.5
                    )
                if group["update_multiplier"] != 1:
                    update = update.float() * group["update_multiplier"]
                param.mul_(1 - group["lr"] * group["weight_decay"])
                param.add_(update, alpha=-group["lr"])
        if self.adamw is not None:
            self.adamw.step()
        return loss
