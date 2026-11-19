import tempfile
import unittest
from pathlib import Path

import torch

import hybrid_muon.optimizer as entry


class DDPProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.model.layers[0].mlp = torch.nn.Module()
        self.model.layers[0].mlp.up_proj = torch.nn.Linear(4, 3)
        self.model.layers[0].mlp.down_proj = torch.nn.Linear(3, 4)

    def forward(self, x):
        layer = self.model.layers[0].mlp
        return layer.down_proj(torch.tanh(layer.up_proj(x)))


def ddp_worker(rank, store):
    import copy
    from datetime import timedelta

    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        init_method=Path(store).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        torch.manual_seed(41)
        model = DDPProbe()
        reference = copy.deepcopy(model)
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        opt = entry.HybridMuonAdamW(model)
        refopt = entry.HybridMuonAdamW(reference)
        for _ in range(3):
            opt.zero_grad()
            refopt.zero_grad()
            ddp(torch.full((1, 4), float(rank + 1))).square().mean().backward()
            reference(
                torch.stack([torch.ones(4), torch.full((4,), 2.0)])
            ).square().mean().backward()
            opt.step()
            refopt.step()
            for p, q in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)
                gathered = [torch.empty_like(p) for _ in range(2)]
                torch.distributed.all_gather(gathered, p.detach())
                torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


class OptimizerTests(unittest.TestCase):
    @unittest.skipUnless(torch.distributed.is_gloo_available(), "Gloo unavailable")
    def test_two_process_ddp(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.multiprocessing.spawn(
                ddp_worker, args=(str(Path(tmp) / "store"),), nprocs=2, join=True
            )

    def model(self):
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = model.model.layers[0]
        layer.linear_attn = torch.nn.Module()
        for name in ("in_proj_qkv", "in_proj_z", "out_proj", "in_proj_a", "in_proj_b"):
            setattr(layer.linear_attn, name, torch.nn.Linear(4, 3, bias=False))
        layer.linear_attn.conv1d = torch.nn.Conv1d(3, 3, 4, groups=3)
        model.model.embed_tokens = torch.nn.Embedding(8, 4)
        model.lm_head = torch.nn.Linear(4, 8, bias=False)
        model.lm_head.weight = model.model.embed_tokens.weight
        return model

    def test_grouping_adamw_and_resume(self):
        import copy

        m = self.model()
        opt = entry.HybridMuonAdamW(m)
        groups = {g["name"]: g for g in opt.param_groups}
        self.assertEqual(len(groups["muon"]["params"]), 3)
        self.assertIn(
            "model.layers.0.linear_attn.in_proj_a.weight",
            groups["adamw_decay"]["param_names"],
        )
        self.assertIn(
            "model.layers.0.linear_attn.conv1d.weight",
            groups["adamw_decay"]["param_names"],
        )
        ids = [id(p) for g in opt.param_groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in m.parameters()})
        p = m.model.embed_tokens.weight
        ref = torch.nn.Parameter(p.detach().clone())
        refopt = torch.optim.AdamW([ref], lr=1.5e-5, betas=(0.9, 0.95), weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 0.8**step)
        for param in m.parameters():
            param.grad = torch.ones_like(param)
        ref.grad = torch.ones_like(ref)
        opt.step()
        refopt.step()
        torch.testing.assert_close(p, ref, rtol=0, atol=0)
        scheduler.step()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "optimizer.pt"
            torch.save(opt.state_dict(), checkpoint)
            saved = torch.load(checkpoint, weights_only=True)
        saved_sched = copy.deepcopy(scheduler.state_dict())
        m2 = copy.deepcopy(m)
        opt2 = entry.HybridMuonAdamW(m2)
        sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda step: 0.8**step)
        opt2.load_state_dict(saved)
        sched2.load_state_dict(saved_sched)
        self.assertIs(opt2.adamw.state, opt2.state)
        for x, y in zip(m.parameters(), m2.parameters()):
            x.grad = torch.full_like(x, 0.3)
            y.grad = x.grad.clone()
        opt.step()
        opt2.step()
        scheduler.step()
        sched2.step()
        for x, y in zip(m.parameters(), m2.parameters()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertEqual(scheduler.get_last_lr(), sched2.get_last_lr())
        self.assertAlmostEqual(
            opt.param_groups[0]["lr"] / opt.param_groups[1]["lr"], 5e-5 / 1.5e-5
        )
        before = [p.clone() for p in m.parameters()]
        opt.zero_grad(set_to_none=True)
        opt.step()
        for x, y in zip(before, m.parameters()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_update_modes(self):
        for mode in ("native", "paper"):
            cfg = dict(entry.MUON_CONFIG, scale_mode=mode, update_multiplier=1.7)
            m = self.model()
            opt = entry.HybridMuonAdamW(m, cfg)
            p = opt.param_groups[0]["params"][0]
            old = p.detach().clone()
            p.grad = torch.ones_like(p)
            grad = p.grad.clone()
            momentum = torch.zeros_like(p).lerp_(grad, 1 - cfg["momentum"])
            u = entry.zeropower_via_newtonschulz5(
                grad.lerp(momentum, cfg["momentum"]), cfg["ns_steps"]
            )
            u = (
                u * max(1, p.shape[0] / p.shape[1]) ** 0.5
                if mode == "native"
                else u.float() * (cfg["paper_scale"] * max(p.shape) ** 0.5)
            )
            u = u.float() * cfg["update_multiplier"]
            expected = old.mul(1 - cfg["lr"] * cfg["weight_decay"]).add(
                u, alpha=-cfg["lr"]
            )
            opt.step()
            torch.testing.assert_close(p, expected, rtol=0, atol=0)
            torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
