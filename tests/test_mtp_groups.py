import unittest

from torch import nn

from hybrid_muon import HybridMuonAdamW


class MTPGroupingTest(unittest.TestCase):
    def test_mtp_projection_and_resume_guard(self):
        model = nn.Module()
        model.mtp = nn.Module()
        model.mtp.layers = nn.ModuleList([nn.Module()])
        model.mtp.layers[0].self_attn = nn.Module()
        model.mtp.layers[0].self_attn.q_proj = nn.Linear(4, 4, bias=False)
        model.mtp.fc = nn.Linear(8, 4, bias=False)
        model.mtp.norm = nn.LayerNorm(4)
        opt = HybridMuonAdamW(model, muon_config={"lr": 0.002})
        groups = {g["name"]: g["param_names"] for g in opt.param_groups}
        self.assertEqual(groups["muon"], ["mtp.layers.0.self_attn.q_proj.weight"])
        self.assertEqual(groups["adamw_decay"], ["mtp.fc.weight"])
        self.assertCountEqual(
            groups["adamw_no_decay"], ["mtp.norm.weight", "mtp.norm.bias"]
        )
        state = opt.state_dict()
        state["param_groups"][0]["param_names"] = ["other.weight"]
        with self.assertRaises(ValueError):
            opt.load_state_dict(state)
