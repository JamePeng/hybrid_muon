import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from hybrid_muon import HybridMuonAdamW, inspect_model
from hybrid_muon.inspection import inspect_checkpoint


class InspectionTests(unittest.TestCase):
    def test_live_report_matches_optimizer_and_handles_frozen_and_tied_weights(self):
        model = nn.Module()
        model.model = nn.Module()
        block = nn.Module()
        model.model.layers = nn.ModuleList([block])
        block.mlp = nn.Module()
        block.mlp.up_proj = nn.Linear(3, 4)
        model.tied = block.mlp.up_proj
        optimizer = HybridMuonAdamW(model)
        report = inspect_model(model, verbose=True)
        for group in optimizer.param_groups:
            for name in group["param_names"]:
                self.assertEqual(report.count(name), 1)
        self.assertRegex(report, r"muon\s+1\s+12\s+75.00%")
        block.mlp.up_proj.weight.requires_grad_(False)
        self.assertIn("including 1 matching Muon projections", inspect_model(model))
        self.assertIn("No trainable Muon projections", inspect_model(model))

    @unittest.skipUnless(importlib.util.find_spec("safetensors"), "Optional checkpoint reader")
    def test_sharded_checkpoint_report(self):
        from safetensors.torch import save_file

        names = ["mtp.layers.0.mlp.up_proj.weight", "model.layers.0.attn.qkv.weight"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            weights = {}
            for index, name in enumerate(names):
                filename = f"weights-{index}.safetensors"
                weights[name] = filename
                save_file({name: torch.zeros(2, 3)}, path / filename)
            (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
            report = inspect_checkpoint(path)
            self.assertRegex(report, r"adamw_decay\s+1\s+6\s+50.00%")
            self.assertIn("MTP: 1 matching", report)
            self.assertIn("1 possible fused projection", report)
