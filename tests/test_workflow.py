"""Integration tests: checkpoint, resume tái lập, pretrain -> fine-tune."""

import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch

from hagct.config import Config
from hagct.engine.common import load_checkpoint, write_json
from hagct.engine.inference import load_model
from hagct.engine.trainer import train
from scripts.make_demo_data import create_demo


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        with contextlib.redirect_stdout(io.StringIO()):
            create_demo(self.base / "data")
        cfg = Config.from_dict({
            "data": {"manifest": str(self.base / "data/manifest.csv"), "max_frames": 12},
            "model": {"num_classes": 3, "d_model": 8, "num_heads": 2, "spatial_layers": 1,
                      "temporal_layers": 1, "dropout": .1, "drop_path": .1},
            "train": {"device": "cpu", "epochs": 2, "batch_size": 4, "accum_steps": 3,
                      "amp": False, "warmup_epochs": 0}})
        self.config = self.base / "config.json"
        write_json(self.config, cfg.to_dict())

    def tearDown(self):
        self.temporary.cleanup()

    def run_train(self, name, pretrain=False, **overrides):
        values = dict(config=str(self.config), output=str(self.base / name), device="cpu", epochs=None,
                      threads=1, resume=None, pretrained=None, stop_after=None, mask_ratio=.3)
        values.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            return train(argparse.Namespace(**values), pretrain=pretrain)

    def test_resume_exact_at_epoch_boundary(self):
        uninterrupted = self.run_train("full")
        resumed = self.run_train("resumed", stop_after=1)
        self.run_train("resumed", resume=str(resumed / "last.pt"))
        a, b = load_checkpoint(uninterrupted / "last.pt"), load_checkpoint(resumed / "last.pt")
        self.assertEqual(a["epoch"], b["epoch"])
        self.assertEqual(a["scheduler"], b["scheduler"])
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)

    def test_pretrain_finetune_save_load(self):
        pretrain = self.run_train("pretrain", pretrain=True, epochs=1)
        finetune = self.run_train("finetune", pretrained=str(pretrain / "best.pt"), epochs=1)
        model, cfg = load_model(finetune / "best.pt", torch.device("cpu"))
        self.assertEqual(cfg.model.num_classes, 3)
        self.assertEqual(len(model.encoders), 4)
        self.assertEqual(load_checkpoint(pretrain / "best.pt")["kind"], "encoder_pretrain")

    def test_refuse_overwrite(self):
        self.run_train("existing", epochs=1)
        with self.assertRaises(FileExistsError):
            self.run_train("existing", epochs=1)

    def test_resume_rejects_changed_schedule(self):
        path = self.run_train("schedule", stop_after=1)
        with self.assertRaisesRegex(ValueError, "cùng cấu hình"):
            self.run_train("schedule", resume=str(path / "last.pt"), epochs=3)


if __name__ == "__main__":
    unittest.main()
