"""Unit tests bằng dữ liệu tổng hợp; không cần tải dataset thật."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hagct.config import AugmentConfig, Config, DataConfig, ModelConfig, TrainConfig
from hagct.data.augment import augment_and_pad, scale_bones
from hagct.data.dataset import read_manifest
from hagct.data.filter import one_euro_filter, smoothing_alpha
from hagct.data.preprocess import pad_sequence, preprocess, resample_time
from hagct.engine.common import lr_schedule, optimizer_update
from hagct.engine.losses import classification_loss, mixup_same_mask
from hagct.engine.metrics import classification_metrics
from hagct.models import FourStreamHAGCT, build_streams
from hagct.models.blocks import AdaptiveGraphConv
from hagct.models.pretrain import MaskedReconstruction, reconstruction_loss, sample_joint_mask
from hagct.topology import PARENTS, graph_subsets

torch.set_num_threads(1)


def small_model_config():
    return ModelConfig(num_classes=3, d_model=8, spatial_layers=1, temporal_layers=1,
                       num_heads=2, dropout=0.0, drop_path=0.0)


def raw_sample(length=6):
    x = np.ones((length, 27, 3), np.float32)
    x[:, 0, :2] = 0
    x[:, 1, :2] = (-0.5, 0.2)
    x[:, 2, :2] = (0.5, 0.2)
    return x


class DataTests(unittest.TestCase):
    def test_filter_alpha(self):
        self.assertAlmostEqual(float(smoothing_alpha(1 / 30, 1)), 0.17317071955, places=8)

    def test_filter_constant(self):
        x = np.ones((5, 27, 2), np.float32)
        np.testing.assert_allclose(one_euro_filter(x), x)

    def test_filter_smooth_step(self):
        x = np.zeros((5, 27, 2), np.float32)
        x[1:] = 1
        y = one_euro_filter(x, beta=0)
        self.assertGreater(float(y[1, 0, 0]), 0)
        self.assertLess(float(y[1, 0, 0]), 1)
        self.assertTrue(np.all(np.diff(y[:, 0, 0]) > 0))

    def test_filter_timestamps(self):
        with self.assertRaises(ValueError):
            one_euro_filter(np.ones((2, 27, 2)), timestamps=np.array([1, 1]))

    def test_layout_equivalence(self):
        raw = raw_sample()
        a = preprocess(raw, DataConfig(filter_enabled=False))
        b = preprocess(raw.transpose(2, 0, 1), DataConfig(layout="CTV", filter_enabled=False))
        np.testing.assert_array_equal(a, b)

    def test_missing_interpolation(self):
        raw = raw_sample(3)
        raw[:, 10, 0] = [0, 999, 2]
        raw[1, 10, 2] = 0
        result = preprocess(raw, DataConfig(filter_enabled=False))
        np.testing.assert_allclose(result[0, :, 10], [0, 1, 2])

    def test_entire_missing_joint(self):
        raw = raw_sample()
        raw[:, 12, 2] = 0
        result = preprocess(raw, DataConfig())
        self.assertTrue(np.all(result[:, :, 12] == 0))

    def test_missing_root_rejected(self):
        raw = raw_sample()
        raw[:, 0, 2] = 0
        with self.assertRaises(ValueError):
            preprocess(raw, DataConfig())

    def test_xyz_not_confidence(self):
        raw = raw_sample()
        raw[:, :, 2] = 100
        self.assertTrue(np.isfinite(preprocess(raw, DataConfig(channels="xyz"))).all())
        with self.assertRaises(ValueError):
            preprocess(raw, DataConfig(channels="xy_conf"))

    def test_valid_length(self):
        raw = raw_sample(7)
        raw[3:] = np.nan
        self.assertEqual(preprocess(raw, DataConfig(), 3).shape, (2, 3, 27))

    def test_padding_position(self):
        x, mask = pad_sequence(np.ones((2, 3, 27)), 8, 2)
        np.testing.assert_array_equal(mask, [False, False, True, True, True, False, False, False])
        self.assertTrue(np.all(x[:, ~mask] == 0))

    def test_resampling_retains_endpoints(self):
        x = np.arange(20, dtype=np.float32)[None, :, None].repeat(2, 0).repeat(27, 2)
        out = resample_time(x, 4)
        np.testing.assert_array_equal(out[:, [0, -1]], x[:, [0, -1]])

    def test_bone_scaling_uses_original_vectors(self):
        x = np.random.default_rng(1).normal(size=(2, 3, 27)).astype(np.float32)
        factors = np.linspace(0.8, 1.2, 27)
        out = scale_bones(x, factors)
        expected = (x - x[:, :, PARENTS]) * factors[None, None]
        np.testing.assert_allclose(out - out[:, :, PARENTS], expected, atol=1e-6)

    def test_augmentation_padding_always_zero(self):
        cfg = AugmentConfig(noise_std=10, random_placement=True)
        for seed in range(10):
            x, mask = augment_and_pad(np.ones((2, 5, 27), np.float32), cfg, 16, np.random.default_rng(seed))
            self.assertTrue(np.all(x[:, ~mask] == 0))
            self.assertGreater(int(mask.sum()), 0)

    def test_speed_changes_valid_length(self):
        cfg = AugmentConfig(crop_probability=0, speed_probability=1, speed_min=2, speed_max=2,
                            rotation_degrees=0, bone_scale=0, mask_probability=0, noise_std=0)
        _, mask = augment_and_pad(np.ones((2, 8, 27)), cfg, 16, np.random.default_rng(1))
        self.assertEqual(mask.sum(), 4)

    def test_config_rejects_unknown(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"models": {}})
        with self.assertRaises(TypeError):
            Config.from_dict({"model": {"typo": 1}})

    def test_manifest_leak_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name in ("a", "b"):
                np.save(base / f"{name}.npy", raw_sample())
            path = base / "manifest.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows([["path", "label", "split", "signer"],
                                  ["a.npy", 0, "train", "S1"], ["b.npy", 0, "val", "S1"]])
            with self.assertRaisesRegex(ValueError, "Signer"):
                read_manifest(path, 1, signer_disjoint=True)
            with path.open("a", newline="") as handle:
                csv.writer(handle).writerow(["a.npy", 0, "test", "S2"])
            with self.assertRaisesRegex(ValueError, "nhiều lần"):
                read_manifest(path, 1)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.model = FourStreamHAGCT(small_model_config(), 12)
        self.x = torch.randn(2, 2, 8, 27)
        self.mask = torch.ones(2, 8, dtype=torch.bool)

    def test_stream_values_and_mask_boundaries(self):
        self.mask[:, :2], self.mask[:, -2:] = False, False
        signals = build_streams(self.x, self.mask)
        self.assertTrue(torch.all(signals["motion"][:, :, 2] == 0))
        self.assertTrue(torch.all(signals["bone_motion"][:, :, 2] == 0))
        for value in signals.values():
            self.assertTrue(torch.all(value[:, :, :2] == 0))
            self.assertTrue(torch.all(value[:, :, -2:] == 0))
        torch.testing.assert_close(signals["motion"][:, :, 3], self.x[:, :, 3] - self.x[:, :, 2])
        torch.testing.assert_close(signals["bone"][:, :, 3, 3], self.x[:, :, 3, 3] - self.x[:, :, 3, 1])

    def test_forward_fusion_and_backward_four_encoders(self):
        fused, scores = self.model(self.x, self.mask, return_streams=True)
        self.assertEqual(fused.shape, (2, 3))
        torch.testing.assert_close(fused, torch.stack(list(scores.values())).mean(0))
        fused.square().sum().backward()
        for encoder in self.model.encoders.values():
            gradient = encoder.input_projection.weight.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertIsNotNone(self.model.fusion_logits.grad)

    def test_padding_values_do_not_change_prediction(self):
        self.model.eval()
        self.mask[:, -3:] = False
        changed = self.x.clone()
        changed[:, :, -3:] = float("nan")
        with torch.no_grad():
            torch.testing.assert_close(self.model(self.x, self.mask), self.model(changed, self.mask))

    def test_extra_right_padding_does_not_change_prediction(self):
        self.model.eval()
        padded = torch.nn.functional.pad(self.x, (0, 0, 0, 4))
        mask = torch.nn.functional.pad(self.mask, (0, 4), value=False)
        with torch.no_grad():
            torch.testing.assert_close(self.model(self.x, self.mask), self.model(padded, mask), atol=2e-6, rtol=1e-5)

    def test_empty_mask_rejected(self):
        with self.assertRaises(ValueError):
            self.model(self.x, torch.zeros_like(self.mask))

    def test_temporal_length_not_27(self):
        self.assertTrue(torch.isfinite(self.model(self.x, self.mask)).all())

    def test_graph_normalized_supported_edges(self):
        body, _ = graph_subsets()
        graph = AdaptiveGraphConv(8, body)
        matrix = graph.adjacency().detach()
        self.assertTrue(torch.all(matrix[graph.support == 0] == 0))
        sums = matrix.sum(dim=1)
        torch.testing.assert_close(sums[sums > 0], torch.ones_like(sums[sums > 0]))

    def test_pretrained_initializes_all_four(self):
        pretrain = MaskedReconstruction(small_model_config(), 12)
        report = self.model.initialize_encoders(pretrain.encoder.state_dict())
        self.assertEqual(len(report), 4)
        for encoder in self.model.encoders.values():
            for key, value in pretrain.encoder.state_dict().items():
                torch.testing.assert_close(encoder.state_dict()[key], value)

    def test_pretrained_rejects_wrong_shape(self):
        pretrain = MaskedReconstruction(small_model_config(), 10)
        with self.assertRaises(ValueError):
            self.model.initialize_encoders(pretrain.encoder.state_dict())

    def test_reconstruction_only_selected_positions(self):
        predicted = torch.zeros_like(self.x)
        joints = torch.zeros(2, 27, dtype=torch.bool)
        joints[:, 5] = True
        target = torch.zeros_like(self.x)
        target[:, :, :, 5] = 2
        self.mask[:, -1] = False
        target[:, :, -1] = 1000
        target[:, :, :, 7] = 1000
        loss = reconstruction_loss(predicted, target, self.mask, joints)
        torch.testing.assert_close(loss, torch.tensor([4., 4.]))

    def test_masked_tokens_keep_position_gradient(self):
        pretrain = MaskedReconstruction(small_model_config(), 12)
        joint_mask = sample_joint_mask(2, .3, "cpu")
        output = pretrain(self.x, self.mask, joint_mask)
        reconstruction_loss(output, self.x, self.mask, joint_mask).mean().backward()
        for parameter in (pretrain.encoder.mask_token, pretrain.encoder.joint_position,
                          pretrain.encoder.frame_position):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)


class TrainingTests(unittest.TestCase):
    def test_accumulation_short_final_batch_matches_large_batch(self):
        torch.manual_seed(11)
        a, b = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
        b.load_state_dict(a.state_dict())
        x, y = torch.randn(7, 3), torch.randn(7, 2)
        oa, ob = torch.optim.SGD(a.parameters(), lr=.01), torch.optim.SGD(b.parameters(), lr=.01)
        ((a(x) - y).square().mean(dim=1)).mean().backward()
        oa.step()
        for start in (0, 3, 6):
            ((b(x[start:start + 3]) - y[start:start + 3]).square().mean(dim=1)).sum().backward()
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        scheduler = lr_schedule(ob, 10, 0, .01)
        optimizer_update(b, ob, scaler, scheduler, 7, 1e9)
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb)

    def test_focal_gamma_zero_equals_ce(self):
        logits, labels = torch.randn(4, 3), torch.tensor([0, 1, 2, 0])
        ce = classification_loss(logits, labels, TrainConfig(loss="ce"))
        focal = classification_loss(logits, labels, TrainConfig(loss="focal", focal_gamma=0))
        torch.testing.assert_close(ce, focal)

    def test_mixup_does_not_cross_masks(self):
        x = torch.arange(4.).view(4, 1, 1, 1).expand(4, 2, 5, 27)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0],
                             [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
        _, _, yb, _ = mixup_same_mask(x, mask, torch.arange(4), .4, np.random.default_rng(2))
        torch.testing.assert_close(mask, mask[yb])

    def test_metrics(self):
        result, confusion, classes = classification_metrics([0, 1, 1], [[2, 0], [0, 2], [2, 0]], 2)
        self.assertAlmostEqual(result["top1"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertEqual(confusion.tolist(), [[1, 0], [1, 1]])
        self.assertEqual(classes[1]["support"], 2)


if __name__ == "__main__":
    unittest.main()
