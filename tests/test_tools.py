"""Kiểm thử các tiện ích chuẩn bị dữ liệu bằng fixture nhỏ, không dùng dataset thật."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hagct.data.dataset import read_manifest
from scripts import convert_vsl400, prepare_multivsl, prepare_27kpt_zip


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def invoke(self, module, *args):
        with patch("sys.argv", [module.__name__, *map(str, args)]), contextlib.redirect_stdout(io.StringIO()):
            module.main()

    def make_multivsl(self):
        source = self.base / "raw"
        for word in ("001", "002"):
            (source / word).mkdir(parents=True)
            for signer in ("S1", "S2", "S3"):
                np.save(source / word / f"{signer}_01.npy", np.ones((4, 27, 3), np.float32))
        return source

    def test_multivsl_signer_manifest(self):
        source, output = self.make_multivsl(), self.base / "manifest"
        self.invoke(prepare_multivsl, "--source", source, "--output", output,
                    "--train-signers", "S1", "--val-signers", "S2", "--test-signers", "S3")
        rows = read_manifest(output / "manifest.csv", 2, signer_disjoint=True)
        self.assertEqual(len(rows), 6)
        self.assertEqual(json.loads((output / "classes.json").read_text()), ["001", "002"])

    def test_multivsl_random_split(self):
        source, output = self.make_multivsl(), self.base / "random"
        self.invoke(prepare_multivsl, "--source", source, "--output", output, "--protocol", "random")
        rows = read_manifest(output / "manifest.csv", 2)
        for split in ("train", "val", "test"):
            self.assertEqual(sum(row["split"] == split for row in rows), 2)

    def test_multivsl_unknown_signer_rejected(self):
        source = self.make_multivsl()
        with self.assertRaisesRegex(ValueError, "chưa được khai báo"):
            self.invoke(prepare_multivsl, "--source", source, "--output", self.base / "invalid",
                        "--train-signers", "S1", "--val-signers", "S2", "--test-signers", "S4")
        self.assertFalse((self.base / "invalid").exists())

    def converter_args(self, layout="NCTV"):
        tensor = np.arange(2 * 2 * 5 * 27, dtype=np.float32).reshape(2, 2, 5, 27)
        if layout == "NCTVM":
            tensor = np.stack((tensor, tensor + 100), axis=-1)
        np.save(self.base / "raw.npy", tensor)
        np.save(self.base / "labels.npy", np.array([0, 1]))
        np.save(self.base / "lengths.npy", np.array([3, 4]))
        args = ["--data", self.base / "raw.npy", "--labels", self.base / "labels.npy", "--split", "train",
                "--layout", layout, "--channels", "xy", "--lengths", self.base / "lengths.npy",
                "--num-classes", "2", "--output", self.base / "converted"]
        return tensor, args

    def test_converter_preserves_axes_and_cuts_padding(self):
        tensor, args = self.converter_args()
        self.invoke(convert_vsl400, *args)
        raw = np.load(self.base / "converted/train/0000000.npy")
        np.testing.assert_array_equal(raw, tensor[0, :, :3].transpose(1, 2, 0))
        self.assertEqual(len(read_manifest(self.base / "converted/manifest.csv", 2)), 2)

    def test_converter_explicit_person(self):
        tensor, args = self.converter_args("NCTVM")
        self.invoke(convert_vsl400, *args, "--person-index", 1)
        raw = np.load(self.base / "converted/train/0000000.npy")
        np.testing.assert_array_equal(raw, tensor[0, :, :3, :, 1].transpose(1, 2, 0))

    def test_converter_refuses_ambiguous_lengths(self):
        _, args = self.converter_args()
        with self.assertRaisesRegex(ValueError, "lengths"):
            self.invoke(convert_vsl400, *args, "--all-frames-valid")

    def test_pickle_requires_opt_in(self):
        path = self.base / "untrusted.pkl"
        path.write_bytes(b"not a pickle")
        with self.assertRaisesRegex(ValueError, "trust-pickle"):
            convert_vsl400.read_labels(path, False)

    def test_prepare_27kpt_zip(self):
        archive = self.base / "27kpt.zip"
        import zipfile
        with zipfile.ZipFile(archive, "w") as handle:
            labels = []
            for label in range(2):
                for signer in ("02", "03", "07"):
                    stem = f"demo_signer{signer}_center_ord1_{label}"
                    sample = self.base / f"{stem}.npy"
                    np.save(sample, np.ones((150, 27, 3), dtype=np.float32))
                    handle.write(sample, f"raw_npy/{sample.name}")
                    labels.append(f"{stem},{label}\n")
            handle.writestr("raw_npy/labels.csv", "".join(labels))
        output = self.base / "trial"
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_27kpt_zip.prepare(archive, output, 2, ("02",), ("03",))
        rows = read_manifest(output / "manifest.csv", 2, signer_disjoint=True)
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["split"] for row in rows}, {"train", "val", "test"})
        config = json.loads((output / "trial_config.json").read_text())
        self.assertEqual(config["data"]["channels"], "xyz")


if __name__ == "__main__":
    unittest.main()
