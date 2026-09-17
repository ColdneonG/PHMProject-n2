from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


class DatasetSmokeTests(unittest.TestCase):
    def test_test_split_sample_shapes(self) -> None:
        from framework.data import LiquidDataset

        data_root = ROOT / "data"
        if not (data_root / "imgs" / "test").exists():
            self.skipTest(f"dataset not found: {data_root}")

        ds = LiquidDataset(data_root, "test", train=False)
        self.assertGreater(len(ds), 0)
        sample = ds[0]
        self.assertEqual(tuple(sample["image"].shape), (3, 256, 256))
        self.assertEqual(tuple(sample["mask"].shape), (1, 256, 256))
        self.assertEqual(tuple(sample["valid"].shape), (1, 256, 256))


if __name__ == "__main__":
    unittest.main()
