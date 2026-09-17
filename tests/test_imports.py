from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


class ImportTests(unittest.TestCase):
    def test_current_modules_import(self) -> None:
        import predict  # noqa: F401
        import evaluate  # noqa: F401
        import train  # noqa: F401
        from framework.data import LiquidDataset  # noqa: F401
        from models import LocalSKGlobalBypassDeepLabV3Plus  # noqa: F401
        from models.baselines import build_baseline_model  # noqa: F401

    def test_parsers_build(self) -> None:
        import predict
        import evaluate
        import train

        self.assertIsNotNone(train.build_parser())
        self.assertIsNotNone(predict.build_parser())
        self.assertIsNotNone(evaluate.build_parser())


if __name__ == "__main__":
    unittest.main()
