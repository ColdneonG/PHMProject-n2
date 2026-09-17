from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PredictSmokeTests(unittest.TestCase):
    def test_cpu_prediction_smoke(self) -> None:
        checkpoint = ROOT / "checkpoints" / "best_model.pt"
        image_root = ROOT / "data" / "imgs" / "test"
        script = ROOT / "src" / "predict.py"
        if not checkpoint.exists():
            self.skipTest(f"checkpoint not found: {checkpoint}")
        images = sorted(p for p in image_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
        if not images:
            self.skipTest(f"no test images found: {image_root}")

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "predict_smoke"
            cmd = [
                sys.executable,
                str(script),
                "--input",
                str(images[0]),
                "--checkpoint",
                str(checkpoint),
                "--out_dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
            result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            self.assertTrue(any(out_dir.rglob("predictions.csv")))
            self.assertTrue(any(out_dir.glob("**/masks/*.png")))
            self.assertTrue(any(out_dir.glob("**/overlays/*.png")))


if __name__ == "__main__":
    unittest.main()
