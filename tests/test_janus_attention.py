"""Segmented vision attention must match upstream's masked SDPA numerically."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines/adapt"))
SOURCE = Path(os.environ.get("JANUS_ROOT", ROOT / "baselines/models/JanusVLN")) / "src"
if SOURCE.is_dir():
    sys.path.insert(0, str(SOURCE))
try:
    import torch
    from janus_attention import segmented_vision_sdpa
    from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLVisionSdpaAttention
except ImportError:
    Qwen2_5_VLVisionSdpaAttention = None


@unittest.skipIf(Qwen2_5_VLVisionSdpaAttention is None, "requires upstream Janus model code")
class JanusAttentionTest(unittest.TestCase):
    def test_variable_window_lengths_match_masked_upstream(self):
        torch.manual_seed(7)
        attention = Qwen2_5_VLVisionSdpaAttention(dim=16, num_heads=4).eval()
        hidden = torch.randn(17, 16)
        cumulative = torch.tensor([0, 4, 9, 13, 17], dtype=torch.int32)
        angles = torch.randn(17, 4)
        embeddings = (angles.cos(), angles.sin())
        with torch.no_grad():
            expected = attention(hidden, cumulative, position_embeddings=embeddings)
            actual = segmented_vision_sdpa(
                attention, hidden, cumulative, position_embeddings=embeddings
            )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
