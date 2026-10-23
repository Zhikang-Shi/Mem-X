"""Check the pretrained CMA input contracts before GPU inference."""
from pathlib import Path
import gzip
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines/adapt"))
from cma_policy import _load_stoi, _resize_depth, _validate_vocab


class CMAInputsTest(unittest.TestCase):
    def test_vocab_supports_official_word_list_and_explicit_stoi(self):
        for vocab in ({"word_list": ["<pad>", "<unk>", "bench"]},
                      {"stoi": {"<pad>": 0, "<unk>": 1, "bench": 2}}):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "train.json.gz"
                with gzip.open(path, "wt") as handle:
                    json.dump({"instruction_vocab": vocab}, handle)
                self.assertEqual(_load_stoi(path), {"<pad>": 0, "<unk>": 1, "bench": 2})

    def test_incompatible_vocabulary_fails_before_cuda_embedding(self):
        _validate_vocab({"<pad>": 0, "<unk>": 1, "bench": 2}, 3)
        for vocab in ({"<pad>": 0, "<unk>": 1, "bench": 3},
                      {"<pad>": 0, "<unk>": 1}):
            with self.assertRaisesRegex(ValueError, "preprocessed"):
                _validate_vocab(vocab, 3)

    def test_close_range_metric_depth_is_still_normalized(self):
        depth = np.full((3, 4, 1), 0.75, dtype=np.float32)
        actual = _resize_depth(depth, 2)
        self.assertEqual(actual.shape, (2, 2, 1))
        np.testing.assert_allclose(actual, 0.075)
        np.testing.assert_allclose(_resize_depth(np.full((2, 2), 12.0), 2), 1.0)
