import unittest

import pandas as pd

from experiment.em_framework import (
    generate_em_ofat_specs,
    sample_by_ratio,
    select_attributes,
    truncate_value,
)


class TestEMFramework(unittest.TestCase):
    def test_generate_spec_count(self):
        sweep_cfg = {
            "levels": {
                "p": [0.25, 1.0],
                "r_len": [0.2, 1.0],
                "k": [0, 3],
                "example_num": [0, 1],
            },
            "defaults": {"p": 1.0, "r_len": 1.0, "k": 3, "example_num": 0},
        }
        specs = generate_em_ofat_specs(sweep_cfg, model="gpt-5.2", seed=42, repeat=0)
        self.assertEqual(len(specs), (2 + 2 + 2) * 2)

    def test_ratio_sampling(self):
        df = pd.DataFrame(
            {
                "ltable_id": list(range(12)),
                "rtable_id": list(range(100, 112)),
                "label": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            }
        )
        sampled = sample_by_ratio(df, true_to_false_ratio=0.5, seed=7, positive_label=1)
        n_pos = int((sampled["label"] == 1).sum())
        n_neg = int((sampled["label"] == 0).sum())
        self.assertGreater(n_pos, 0)
        self.assertGreater(n_neg, 0)
        self.assertAlmostEqual(n_pos / n_neg, 0.5, places=1)

    def test_attribute_transform_edges(self):
        row = {"Song_Name": "abcdef", "Artist_Name": "xy", "Album_Name": "z"}

        self.assertEqual(truncate_value("abcdef", 1.0), "abcdef")
        self.assertEqual(truncate_value("abcdef", 0.5), "abc")
        self.assertEqual(truncate_value("abcdef", 0.0), "")

        attrs = select_attributes(row, ["Song_Name", "Artist_Name", "Album_Name"], k=2, r_len=0.5)
        self.assertEqual(list(attrs.keys()), ["Song_Name", "Artist_Name"])
        self.assertEqual(attrs["Song_Name"], "abc")


if __name__ == "__main__":
    unittest.main()
