import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import run
from experiment.em_framework import EMRunSpec, generate_em_ofat_specs


class TestRunIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

        (self.base / "prompts" / "entity_matching").mkdir(parents=True)
        (self.base / "prompts" / "entity_matching" / "system.txt").write_text("em task", encoding="utf-8")

        table_a = pd.DataFrame(
            [
                {"id": 1, "Song_Name": "alpha", "Artist_Name": "a1", "Album_Name": "al1"},
                {"id": 2, "Song_Name": "beta", "Artist_Name": "b1", "Album_Name": "bl1"},
                {"id": 3, "Song_Name": "gamma", "Artist_Name": "c1", "Album_Name": "cl1"},
            ]
        )
        table_b = pd.DataFrame(
            [
                {"id": 101, "Song_Name": "alpha", "Artist_Name": "a2", "Album_Name": "al2"},
                {"id": 102, "Song_Name": "delta", "Artist_Name": "d2", "Album_Name": "dl2"},
                {"id": 103, "Song_Name": "gamma", "Artist_Name": "c2", "Album_Name": "cl2"},
            ]
        )

        train = pd.DataFrame(
            [
                {"ltable_id": 1, "rtable_id": 101, "label": 1},
                {"ltable_id": 2, "rtable_id": 102, "label": 0},
                {"ltable_id": 3, "rtable_id": 103, "label": 1},
            ]
        )
        test_label = pd.DataFrame(
            [
                {"ltable_id": 1, "rtable_id": 101, "label": 1},
                {"ltable_id": 2, "rtable_id": 102, "label": 0},
                {"ltable_id": 3, "rtable_id": 103, "label": 1},
            ]
        )

        table_a.to_csv(self.base / "tableA.csv", index=False)
        table_b.to_csv(self.base / "tableB.csv", index=False)
        train.to_csv(self.base / "train.csv", index=False)
        test_label.to_csv(self.base / "test_label.csv", index=False)

        self.cfg = {
            "prompts_dir": str(self.base / "prompts"),
            "out_dir": str(self.base / "outputs"),
            "batch_size": 2,
            "temperature": 0,
            "max_tokens": 100,
            "entity_matching": {
                "tableA_csv": str(self.base / "tableA.csv"),
                "tableB_csv": str(self.base / "tableB.csv"),
                "train_csv": str(self.base / "train.csv"),
                "test_label_csv": str(self.base / "test_label.csv"),
            },
            "em_sweep": {
                "levels": {"p": [1.0, 0.5], "r_len": [1.0], "k": [0], "example_num": [0, 1]},
                "defaults": {"p": 1.0, "r_len": 1.0, "k": 3, "example_num": 0},
                "key_attributes": ["Song_Name", "Artist_Name", "Album_Name"],
                "seeds": [42],
                "n_repeats": 1,
                "positive_label": 1,
            },
        }

        self.old_call_model = run.call_model

        def fake_call_model(**kwargs):
            user = kwargs["user"]
            ids = []
            for line in user.splitlines():
                line = line.strip()
                if line.startswith('{"id"'):
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if "text" in row:
                        ids.append(str(row["id"]))
            # Always predict 1 for odd ids, 0 for even ids.
            out = []
            for cid in ids:
                try:
                    v = int(cid)
                except Exception:
                    v = 0
                out.append(
                    {
                        "id": cid,
                        "result": 1 if v % 2 else 0,
                        "reason": "quick rule-based guess",
                        "confidence": 0.7,
                    }
                )
            return {
                "text": json.dumps(out),
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                "usage_unavailable": False,
            }

        run.call_model = fake_call_model

    def tearDown(self):
        run.call_model = self.old_call_model
        self.tmp.cleanup()

    def test_run_once_outputs(self):
        spec = EMRunSpec(
            run_id="test_run",
            sweep_axis="single",
            p=1.0,
            r_len=1.0,
            k=2,
            example_num=1,
            seed=42,
            repeat=0,
        )
        rec = run.run_em_once(self.cfg, "openai", "gpt-5.2", "dummy", spec, Path(self.cfg["out_dir"]), False)
        run_dir = Path(rec["output_dir"])
        self.assertTrue((run_dir / "predictions.jsonl").exists())
        self.assertTrue((run_dir / "predictions.csv").exists())
        self.assertTrue((run_dir / "metrics.json").exists())
        self.assertTrue((run_dir / "metadata.json").exists())
        first = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertIn("result", first)
        self.assertIn("reason", first)
        self.assertIn("confidence", first)

    def test_sweep_summary_rows(self):
        specs = generate_em_ofat_specs(self.cfg["em_sweep"], model="gpt-5.2", seed=42, repeat=0)
        records = []
        for spec in specs[:4]:
            records.append(run.run_em_once(self.cfg, "openai", "gpt-5.2", "dummy", spec, Path(self.cfg["out_dir"]), False))

        run.write_summary_files(Path(self.cfg["out_dir"]), "gpt-5.2", records)
        summary = Path(self.cfg["out_dir"]) / "entity_matching" / "gpt-5.2" / "summary.csv"
        self.assertTrue(summary.exists())

        df = pd.read_csv(summary)
        self.assertEqual(len(df), 4)


if __name__ == "__main__":
    unittest.main()
