from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run
from experiment.em_framework import EMRunSpec, build_run_id


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required. Install dependencies first.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_specs(model: str, seed: int, repeat: int) -> list[EMRunSpec]:
    specs: list[EMRunSpec] = []
    for example_num in range(0, 11):
        spec_dict = {
            "axis": "itu-amazon-example",
            "p": 1.0,
            "r_len": 1.0,
            "k": 3,
            "example_num": example_num,
        }
        run_id = build_run_id(spec_dict, model=model, seed=seed, repeat=repeat)
        specs.append(
            EMRunSpec(
                run_id=run_id,
                sweep_axis="itu-amazon-example-num",
                p=1.0,
                r_len=1.0,
                k=3,
                example_num=example_num,
                seed=seed,
                repeat=repeat,
            )
        )
    return specs


def patch_cfg_for_itu_amazon(cfg: dict, out_dir: str) -> dict:
    patched = deepcopy(cfg)
    patched["task"] = "entity_matching"

    em = patched.setdefault("entity_matching", {})
    em["tableA_csv"] = "./datasets/entity_matching/iTunes-Amazon/tableA.csv"
    em["tableB_csv"] = "./datasets/entity_matching/iTunes-Amazon/tableB.csv"
    em["train_csv"] = "./datasets/entity_matching/iTunes-Amazon/train.csv"
    em["valid_csv"] = "./datasets/entity_matching/iTunes-Amazon/valid.csv"
    em["test_csv"] = "./datasets/entity_matching/iTunes-Amazon/test.csv"
    em["test_label_csv"] = "./datasets/entity_matching/iTunes-Amazon/test_label.csv"

    patched.setdefault("em_sweep", {})
    patched["em_sweep"]["key_attributes"] = ["Song_Name", "Artist_Name", "Album_Name"]
    patched["out_dir"] = out_dir
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description="Run iTu-Amazon EM (gpt-5.2) for example_num=0..10 at fixed p=1.0, r_len=1.0, k=3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    cfg = run.ensure_default_em_sweep(cfg)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(args.out_dir) / f"batch_itu_amazon_gpt52_p100_r100_k3_seed{args.seed}_rep{args.repeat}_{stamp}"
    cfg = patch_cfg_for_itu_amazon(cfg, out_dir=str(batch_dir))

    provider, model, api_key = run.map_model("gpt52", cfg)
    out_root = Path(cfg.get("out_dir", "outputs"))

    specs = build_specs(model=model, seed=args.seed, repeat=args.repeat)
    records: list[dict] = []

    for spec in specs:
        rec = run.run_em_once(cfg, provider, model, api_key, spec, out_root, args.debug)
        records.append(rec)
        print(
            f"done example={spec.example_num} run_id={spec.run_id} "
            f"f1={rec['metrics']['f1']:.4f} acc={rec['metrics']['accuracy']:.4f} "
            f"coverage={rec['metrics']['coverage']:.4f}"
        )

    run.write_summary_files(out_root, model, records)

    summary_dir = out_root / "entity_matching" / model.replace("/", "_")
    summary_path = summary_dir / "summary.csv"
    dedicated = summary_dir / f"summary_itu_amazon_gpt52_p100_r100_k3_seed{args.seed}_rep{args.repeat}.csv"
    summary_df = run.summarize_results(records)
    summary_df.to_csv(dedicated, index=False)

    print("\ncompleted total_runs=11")
    print(f"batch_root: {out_root}")
    print(f"summary: {summary_path}")
    print(f"dedicated_summary: {dedicated}")


if __name__ == "__main__":
    main()
