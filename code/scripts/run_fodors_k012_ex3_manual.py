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


def _parse_attrs(s: str) -> list[str]:
    items = [x.strip() for x in str(s).split(",")]
    return [x for x in items if x]


def build_specs(model: str, seed: int, repeat: int) -> list[EMRunSpec]:
    specs: list[EMRunSpec] = []
    for k in (1, 2, 3):
        spec_dict = {
            "axis": "fodors-k-manual",
            "p": 0.5,
            "r_len": 1.0,
            "k": k,
            "example_num": 5,
        }
        run_id = build_run_id(spec_dict, model=model, seed=seed, repeat=repeat)
        specs.append(
            EMRunSpec(
                run_id=run_id,
                sweep_axis="fodors-k-manual",
                p=0.5,
                r_len=1.0,
                k=k,
                example_num=5,
                seed=seed,
                repeat=repeat,
            )
        )
    return specs


def patch_cfg_for_fodors(cfg: dict, out_dir: str) -> dict:
    patched = deepcopy(cfg)
    patched["task"] = "entity_matching"
    em = patched.setdefault("entity_matching", {})
    em["tableA_csv"] = "./datasets/entity_matching/Fodors-Zagats/tableA.csv"
    em["tableB_csv"] = "./datasets/entity_matching/Fodors-Zagats/tableB.csv"
    em["train_csv"] = "./datasets/entity_matching/Fodors-Zagats/train.csv"
    em["valid_csv"] = "./datasets/entity_matching/Fodors-Zagats/valid.csv"
    em["test_csv"] = "./datasets/entity_matching/Fodors-Zagats/test.csv"
    em["test_label_csv"] = "./datasets/entity_matching/Fodors-Zagats/test.csv"
    patched["batch_size"] = 2
    patched["max_tokens"] = 4096
    patched["out_dir"] = out_dir
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Fodors-Zagats EM with manual key attributes for k=1,2,3 (fixed ex=5, p=0.5, r_len=1.0)"
    )
    parser.add_argument("--model", default="gpt4o", help="gpt4o | gpt52 | gf | gp | raw model name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument(
        "--k1_attrs",
        default="name",
        help="Comma-separated attrs used when k=1, e.g. name",
    )
    parser.add_argument(
        "--k2_attrs",
        default="name,addr",
        help="Comma-separated attrs used when k=2, e.g. name,addr",
    )
    parser.add_argument(
        "--k3_attrs",
        default="name,addr,city",
        help="Comma-separated attrs used when k=3, e.g. name,addr,city",
    )
    args = parser.parse_args()

    k1_attrs = _parse_attrs(args.k1_attrs)
    k2_attrs = _parse_attrs(args.k2_attrs)
    k3_attrs = _parse_attrs(args.k3_attrs)
    if len(k1_attrs) < 1:
        raise ValueError("--k1_attrs must include at least 1 attribute")
    if len(k2_attrs) < 2:
        raise ValueError("--k2_attrs must include at least 2 attributes")
    if len(k3_attrs) < 3:
        raise ValueError("--k3_attrs must include at least 3 attributes")

    cfg = load_yaml(Path(args.config))
    cfg = run.ensure_default_em_sweep(cfg)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = (
        Path(args.out_dir)
        / f"batch_fodors_k123_ex5_manual_seed{args.seed}_rep{args.repeat}_{stamp}"
    )
    cfg = patch_cfg_for_fodors(cfg, out_dir=str(batch_dir))

    provider, model, api_key = run.map_model(args.model, cfg)
    out_root = Path(cfg.get("out_dir", "outputs"))
    specs = build_specs(model=model, seed=args.seed, repeat=args.repeat)

    records: list[dict] = []
    key_map = {
        1: k1_attrs,
        2: k2_attrs,
        3: k3_attrs,
    }

    # Keep all columns in prompt text, while changing key-attribute decision standard by k.
    cfg.setdefault("em_sweep", {})
    cfg["em_sweep"]["key_attributes"] = ["name", "addr", "city", "phone", "type", "class"]
    cfg["em_sweep"]["display_all_attributes"] = True
    cfg["em_sweep"]["include_key_guidance"] = False
    cfg["em_sweep"]["relabel_by_selected_keys"] = True
    cfg["em_sweep"]["example_positive_only"] = True
    cfg["em_sweep"]["example_require_non_key_variation"] = True
    cfg.setdefault("gemini", {})
    cfg["gemini"]["max_retries"] = 1
    cfg["gemini"]["base_retry_seconds"] = 1.0
    cfg["gemini"]["max_retry_seconds"] = 5.0
    cfg["gemini"]["fail_fast_on_429"] = True
    cfg["em_sweep"]["manual_key_guidance_by_k"] = {
        "1": key_map[1],
        "2": key_map[2],
        "3": key_map[3],
    }

    for spec in specs:
        run_cfg = deepcopy(cfg)
        rec = run.run_em_once(run_cfg, provider, model, api_key, spec, out_root, args.debug)
        records.append(rec)
        print(
            f"done k={spec.k} ex={spec.example_num} attrs={key_map[spec.k]} "
            f"f1={rec['metrics']['f1']:.4f} acc={rec['metrics']['accuracy']:.4f} "
            f"coverage={rec['metrics']['coverage']:.4f}"
        )

    run.write_summary_files(out_root, model, records)
    summary_dir = out_root / "entity_matching" / model.replace("/", "_")
    dedicated = summary_dir / f"summary_fodors_k123_ex5_manual_seed{args.seed}_rep{args.repeat}.csv"
    run.summarize_results(records).to_csv(dedicated, index=False)

    print("\ncompleted total_runs=3")
    print(f"batch_root: {out_root}")
    print(f"dedicated_summary: {dedicated}")


if __name__ == "__main__":
    main()
