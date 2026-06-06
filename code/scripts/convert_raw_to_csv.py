from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run


def infer_task(run_dir: Path) -> str:
    meta = run_dir / "metadata.json"
    if meta.exists():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            t = str(data.get("task", "")).strip()
            if t:
                return t
        except Exception:
            pass
    p = str(run_dir)
    if "data_imputation" in p:
        return "data_imputation"
    if "entity_matching" in p:
        return "entity_matching"
    return "unknown"


def convert_one(run_dir: Path) -> bool:
    raw_txt = run_dir / "raw_response.txt"
    if not raw_txt.exists():
        return False

    text = raw_txt.read_text(encoding="utf-8")
    parts = text.split("\n\n===== BATCH SPLIT =====\n\n")
    rows = run.raw_batches_to_csv_rows(parts)

    task = infer_task(run_dir)
    if task == "data_imputation":
        # DI commonly uses `imputed` instead of `result`; enrich with parsed item keys.
        enriched = []
        for r in rows:
            item_json = r.get("raw_item_json", "")
            imputed = ""
            try:
                item = json.loads(item_json) if item_json else {}
            except Exception:
                item = {}
            if isinstance(item, dict) and "imputed" in item:
                imputed = json.dumps(item.get("imputed"), ensure_ascii=False)
            enriched.append(
                {
                    "batch_idx": r.get("batch_idx", ""),
                    "id": r.get("id", ""),
                    "imputed": imputed,
                    "reason": r.get("reason", ""),
                    "confidence": r.get("confidence", ""),
                    "raw_item_json": item_json,
                }
            )
        out_df = pd.DataFrame(enriched)
    else:
        out_df = pd.DataFrame(rows)

    out_df.to_csv(run_dir / "raw_response.csv", index=False, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw_response.txt to raw_response.csv for EM/DI runs")
    parser.add_argument("--root", default="outputs", help="Root outputs directory")
    args = parser.parse_args()

    root = Path(args.root)
    run_dirs = [p.parent for p in root.rglob("raw_response.txt")]

    converted = 0
    for d in sorted(set(run_dirs)):
        if convert_one(d):
            converted += 1

    print(f"converted_runs={converted}")


if __name__ == "__main__":
    main()
