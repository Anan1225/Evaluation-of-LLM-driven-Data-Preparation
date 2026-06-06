from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run
import run_data_imputation_single_round as di


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data Imputation OFAT runner (thesis defaults)")
    p.add_argument("--csv", default=None, help="Input CSV for DI; default from config.data_imputation.csv")
    p.add_argument("--model", default="gpt52", help="Model alias or raw model")
    p.add_argument("--id_col", default=None, help="ID column; default from config.data_imputation.id_col or 'id'")
    p.add_argument("--target_cols", default=None, help="Comma-separated columns to impute; default: all except id_col")
    p.add_argument("--case_limit", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=None, help="Override max output tokens (default: 3000)")
    p.add_argument("--missing_batch_size", type=int, default=8, help="Missing-case batch size per model call")
    p.add_argument("--batch_retries", type=int, default=2, help="Retries per batch when coverage is incomplete")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out_dir", default="outputs")
    return p.parse_args()


def _build_ofat_grid() -> list[dict]:
    grid: list[dict] = []

    # Practical OFAT defaults for DI:
    # keep missingness active so example_num axis has measurable signal.
    default_ex = 3
    default_m = 2
    default_e = 10

    for ex in range(0, 11):
        grid.append({"axis": "example_num", "example_num": ex, "m": default_m, "e_pct": default_e})

    for m in [0, 1, 2, 3]:
        grid.append({"axis": "m", "example_num": default_ex, "m": m, "e_pct": default_e})

    for e in [0, 5, 10, 15, 20, 25, 30]:
        grid.append({"axis": "e_pct", "example_num": default_ex, "m": default_m, "e_pct": e})

    return grid


def _resolve_id_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested and requested in df.columns:
        return requested
    for cand in ("id", "ltable_id", "rtable_id"):
        if cand in df.columns:
            return cand
    # Fallback: use first column as ID-like key.
    return str(df.columns[0])


def _predict_missing_in_batches(
    *,
    examples: list[dict],
    missing_cases: list[dict],
    target_cols: list[str],
    provider: str,
    model: str,
    api_key: str,
    cfg: dict,
    system_prompt: str,
    max_tokens: int,
    debug: bool,
    batch_size: int,
    retries: int,
) -> tuple[dict[str, dict], list[dict[str, Any]], list[str]]:
    pred_map: dict[str, dict] = {}
    attempts_meta: list[dict[str, Any]] = []
    raw_chunks: list[str] = []

    if not missing_cases:
        return pred_map, attempts_meta, raw_chunks

    case_by_id = {str(c["id"]): c for c in missing_cases}
    ordered_ids = [str(c["id"]) for c in missing_cases]
    chunks = [ordered_ids[i : i + max(1, batch_size)] for i in range(0, len(ordered_ids), max(1, batch_size))]

    for chunk_idx, id_chunk in enumerate(chunks):
        pending_ids = list(id_chunk)
        for attempt in range(max(0, retries) + 1):
            if not pending_ids:
                break

            batch_cases = [case_by_id[cid] for cid in pending_ids if cid in case_by_id]
            user_prompt = di.build_user_prompt(examples, batch_cases)
            model_resp = run.call_model(
                provider=provider,
                model=model,
                api_key=api_key,
                cfg=cfg,
                system=system_prompt,
                user=user_prompt,
                temperature=float(cfg.get("temperature", 0)),
                max_tokens=max_tokens,
                debug=debug,
            )
            raw = str(model_resp.get("text", ""))
            raw_chunks.append(raw)

            parsed = run.parse_model_json(raw)
            batch_pred = di.normalize_di_predictions(parsed, target_cols=target_cols)
            if batch_pred:
                pred_map.update(batch_pred)

            got_ids = set(batch_pred.keys())
            wanted_ids = set(pending_ids)
            matched_ids = wanted_ids.intersection(got_ids)
            pending_ids = [cid for cid in pending_ids if cid not in matched_ids]

            attempts_meta.append(
                {
                    "chunk_idx": chunk_idx,
                    "attempt": attempt,
                    "requested_ids": len(wanted_ids),
                    "predicted_ids": len(got_ids),
                    "matched_ids": len(matched_ids),
                    "remaining_ids": len(pending_ids),
                }
            )

    return pred_map, attempts_meta, raw_chunks


def main() -> None:
    args = parse_args()

    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required.") from exc

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    provider, model, api_key = run.map_model(args.model, cfg)

    di_cfg = cfg.get("data_imputation", {}) or {}
    csv_path = args.csv or di_cfg.get("csv")
    if not csv_path:
        raise ValueError("Missing DI csv. Set --csv or config.yaml -> data_imputation.csv")

    id_col = args.id_col or di_cfg.get("id_col", "id")
    target_cols_arg = args.target_cols if args.target_cols is not None else di_cfg.get("target_cols")

    df = pd.read_csv(csv_path)
    id_col = _resolve_id_col(df, id_col)
    target_cols = di._choose_target_cols(df, id_col=id_col, target_cols_arg=target_cols_arg)
    system_prompt = "You are a data imputation assistant. Fill missing values accurately."

    out_root = Path(args.out_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = out_root / f"batch_data_imputation_ofat_defaults_{model.replace('/', '_')}_{stamp}"

    grid = _build_ofat_grid()
    records: list[dict] = []

    for i, g in enumerate(grid):
        ex = int(g["example_num"])
        m = int(g["m"])
        e = int(g["e_pct"])
        axis = str(g["axis"])
        combo_seed = int(args.seed + i * 1000 + ex * 100 + m * 10 + e)

        examples, cases = di.build_examples_and_cases(
            df=df,
            id_col=id_col,
            target_cols=target_cols,
            example_num=ex,
            m=m,
            e_pct=e,
            case_limit=args.case_limit,
            seed=combo_seed,
        )

        # Only send cases that actually contain masked values. This improves
        # token efficiency and reduces truncation risk.
        missing_cases = [c for c in cases if (c.get("answer") or {})]
        pred_map, attempts_meta, raw_chunks = _predict_missing_in_batches(
            examples=examples,
            missing_cases=missing_cases,
            target_cols=target_cols,
            provider=provider,
            model=model,
            api_key=api_key,
            cfg=cfg,
            system_prompt=system_prompt,
            max_tokens=(args.max_tokens if args.max_tokens is not None else 3000),
            debug=args.debug,
            batch_size=int(args.missing_batch_size),
            retries=int(args.batch_retries),
        )

        rows, metrics = di.evaluate_cases(cases, pred_map)
        raw = "\n\n----- batch_split -----\n\n".join(raw_chunks) if raw_chunks else "[]"
        user_prompt = di.build_user_prompt(examples, missing_cases)

        run_id = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_axis-di-{axis}"
            f"_ex-{ex}_m-{m}_e-{e}_seed-{combo_seed}_{model.replace('/', '_')}"
        )
        run_dir = batch_root / "data_imputation" / model.replace("/", "_") / run_id

        metadata = {
            "task": "data_imputation",
            "ofat_defaults": True,
            "dataset_csv": str(csv_path),
            "id_col": id_col,
            "target_cols": target_cols,
            "axis": axis,
            "params": {"example_num": ex, "m": m, "e_pct": e},
            "defaults": {"example_num": 3, "m": 2, "e_pct": 10},
            "seed": combo_seed,
            "missing_cases_sent": len(missing_cases),
            "batching": {
                "missing_batch_size": int(args.missing_batch_size),
                "batch_retries": int(args.batch_retries),
                "attempts": attempts_meta,
            },
            "model": model,
            "provider": provider,
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
        }

        di.write_run_outputs(run_dir, user_prompt, raw, rows, metrics, metadata)

        rec = {
            "run_id": run_id,
            "axis": axis,
            "example_num": ex,
            "m": m,
            "e_pct": e,
            "seed": combo_seed,
            "cell_accuracy": metrics["cell_accuracy"],
            "case_accuracy": metrics["case_accuracy"],
            "prediction_coverage": metrics["prediction_coverage"],
            "n_masked_cells": metrics["n_masked_cells"],
            "output_dir": str(run_dir),
        }
        records.append(rec)

        print(
            f"done axis={axis} ex={ex} m={m} e={e} "
            f"cell_acc={metrics['cell_accuracy']:.4f} "
            f"coverage={metrics['prediction_coverage']:.4f}"
        )

    summary_dir = batch_root / "data_imputation" / model.replace("/", "_")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_dir / "summary.csv"
    pd.DataFrame(records).to_csv(summary_csv, index=False)

    print("\ncompleted")
    print(f"batch_root: {batch_root}")
    print(f"summary: {summary_csv}")


if __name__ == "__main__":
    main()
