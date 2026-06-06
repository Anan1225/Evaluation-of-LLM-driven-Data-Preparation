from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run


MISSING_TOKEN = "__MISSING__"


def _stringify_row(row: pd.Series) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in row.items():
        if pd.isna(v):
            out[str(k)] = ""
        else:
            out[str(k)] = str(v)
    return out


def _choose_target_cols(df: pd.DataFrame, id_col: str, target_cols_arg: str | None) -> list[str]:
    if target_cols_arg:
        cols = [c.strip() for c in target_cols_arg.split(",") if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"target columns not found: {missing}")
        return cols

    cols = [c for c in df.columns if c != id_col]
    if not cols:
        raise ValueError("No usable target columns found")
    return cols


def _mask_one_row(
    row_dict: dict[str, str],
    target_cols: list[str],
    m: int,
    rng: random.Random,
) -> tuple[dict[str, str], dict[str, str]]:
    masked = dict(row_dict)
    if m <= 0:
        return masked, {}
    # Only mask columns that currently have a non-empty value so evaluation
    # compares meaningful ground truth instead of empty-string placeholders.
    candidates = [c for c in target_cols if str(row_dict.get(c, "")).strip() != ""]
    if not candidates:
        return masked, {}
    k = min(m, len(candidates))
    picked = rng.sample(candidates, k=k)

    answer: dict[str, str] = {}
    for c in picked:
        answer[c] = row_dict.get(c, "")
        masked[c] = MISSING_TOKEN
    return masked, answer


def build_examples_and_cases(
    df: pd.DataFrame,
    id_col: str,
    target_cols: list[str],
    example_num: int,
    m: int,
    e_pct: int,
    case_limit: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)

    if id_col not in df.columns:
        df = df.copy()
        df[id_col] = [str(i) for i in range(len(df))]

    all_idx = list(range(len(df)))
    rng.shuffle(all_idx)

    case_n = min(case_limit, len(all_idx))
    case_idx = all_idx[:case_n]
    rem = all_idx[case_n:]

    ex_n = min(example_num, len(rem))
    example_idx = rem[:ex_n]

    # examples: always masked when m>0 (to make demonstrations useful)
    examples: list[dict] = []
    for i in example_idx:
        row = _stringify_row(df.iloc[i])
        masked, answer = _mask_one_row(row, target_cols, m=m, rng=rng)
        examples.append({"id": str(row[id_col]), "input": masked, "output": answer})

    # cases: missing-rate controlled by e_pct
    e_rate = max(0, min(100, e_pct)) / 100.0
    cases: list[dict] = []
    for i in case_idx:
        row = _stringify_row(df.iloc[i])
        cid = str(row[id_col])

        should_mask = m > 0 and (rng.random() < e_rate)
        if should_mask:
            masked, answer = _mask_one_row(row, target_cols, m=m, rng=rng)
        else:
            masked, answer = dict(row), {}

        cases.append({"id": cid, "input": masked, "answer": answer})

    return examples, cases


def build_user_prompt(examples: list[dict], cases: list[dict]) -> str:
    case_ids = [str(c.get("id", "")) for c in cases]
    lines = [
        "Return ONLY valid JSON.",
        'Return JSON array: [{"id":"...","imputed":{"col":"value",...},"reason":"short reason","confidence":0.0~1.0}]',
        "Only include imputed columns in the `imputed` object.",
        "Only impute fields that are __MISSING__ in each case.",
        "You MUST return exactly one object for each case id.",
        "If uncertain, still provide a best-effort value and lower confidence.",
        "Do NOT return an empty array unless there are zero cases.",
        f"Case IDs to cover: {case_ids}",
        "No extra text.",
        "",
    ]

    if examples:
        lines.append("Examples:")
        for ex in examples:
            lines.append(json.dumps(ex, ensure_ascii=False))
        lines.append("")

    lines.append("Cases:")
    for c in cases:
        lines.append(json.dumps({"id": c["id"], "input": c["input"]}, ensure_ascii=False))

    return "\n".join(lines)


def normalize_di_predictions(parsed: Any, target_cols: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}

    def _extract_json_objects_from_text(raw_text: str) -> list[dict]:
        text = str(raw_text or "").strip()
        if not text:
            return []
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2 and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        dec = json.JSONDecoder()
        items: list[dict] = []
        i = 0
        n = len(text)
        while i < n:
            if text[i] != "{":
                i += 1
                continue
            try:
                obj, end = dec.raw_decode(text, i)
            except Exception:
                i += 1
                continue
            if isinstance(obj, dict):
                items.append(obj)
            i = max(i + 1, end)
        return items

    def _normalize_item(item: dict) -> dict:
        imp = item.get("imputed")
        if isinstance(imp, dict):
            imputed = {str(k): str(v) for k, v in imp.items()}
        else:
            # fallback: pick target columns directly from item
            imputed = {c: str(item[c]) for c in target_cols if c in item}

        reason = str(item.get("reason", "")).strip()
        try:
            conf = float(item.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))

        return {"imputed": imputed, "reason": reason, "confidence": conf}

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and "id" in item:
                out[str(item["id"])] = _normalize_item(item)

    elif isinstance(parsed, dict):
        if "id" in parsed:
            out[str(parsed["id"])] = _normalize_item(parsed)
        for key in ("items", "results", "predictions"):
            arr = parsed.get(key)
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, dict) and "id" in item:
                        out[str(item["id"])] = _normalize_item(item)
        if not out and "_raw" in parsed:
            recovered = _extract_json_objects_from_text(parsed.get("_raw", ""))
            for item in recovered:
                if isinstance(item, dict) and "id" in item:
                    out[str(item["id"])] = _normalize_item(item)

    return out


def evaluate_cases(cases: list[dict], pred_map: dict[str, dict]) -> tuple[list[dict], dict]:
    total_cells = 0
    matched_cells = 0
    missing_cases = 0
    perfect_cases = 0
    predicted_cells = 0

    rows: list[dict] = []

    for c in cases:
        cid = str(c["id"])
        answer: dict = c.get("answer", {}) or {}
        pred = pred_map.get(cid, {"imputed": {}, "reason": "", "confidence": 0.0})
        imputed: dict = pred.get("imputed", {}) or {}

        eval_cells = 0

        case_match = 0
        for k, gt in answer.items():
            # Skip empty-string ground truth cells; they are not informative for
            # imputation quality and can inflate match rate spuriously.
            if str(gt).strip() == "":
                continue
            eval_cells += 1
            total_cells += 1
            pv = str(imputed.get(k, ""))
            if pv != "":
                predicted_cells += 1
            if str(gt).strip().lower() == pv.strip().lower():
                matched_cells += 1
                case_match += 1

        if eval_cells > 0:
            missing_cases += 1
        if eval_cells > 0 and case_match == eval_cells:
            perfect_cases += 1

        rows.append(
            {
                "id": cid,
                "masked_cols": list(answer.keys()),
                "answer": answer,
                "imputed": imputed,
                "reason": pred.get("reason", ""),
                "confidence": pred.get("confidence", 0.0),
                "cell_match": case_match,
                "cell_total": eval_cells,
            }
        )

    metrics = {
        "n_cases": len(cases),
        "n_cases_with_missing": missing_cases,
        "n_masked_cells": total_cells,
        "n_predicted_cells": predicted_cells,
        "n_matched_cells": matched_cells,
        "cell_accuracy": (matched_cells / total_cells) if total_cells else 0.0,
        "case_accuracy": (perfect_cases / missing_cases) if missing_cases else 0.0,
        "prediction_coverage": (predicted_cells / total_cells) if total_cells else 0.0,
    }
    return rows, metrics


def write_run_outputs(run_dir: Path, prompt_user: str, raw_response: str, rows: list[dict], metrics: dict, metadata: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "prompt_user.txt").write_text(prompt_user, encoding="utf-8")
    (run_dir / "raw_response.txt").write_text(raw_response, encoding="utf-8")

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pd.DataFrame(rows).to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(run.raw_batches_to_csv_rows([raw_response])).to_csv(
        run_dir / "raw_response.csv", index=False, encoding="utf-8"
    )

    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-round Data Imputation grid runner")
    p.add_argument("--csv", default=None, help="Input CSV for DI; default from config.data_imputation.csv")
    p.add_argument("--model", default="gpt52", help="Model alias or raw model")
    p.add_argument("--id_col", default=None, help="ID column; default from config.data_imputation.id_col or 'id'")
    p.add_argument("--target_cols", default=None, help="Comma-separated columns to impute; default: all except id_col")
    p.add_argument("--case_limit", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=None, help="Override max output tokens (default: 3000)")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out_dir", default="outputs")
    return p.parse_args()


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
    target_cols = _choose_target_cols(df, id_col=id_col, target_cols_arg=target_cols_arg)

    system_prompt = "You are a data imputation assistant. Fill missing values accurately."

    out_root = Path(args.out_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = out_root / f"batch_data_imputation_single_round_{model.replace('/', '_')}_{stamp}"

    records: list[dict] = []

    example_levels = list(range(0, 11))
    m_levels = [0, 1, 2, 3]
    e_levels = [0, 5, 10, 15, 20, 25, 30]

    for example_num in example_levels:
        for m in m_levels:
            for e in e_levels:
                combo_seed = int(args.seed + example_num * 1000 + m * 100 + e)
                examples, cases = build_examples_and_cases(
                    df=df,
                    id_col=id_col,
                    target_cols=target_cols,
                    example_num=example_num,
                    m=m,
                    e_pct=e,
                    case_limit=args.case_limit,
                    seed=combo_seed,
                )

                user_prompt = build_user_prompt(examples, cases)
                model_resp = run.call_model(
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    cfg=cfg,
                    system=system_prompt,
                    user=user_prompt,
                    temperature=float(cfg.get("temperature", 0)),
                    max_tokens=(args.max_tokens if args.max_tokens is not None else 3000),
                    debug=args.debug,
                )
                raw = str(model_resp.get("text", ""))

                parsed = run.parse_model_json(raw)
                pred_map = normalize_di_predictions(parsed, target_cols=target_cols)
                rows, metrics = evaluate_cases(cases, pred_map)

                run_id = (
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_axis-data-imputation"
                    f"_ex-{example_num}_m-{m}_e-{e}_seed-{combo_seed}_{model.replace('/', '_')}"
                )
                run_dir = batch_root / "data_imputation" / model.replace("/", "_") / run_id

                metadata = {
                    "task": "data_imputation",
                    "single_round": True,
                    "dataset_csv": str(csv_path),
                    "id_col": id_col,
                    "target_cols": target_cols,
                    "params": {"example_num": example_num, "m": m, "e_pct": e},
                    "seed": combo_seed,
                    "model": model,
                    "provider": provider,
                    "created_at_utc": datetime.utcnow().isoformat() + "Z",
                }

                write_run_outputs(run_dir, user_prompt, raw, rows, metrics, metadata)

                rec = {
                    "run_id": run_id,
                    "example_num": example_num,
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
                    f"done ex={example_num} m={m} e={e} "
                    f"cell_acc={metrics['cell_accuracy']:.4f} "
                    f"coverage={metrics['prediction_coverage']:.4f}"
                )

    summary_dir = batch_root / "data_imputation" / model.replace("/", "_")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_dir / "summary.csv"
    pd.DataFrame(records).to_csv(summary_csv, index=False)

    print("\\ncompleted")
    print(f"batch_root: {batch_root}")
    print(f"summary: {summary_csv}")


if __name__ == "__main__":
    main()
