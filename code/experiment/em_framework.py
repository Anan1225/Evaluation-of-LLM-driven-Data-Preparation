from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class EMRunSpec:
    run_id: str
    sweep_axis: str
    p: float
    r_len: float
    k: int
    example_num: int
    seed: int
    repeat: int


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def load_em_tables(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    em_cfg = cfg["entity_matching"]
    table_a = pd.read_csv(em_cfg["tableA_csv"]).set_index("id", drop=False)
    table_b = pd.read_csv(em_cfg["tableB_csv"]).set_index("id", drop=False)
    train = pd.read_csv(em_cfg["train_csv"])

    label_csv = em_cfg.get("test_label_csv")
    if label_csv:
        test_labeled = pd.read_csv(label_csv)
    else:
        test_csv = pd.read_csv(em_cfg["test_csv"])
        if "label" in test_csv.columns:
            test_labeled = test_csv.copy()
        else:
            raise ValueError("entity_matching.test_label_csv is required for sweep evaluation")

    return table_a, table_b, train, test_labeled


def _sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0].copy()
    if n >= len(df):
        return df.copy().reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def sample_by_ratio(
    labeled_pairs: pd.DataFrame,
    true_to_false_ratio: float,
    seed: int,
    positive_label: int = 1,
) -> pd.DataFrame:
    if true_to_false_ratio <= 0:
        raise ValueError("true_to_false_ratio must be > 0")

    pos = labeled_pairs[labeled_pairs["label"] == positive_label]
    neg = labeled_pairs[labeled_pairs["label"] != positive_label]

    if len(pos) == 0 or len(neg) == 0:
        return labeled_pairs.copy().reset_index(drop=True)

    target_pos = int(len(neg) * true_to_false_ratio)
    target_pos = max(1, min(len(pos), target_pos))
    target_neg = int(round(target_pos / true_to_false_ratio))
    target_neg = max(1, min(len(neg), target_neg))

    sampled_pos = _sample_df(pos, target_pos, seed)
    sampled_neg = _sample_df(neg, target_neg, seed + 1)

    mixed = pd.concat([sampled_pos, sampled_neg], ignore_index=True)
    mixed = mixed.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return mixed


def truncate_value(value: Any, ratio: float) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        text = ""
    if ratio >= 1.0:
        return text
    if ratio <= 0:
        return ""
    n = max(1, int(len(text) * ratio)) if text else 0
    return text[:n]


def select_attributes(row: dict, key_attributes: list[str], k: int, r_len: float) -> dict[str, str]:
    if k < 0:
        raise ValueError("k must be >= 0")
    picked = key_attributes[:k]
    out: dict[str, str] = {}
    for attr in picked:
        out[attr] = truncate_value(row.get(attr, ""), r_len)
    return out


def format_entity_pair_text(left: dict, right: dict) -> str:
    left_part = ", ".join([f"{k}: {v}" for k, v in left.items()])
    right_part = ", ".join([f"{k}: {v}" for k, v in right.items()])
    if not left_part:
        left_part = "(no key attributes selected)"
    if not right_part:
        right_part = "(no key attributes selected)"
    return f"Song A {left_part}; Song B {right_part}"


def build_examples(
    train_df: pd.DataFrame,
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    key_attributes: list[str],
    k: int,
    r_len: float,
    example_num: int,
    seed: int,
) -> list[dict]:
    if example_num <= 0:
        return []

    sampled = _sample_df(train_df, example_num, seed)
    examples: list[dict] = []
    for i, row in sampled.iterrows():
        left = table_a.loc[_safe_int(row["ltable_id"])].to_dict()
        right = table_b.loc[_safe_int(row["rtable_id"])].to_dict()
        left_attrs = select_attributes(left, key_attributes, k, r_len)
        right_attrs = select_attributes(right, key_attributes, k, r_len)
        examples.append(
            {
                "id": f"ex_{i}",
                "text": format_entity_pair_text(left_attrs, right_attrs),
                "label": _safe_int(row.get("label", 0)),
            }
        )
    return examples


def build_cases(
    sampled_test: pd.DataFrame,
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    key_attributes: list[str],
    k: int,
    r_len: float,
    id_col: Optional[str] = None,
) -> list[dict]:
    cases: list[dict] = []
    for idx, row in sampled_test.iterrows():
        left = table_a.loc[_safe_int(row["ltable_id"])].to_dict()
        right = table_b.loc[_safe_int(row["rtable_id"])].to_dict()
        left_attrs = select_attributes(left, key_attributes, k, r_len)
        right_attrs = select_attributes(right, key_attributes, k, r_len)

        if id_col and id_col in sampled_test.columns:
            case_id = str(row[id_col])
        else:
            case_id = str(idx)

        cases.append(
            {
                "id": case_id,
                "text": format_entity_pair_text(left_attrs, right_attrs),
                "label": _safe_int(row.get("label", 0)),
            }
        )
    return cases


def build_user_prompt(examples: list[dict], cases: list[dict]) -> str:
    import json

    lines = [
        "Return ONLY valid JSON.",
        'Return a JSON array, one item per case: [{"id":"...","result":0 or 1,"reason":"short reason","confidence":0.0~1.0}, ...]',
        "When examples are provided, infer the matching rule from examples and apply it consistently to all cases.",
        "reason should be short and simple (about one sentence).",
        "confidence must be a float between 0 and 1.",
        "No extra text.",
        "",
    ]

    if examples:
        lines.append("Examples (JSONL):")
        for ex in examples:
            lines.append(json.dumps(ex, ensure_ascii=False))
        lines.append("")

    lines.append("Cases (JSONL):")
    for c in cases:
        lines.append(json.dumps({"id": c["id"], "text": c["text"]}, ensure_ascii=False))

    return "\n".join(lines).strip()


def build_run_id(spec: dict, model: str, seed: int, repeat: int, now: Optional[datetime] = None) -> str:
    ts = (now or datetime.utcnow()).strftime("%Y%m%d_%H%M%S")
    return (
        f"{ts}_axis-{spec['axis']}_p-{int(spec['p']*100)}"
        f"_r-{int(spec['r_len']*100)}_k-{int(spec['k'])}"
        f"_ex-{int(spec['example_num'])}_seed-{seed}_rep-{repeat}_{model.replace('/', '_')}"
    )


def generate_em_ofat_specs(
    sweep_cfg: dict,
    model: str,
    seed: int,
    repeat: int,
) -> list[EMRunSpec]:
    defaults = sweep_cfg["defaults"]
    levels = sweep_cfg["levels"]

    p_default = _safe_float(defaults.get("p", 1.0), 1.0)
    r_default = _safe_float(defaults.get("r_len", 1.0), 1.0)
    k_default = _safe_int(defaults.get("k", 3), 3)

    example_levels = levels.get("example_num", list(range(0, 11)))

    raw_specs: list[dict] = []

    for p in levels.get("p", [p_default]):
        for ex in example_levels:
            raw_specs.append({"axis": "p", "p": _safe_float(p), "r_len": r_default, "k": k_default, "example_num": _safe_int(ex)})

    for r in levels.get("r_len", [r_default]):
        for ex in example_levels:
            raw_specs.append({"axis": "r_len", "p": p_default, "r_len": _safe_float(r), "k": k_default, "example_num": _safe_int(ex)})

    for k in levels.get("k", [k_default]):
        for ex in example_levels:
            raw_specs.append({"axis": "k", "p": p_default, "r_len": r_default, "k": _safe_int(k), "example_num": _safe_int(ex)})

    out: list[EMRunSpec] = []
    for s in raw_specs:
        rid = build_run_id(s, model=model, seed=seed, repeat=repeat)
        out.append(
            EMRunSpec(
                run_id=rid,
                sweep_axis=s["axis"],
                p=s["p"],
                r_len=s["r_len"],
                k=s["k"],
                example_num=s["example_num"],
                seed=seed,
                repeat=repeat,
            )
        )

    return out


def summarize_results(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        m = r.get("metrics", {})
        rows.append(
            {
                "run_id": r.get("run_id"),
                "sweep_axis": r.get("sweep_axis"),
                "p": r.get("p"),
                "r_len": r.get("r_len"),
                "k": r.get("k"),
                "example_num": r.get("example_num"),
                "seed": r.get("seed"),
                "repeat": r.get("repeat"),
                "n": m.get("n"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "accuracy": m.get("accuracy"),
                "tp": m.get("tp"),
                "tn": m.get("tn"),
                "fp": m.get("fp"),
                "fn": m.get("fn"),
                "output_dir": r.get("output_dir"),
            }
        )
    return pd.DataFrame(rows)
