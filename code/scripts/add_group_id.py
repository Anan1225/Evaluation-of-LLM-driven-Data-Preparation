from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _resolve_group_columns(df: pd.DataFrame) -> list[str]:
    alias_map = {
        "ingredient": ["ingredient", "active ingredients", "active_ingredients"],
        "strength": ["strength"],
        "dosage_form": ["dosage_form", "dosage form"],
        "route": ["route"],
    }

    lower_to_col = {str(c).strip().lower(): c for c in df.columns}
    resolved: list[str] = []
    for key in ["ingredient", "strength", "dosage_form", "route"]:
        actual = None
        for alias in alias_map[key]:
            if alias in lower_to_col:
                actual = lower_to_col[alias]
                break
        if actual is None:
            raise ValueError(
                f"Missing required column for `{key}`. "
                f"Tried aliases: {alias_map[key]}"
            )
        resolved.append(actual)
    return resolved


def add_group_id(input_path: Path, output_path: Path | None) -> Path:
    df = pd.read_csv(input_path)
    group_cols = _resolve_group_columns(df)

    # user requested logic:
    # df["group_id"] = df.groupby(["ingredient","strength","dosage_form","route"]).ngroup()
    df["group_id"] = df.groupby(group_cols).ngroup()

    out = output_path if output_path is not None else input_path
    df.to_csv(out, index=False, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add group_id by grouping on ingredient/strength/dosage_form/route."
    )
    parser.add_argument(
        "--input",
        default="datasets/semantic_understanding/drug_ground_reconstruction_20_each_enriched.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. If omitted, overwrite input file.",
    )
    args = parser.parse_args()

    output = add_group_id(Path(args.input), Path(args.output) if args.output else None)
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
