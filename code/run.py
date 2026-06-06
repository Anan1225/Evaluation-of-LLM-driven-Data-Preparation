from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
try:
    import httpx
except Exception:  # pragma: no cover - handled at runtime in model calls.
    httpx = None  # type: ignore

OPENAI_AVAILABLE = True
try:
    from openai import OpenAI
except Exception:  # pragma: no cover - handled at runtime in model calls.
    OPENAI_AVAILABLE = False
    OpenAI = Any  # type: ignore

from experiment.em_framework import (
    EMRunSpec,
    build_cases,
    build_examples,
    build_run_id,
    build_user_prompt,
    generate_em_ofat_specs,
    load_em_tables,
    sample_by_ratio,
    summarize_results,
)
from experiment.evaluation import classification_metrics


def read_text(p: str) -> str:
    return Path(p).read_text(encoding="utf-8")


def chunk(lst: list[Any], n: int) -> list[list[Any]]:
    out: list[list[Any]] = []
    for i in range(0, len(lst), n):
        out.append(lst[i : i + n])
    return out


def _mask_key(v: str) -> str:
    if not v:
        return ""
    s = str(v)
    return "****" + (s[-4:] if len(s) > 4 else s)


def _ascii_safe(v: str) -> str:
    if v is None:
        return ""
    return v.encode("ascii", "ignore").decode("ascii")


def load_system_prompt(cfg: dict, task: str) -> str:
    prompts_dir = cfg.get("prompts_dir", "prompts")
    system_path = Path(prompts_dir) / task / "system.txt"
    return read_text(str(system_path)) if system_path.exists() else ""


def parse_model_json(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        return {"_raw": text}

    try:
        return json.loads(raw)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.S)
    if fenced:
        payload = fenced.group(1)
        try:
            return json.loads(payload)
        except Exception:
            pass

    for pattern in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pattern, raw, flags=re.S)
        if m:
            snippet = m.group(0)
            try:
                return json.loads(snippet)
            except Exception:
                continue

    return {"_raw": text}


def _to_confidence(v: Any) -> float:
    try:
        c = float(v)
    except Exception:
        return 0.0
    if c < 0:
        return 0.0
    if c > 1:
        return 1.0
    return c


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _usage_payload(
    input_tokens: Any = 0,
    output_tokens: Any = 0,
    total_tokens: Any = 0,
) -> dict[str, int]:
    return {
        "input_tokens": max(0, _as_int(input_tokens, 0)),
        "output_tokens": max(0, _as_int(output_tokens, 0)),
        "total_tokens": max(0, _as_int(total_tokens, 0)),
    }


def _model_response(text: str, usage: Optional[dict[str, int]] = None, usage_unavailable: bool = False) -> dict:
    return {
        "text": text or "",
        "usage": usage or _usage_payload(),
        "usage_unavailable": bool(usage_unavailable),
    }


def _parse_usage_from_openai_compat(data: dict) -> tuple[dict[str, int], bool]:
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return _usage_payload(), True

    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    if _as_int(total_tokens, 0) <= 0:
        total_tokens = _as_int(input_tokens, 0) + _as_int(output_tokens, 0)
    return _usage_payload(input_tokens, output_tokens, total_tokens), False


def _parse_usage_from_gemini(data: dict) -> tuple[dict[str, int], bool]:
    um = data.get("usageMetadata") if isinstance(data, dict) else None
    if not isinstance(um, dict):
        return _usage_payload(), True

    input_tokens = um.get("promptTokenCount", 0)
    output_tokens = um.get("candidatesTokenCount", 0)
    total_tokens = um.get("totalTokenCount", 0)
    if _as_int(total_tokens, 0) <= 0:
        total_tokens = _as_int(input_tokens, 0) + _as_int(output_tokens, 0)
    return _usage_payload(input_tokens, output_tokens, total_tokens), False


def _get_billing_context(cfg: dict, model: str) -> dict:
    billing = cfg.get("billing", {}) or {}
    pricing = billing.get("pricing_per_million", {}) or {}
    model_pricing = pricing.get(model) or pricing.get(model.replace("/", "_")) or {}

    input_price = _as_float((model_pricing or {}).get("input_usd"))
    output_price = _as_float((model_pricing or {}).get("output_usd"))
    total_budget = _as_float(billing.get("total_budget_usd"))

    return {
        "total_budget_usd": total_budget,
        "input_price_usd_per_million": input_price,
        "output_price_usd_per_million": output_price,
    }


def _estimate_cost_usd(input_tokens: int, output_tokens: int, billing_ctx: dict) -> Optional[float]:
    in_price = billing_ctx.get("input_price_usd_per_million")
    out_price = billing_ctx.get("output_price_usd_per_million")
    if in_price is None or out_price is None:
        return None
    return (float(input_tokens) / 1_000_000.0) * float(in_price) + (float(output_tokens) / 1_000_000.0) * float(out_price)


def _fmt_money(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"${v:.6f}"


def _gemini_alias_candidates(model: str) -> list[str]:
    m = (model or "").strip()
    out: list[str] = []
    if m:
        out.append(m)
    if m and not m.endswith("-latest"):
        out.append(f"{m}-latest")

    ml = m.lower()
    if "flash" in ml:
        out.extend(["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"])
    if "pro" in ml:
        out.extend(["gemini-2.5-pro"])

    dedup: list[str] = []
    for x in out:
        if x and x not in dedup:
            dedup.append(x)
    return dedup


def _normalize_match_value(v: Any) -> str:
    s = str(v or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    # Keep alnum and spaces only for tolerant matching.
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s.strip()


def _label_by_key_attributes(left: dict, right: dict, attrs: list[str]) -> int:
    if not attrs:
        return 0
    for a in attrs:
        lv = _normalize_match_value(left.get(a, ""))
        rv = _normalize_match_value(right.get(a, ""))
        if lv != rv:
            return 0
    return 1


def _relabel_pairs_by_keys(
    pairs: pd.DataFrame,
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    attrs: list[str],
) -> pd.DataFrame:
    if not attrs:
        return pairs.copy()
    out = pairs.copy()
    labels: list[int] = []
    for _, row in out.iterrows():
        left = table_a.loc[_as_int(row.get("ltable_id"), 0)].to_dict()
        right = table_b.loc[_as_int(row.get("rtable_id"), 0)].to_dict()
        labels.append(_label_by_key_attributes(left, right, attrs))
    out["label"] = labels
    return out


def _has_non_key_variation(
    row: pd.Series,
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    non_key_attrs: list[str],
) -> bool:
    if not non_key_attrs:
        return False
    left = table_a.loc[_as_int(row.get("ltable_id"), 0)].to_dict()
    right = table_b.loc[_as_int(row.get("rtable_id"), 0)].to_dict()
    for a in non_key_attrs:
        if _normalize_match_value(left.get(a, "")) != _normalize_match_value(right.get(a, "")):
            return True
    return False


def _filter_example_pool(
    pairs: pd.DataFrame,
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    selected_keys: list[str],
    all_attrs: list[str],
    positive_only: bool,
    require_non_key_variation: bool,
) -> pd.DataFrame:
    out = pairs.copy()
    if positive_only:
        out = out[out["label"] == 1].copy()

    if require_non_key_variation:
        non_key_attrs = [a for a in all_attrs if a not in set(selected_keys)]
        if non_key_attrs:
            keep = []
            for _, row in out.iterrows():
                keep.append(_has_non_key_variation(row, table_a, table_b, non_key_attrs))
            out = out.loc[keep].copy()

    return out.reset_index(drop=True)


def _retry_after_seconds(headers: Any) -> Optional[float]:
    if headers is None:
        return None
    raw = None
    try:
        raw = headers.get("retry-after")
    except Exception:
        raw = None
    if raw is None:
        return None
    try:
        v = float(str(raw).strip())
    except Exception:
        return None
    if v < 0:
        return None
    return v


def _normalize_pred_item(item: dict) -> dict:
    return {
        "result": int(item.get("result", -1)),
        "reason": str(item.get("reason", "")).strip(),
        "confidence": _to_confidence(item.get("confidence", 0.0)),
    }


def normalize_prediction_map(parsed: Any) -> dict[str, dict]:
    out: dict[str, dict] = {}

    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            if "id" not in item:
                continue
            if "result" not in item:
                continue
            out[str(item["id"])] = _normalize_pred_item(item)

    elif isinstance(parsed, dict):
        # Optional fallback: {"items": [...]} or {"results": [...]}.
        for key in ("items", "results", "predictions"):
            if isinstance(parsed.get(key), list):
                for item in parsed[key]:
                    if isinstance(item, dict) and "id" in item and "result" in item:
                        out[str(item["id"])] = _normalize_pred_item(item)

    return out


def _extract_prediction_items(parsed: Any) -> list[dict]:
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if isinstance(parsed, dict):
        for key in ("items", "results", "predictions"):
            arr = parsed.get(key)
            if isinstance(arr, list):
                return [x for x in arr if isinstance(x, dict)]
    return []


def raw_batches_to_csv_rows(raw_batches: list[str]) -> list[dict]:
    rows: list[dict] = []
    for bidx, raw in enumerate(raw_batches):
        parsed = parse_model_json(raw)
        items = _extract_prediction_items(parsed)
        if not items:
            rows.append(
                {
                    "batch_idx": bidx,
                    "id": "",
                    "result": "",
                    "reason": "",
                    "confidence": "",
                    "raw_item_json": json.dumps(parsed, ensure_ascii=False),
                }
            )
            continue

        for item in items:
            rows.append(
                {
                    "batch_idx": bidx,
                    "id": str(item.get("id", "")),
                    "result": item.get("result", ""),
                    "reason": str(item.get("reason", "")),
                    "confidence": item.get("confidence", ""),
                    "raw_item_json": json.dumps(item, ensure_ascii=False),
                }
            )
    return rows


def call_openai_gpt4o_chat(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: Optional[int],
) -> dict:
    kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        **kwargs,
    )
    usage_obj = getattr(resp, "usage", None)
    input_tokens = getattr(usage_obj, "prompt_tokens", 0) if usage_obj else 0
    output_tokens = getattr(usage_obj, "completion_tokens", 0) if usage_obj else 0
    total_tokens = getattr(usage_obj, "total_tokens", 0) if usage_obj else 0
    if _as_int(total_tokens, 0) <= 0:
        total_tokens = _as_int(input_tokens, 0) + _as_int(output_tokens, 0)
    usage_unavailable = usage_obj is None

    return _model_response(
        text=resp.choices[0].message.content or "",
        usage=_usage_payload(input_tokens, output_tokens, total_tokens),
        usage_unavailable=usage_unavailable,
    )


def call_openai_responses(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_output_tokens: Optional[int],
) -> dict:
    kwargs: dict[str, Any] = {}
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = int(max_output_tokens)

    resp = client.responses.create(
        model=model,
        instructions=system,
        input=user,
        temperature=temperature,
        **kwargs,
    )
    usage_obj = getattr(resp, "usage", None)
    input_tokens = getattr(usage_obj, "input_tokens", 0) if usage_obj else 0
    output_tokens = getattr(usage_obj, "output_tokens", 0) if usage_obj else 0
    total_tokens = getattr(usage_obj, "total_tokens", 0) if usage_obj else 0
    if _as_int(total_tokens, 0) <= 0:
        total_tokens = _as_int(input_tokens, 0) + _as_int(output_tokens, 0)
    usage_unavailable = usage_obj is None

    return _model_response(
        text=resp.output_text or "",
        usage=_usage_payload(input_tokens, output_tokens, total_tokens),
        usage_unavailable=usage_unavailable,
    )


def call_openai_compat(
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: Optional[int],
    debug: bool = False,
) -> dict:
    if httpx is None:
        raise RuntimeError("httpx is not installed. Install dependencies before calling model APIs.")
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    headers = {"Authorization": f"Bearer {api_key}"}
    safe_headers = {k: _ascii_safe(v) for k, v in headers.items()}

    if debug:
        masked = {k: (_mask_key(v) if k.lower() == "authorization" else v) for k, v in headers.items()}
        print("[DEBUG] POST", url, file=sys.stderr)
        print("[DEBUG] headers:", masked, file=sys.stderr)
        print("[DEBUG] payload:", json.dumps(payload, ensure_ascii=False)[:10000], file=sys.stderr)

    with httpx.Client(timeout=120) as hclient:
        r = hclient.post(url, headers=safe_headers, json=payload)
        r.raise_for_status()
        data = r.json()
    usage, usage_unavailable = _parse_usage_from_openai_compat(data)
    text = data["choices"][0]["message"]["content"]
    return _model_response(text=text, usage=usage, usage_unavailable=usage_unavailable)


def call_gemini(
    model: str,
    api_key: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: Optional[int],
    max_retries: int = 4,
    base_retry_seconds: float = 2.0,
    max_retry_seconds: float = 30.0,
    fail_fast_on_429: bool = False,
    debug: bool = False,
) -> dict:
    if httpx is None:
        raise RuntimeError("httpx is not installed. Install dependencies before calling model APIs.")
    text = f"[SYSTEM]\n{system}\n\n[USER]\n{user}\n"
    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
    }
    if max_tokens is not None:
        payload["generationConfig"]["maxOutputTokens"] = int(max_tokens)

    last_error: Optional[str] = None
    last_rate_limit_error: Optional[str] = None
    saw_rate_limit = False
    tried: list[str] = []
    with httpx.Client(timeout=120) as hclient:
        for candidate in _gemini_alias_candidates(model):
            tried.append(candidate)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{candidate}:generateContent"
            for attempt in range(max(1, int(max_retries)) + 1):
                if debug:
                    print(f"[gemini] request model={candidate} attempt={attempt}", file=sys.stderr)
                r = hclient.post(url, params={"key": api_key}, json=payload)
                if 200 <= r.status_code < 300:
                    data = r.json()
                    usage, usage_unavailable = _parse_usage_from_gemini(data)
                    text_out = data["candidates"][0]["content"]["parts"][0]["text"]
                    return _model_response(text=text_out, usage=usage, usage_unavailable=usage_unavailable)

                body = (r.text or "")[:500]
                last_error = f"status={r.status_code} model={candidate} attempt={attempt} body={body}"

                # Model not found: try next alias.
                if r.status_code == 404:
                    break

                # Quota / transient: retry with backoff, then try next alias.
                if r.status_code in (429, 503):
                    saw_rate_limit = saw_rate_limit or (r.status_code == 429)
                    last_rate_limit_error = f"status={r.status_code} model={candidate} attempt={attempt} body={body}"
                    if r.status_code == 429 and fail_fast_on_429:
                        raise RuntimeError(
                            f"Gemini quota/rate limited (fail-fast). requested={model} model={candidate} "
                            f"attempt={attempt} body={body}"
                        )
                    if attempt >= max(1, int(max_retries)):
                        break

                    retry_after = _retry_after_seconds(getattr(r, "headers", None))
                    backoff = min(float(max_retry_seconds), float(base_retry_seconds) * (2 ** attempt))
                    sleep_s = retry_after if (retry_after is not None and retry_after > 0) else backoff
                    if debug:
                        print(
                            f"[gemini] retry status={r.status_code} model={candidate} "
                            f"attempt={attempt} sleep={sleep_s:.1f}s",
                            file=sys.stderr,
                        )
                    time.sleep(max(0.0, sleep_s))
                    continue

                # Non-retryable errors.
                r.raise_for_status()

    if saw_rate_limit:
        raise RuntimeError(
            f"Gemini rate limit or quota exceeded after retries. requested={model} tried={tried}. "
            f"Rate-limit error: {last_rate_limit_error or last_error}. "
            "Wait and retry, reduce request frequency, or switch to a model/project with available quota."
        )

    raise RuntimeError(
        f"Gemini request failed. requested={model} tried={tried}. "
        f"Last error: {last_error}. "
        "If this is a model-availability issue, set config.yaml -> gemini_models.fast / gemini_models.pro."
    )


def map_model(model_arg: str, cfg: dict) -> tuple[str, str, str]:
    api_key_section = cfg.get("api_key", {}) or {}
    provider = cfg.get("provider", "openai")
    gemini_models = cfg.get("gemini_models", {}) or {}
    gemini_fast = gemini_models.get("fast", "gemini-2.0-flash")
    gemini_pro = gemini_models.get("pro", "gemini-2.5-pro")

    if model_arg == "gpt52":
        return "openai", "gpt-5.2", api_key_section.get("openai", "") or ""
    if model_arg == "gpt4o":
        return "openai", "gpt-4o", api_key_section.get("openai", "") or ""
    if model_arg == "gf":
        return "gemini", str(gemini_fast), api_key_section.get("gemini", "") or ""
    if model_arg == "gp":
        return "gemini", str(gemini_pro), api_key_section.get("gemini", "") or ""
    if model_arg == "l4":
        return "llama_compat", "llama-4", api_key_section.get("llama", "") or ""

    return provider, model_arg, ""


def call_model(
    provider: str,
    model: str,
    api_key: str,
    cfg: dict,
    system: str,
    user: str,
    temperature: float,
    max_tokens: Optional[int],
    debug: bool,
) -> dict:
    if provider in ("openai", "llama_compat"):
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")

    if provider == "openai":
        if not OPENAI_AVAILABLE:
            raise RuntimeError("openai SDK is not installed. Install dependencies before calling model APIs.")
        base_url = cfg.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        client = OpenAI(api_key=api_key, base_url=base_url)
        if model.startswith("gpt-4"):
            return call_openai_gpt4o_chat(client, model, system, user, temperature, max_tokens)
        return call_openai_responses(client, model, system, user, temperature, max_tokens)

    if provider == "llama_compat":
        base_url = cfg.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        return call_openai_compat(base_url, api_key, model, system, user, temperature, max_tokens, debug=debug)

    if provider == "gemini":
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "Missing Gemini API key. Set one of: "
                "config.yaml -> api_key.gemini, "
                "env GEMINI_API_KEY, or env GOOGLE_API_KEY."
            )
        gemini_cfg = cfg.get("gemini", {}) or {}
        return call_gemini(
            model,
            api_key,
            system,
            user,
            temperature,
            max_tokens,
            max_retries=_as_int(gemini_cfg.get("max_retries", 4), 4),
            base_retry_seconds=float(gemini_cfg.get("base_retry_seconds", 2.0)),
            max_retry_seconds=float(gemini_cfg.get("max_retry_seconds", 30.0)),
            fail_fast_on_429=bool(gemini_cfg.get("fail_fast_on_429", False)),
            debug=debug,
        )

    raise ValueError(f"Unknown provider: {provider}")


def write_run_outputs(
    run_dir: Path,
    predictions: list[dict],
    metrics: dict,
    metadata: dict,
    raw_batches: Optional[list[str]] = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Also save a CSV copy for easier manual inspection in spreadsheets.
    pred_df = pd.DataFrame(predictions)
    preferred_cols = ["id", "label", "result", "pred", "reason", "confidence", "text"]
    cols = [c for c in preferred_cols if c in pred_df.columns] + [c for c in pred_df.columns if c not in preferred_cols]
    pred_df = pred_df[cols]
    pred_df.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8")

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if raw_batches:
        raw_df = pd.DataFrame(raw_batches_to_csv_rows(raw_batches))
        raw_df.to_csv(run_dir / "raw_response.csv", index=False, encoding="utf-8")


def run_em_once(
    cfg: dict,
    provider: str,
    model: str,
    api_key: str,
    spec: EMRunSpec,
    out_root: Path,
    debug: bool,
) -> dict:
    system = load_system_prompt(cfg, "entity_matching")
    table_a, table_b, train_df, test_labeled = load_em_tables(cfg)

    sweep_cfg = cfg.get("em_sweep", {})
    key_attrs = sweep_cfg.get("key_attributes", ["Song_Name", "Artist_Name", "Album_Name"])
    display_all_attributes = bool(sweep_cfg.get("display_all_attributes", False))
    include_key_guidance = bool(sweep_cfg.get("include_key_guidance", False))
    relabel_by_selected_keys = bool(sweep_cfg.get("relabel_by_selected_keys", False))
    example_positive_only = bool(sweep_cfg.get("example_positive_only", False))
    example_require_non_key_variation = bool(sweep_cfg.get("example_require_non_key_variation", False))
    manual_key_guidance = sweep_cfg.get("manual_key_guidance_by_k", {}) or {}
    id_col = cfg.get("entity_matching", {}).get("id_col")
    positive_label = int(sweep_cfg.get("positive_label", 1))
    effective_k = len(key_attrs) if display_all_attributes else spec.k

    selected_keys: list[str] = []
    manual_keys = None
    if isinstance(manual_key_guidance, dict):
        manual_keys = manual_key_guidance.get(str(spec.k), manual_key_guidance.get(spec.k))
    if isinstance(manual_keys, list):
        selected_keys = [str(x) for x in manual_keys if str(x).strip()]
    else:
        selected_keys = [str(x) for x in key_attrs[: max(0, int(spec.k))]]

    if include_key_guidance:
        if selected_keys:
            key_line = ", ".join(selected_keys)
        else:
            key_line = "(none)"
        system = (
            system.rstrip()
            + "\n\n"
            + f"Key attributes to prioritize for match decision: {key_line}.\n"
            + "Use the remaining attributes as supporting context."
        )

    sampled_test = sample_by_ratio(test_labeled, true_to_false_ratio=spec.p, seed=spec.seed, positive_label=positive_label)
    if relabel_by_selected_keys and selected_keys:
        train_df = _relabel_pairs_by_keys(train_df, table_a, table_b, selected_keys)
        sampled_test = _relabel_pairs_by_keys(sampled_test, table_a, table_b, selected_keys)

    example_pool_df = _filter_example_pool(
        pairs=train_df,
        table_a=table_a,
        table_b=table_b,
        selected_keys=selected_keys,
        all_attrs=[str(x) for x in key_attrs],
        positive_only=example_positive_only,
        require_non_key_variation=example_require_non_key_variation,
    )
    if len(example_pool_df) < max(1, int(spec.example_num)):
        example_pool_df = train_df

    examples = build_examples(
        train_df=example_pool_df,
        table_a=table_a,
        table_b=table_b,
        key_attributes=key_attrs,
        k=effective_k,
        r_len=spec.r_len,
        example_num=spec.example_num,
        seed=spec.seed,
    )
    cases = build_cases(
        sampled_test=sampled_test,
        table_a=table_a,
        table_b=table_b,
        key_attributes=key_attrs,
        k=effective_k,
        r_len=spec.r_len,
        id_col=id_col,
    )
    user = build_user_prompt(examples, cases)

    batch_size = max(1, int(cfg.get("batch_size", 10)))
    temperature = float(cfg.get("temperature", 0))
    max_tokens = cfg.get("max_tokens", None)

    pred_map: dict[str, dict] = {}
    raw_responses: list[str] = []
    usage_totals = _usage_payload()
    usage_unavailable_any = False
    billing_ctx = _get_billing_context(cfg, model)

    batches = chunk(cases, batch_size)
    for bidx, batch in enumerate(batches):
        user_batch = build_user_prompt(examples, batch)
        if debug:
            print(
                f"[run_em_once] run_id={spec.run_id} batch={bidx+1}/{len(batches)} size={len(batch)} start",
                file=sys.stderr,
            )
        model_resp = call_model(
            provider=provider,
            model=model,
            api_key=api_key,
            cfg=cfg,
            system=system,
            user=user_batch,
            temperature=temperature,
            max_tokens=max_tokens,
            debug=debug,
        )
        text = str(model_resp.get("text", ""))
        usage = model_resp.get("usage", _usage_payload())
        usage_totals["input_tokens"] += _as_int(usage.get("input_tokens", 0), 0)
        usage_totals["output_tokens"] += _as_int(usage.get("output_tokens", 0), 0)
        usage_totals["total_tokens"] += _as_int(usage.get("total_tokens", 0), 0)
        usage_unavailable_any = usage_unavailable_any or bool(model_resp.get("usage_unavailable", False))
        raw_responses.append(text)
        parsed = parse_model_json(text)
        norm = normalize_prediction_map(parsed)
        pred_map.update(norm)
        if debug:
            print(
                f"[run_em_once] run_id={spec.run_id} batch={bidx+1}/{len(batches)} done parsed_items={len(norm)} "
                f"usage_in={usage.get('input_tokens', 0)} usage_out={usage.get('output_tokens', 0)}",
                file=sys.stderr,
            )

    y_true: list[int] = []
    y_pred: list[int] = []
    predictions: list[dict] = []
    for c in cases:
        cid = str(c["id"])
        pred_item = pred_map.get(cid, {"result": -1, "reason": "", "confidence": 0.0})
        pred = int(pred_item.get("result", -1))
        true = int(c["label"])
        predictions.append(
            {
                "id": cid,
                "label": true,
                "result": pred,
                "pred": pred,
                "reason": str(pred_item.get("reason", "")),
                "confidence": _to_confidence(pred_item.get("confidence", 0.0)),
                "text": c["text"],
            }
        )
        if pred in (0, 1):
            y_true.append(true)
            y_pred.append(pred)

    metrics = classification_metrics(y_true, y_pred)
    metrics["coverage"] = (len(y_pred) / len(cases)) if cases else 0.0
    metrics["n_total_cases"] = len(cases)
    metrics["n_predicted"] = len(y_pred)

    run_cost_usd = _estimate_cost_usd(
        input_tokens=usage_totals["input_tokens"],
        output_tokens=usage_totals["output_tokens"],
        billing_ctx=billing_ctx,
    )
    total_budget_usd = billing_ctx.get("total_budget_usd")
    remaining_budget_usd = (float(total_budget_usd) - float(run_cost_usd)) if (total_budget_usd is not None and run_cost_usd is not None) else None

    run_dir = out_root / "entity_matching" / model.replace("/", "_") / spec.run_id
    metadata = {
        "task": "entity_matching",
        "provider": provider,
        "model": model,
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "sweep_axis": spec.sweep_axis,
        "params": {
            "p": spec.p,
            "r_len": spec.r_len,
            "k": spec.k,
            "effective_k_for_prompt": effective_k,
            "selected_key_attributes": selected_keys,
            "example_positive_only": example_positive_only,
            "example_require_non_key_variation": example_require_non_key_variation,
            "example_num": spec.example_num,
        },
        "seed": spec.seed,
        "repeat": spec.repeat,
        "prompt": {
            "system_path": f"{cfg.get('prompts_dir','prompts')}/entity_matching/system.txt",
            "user_builder": "experiment.em_framework.build_user_prompt",
        },
        "data": {
            "tableA_csv": cfg["entity_matching"].get("tableA_csv"),
            "tableB_csv": cfg["entity_matching"].get("tableB_csv"),
            "train_csv": cfg["entity_matching"].get("train_csv"),
            "test_label_csv": cfg["entity_matching"].get("test_label_csv"),
            "sampled_cases": len(cases),
        },
        "usage": {
            "input_tokens": usage_totals["input_tokens"],
            "output_tokens": usage_totals["output_tokens"],
            "total_tokens": usage_totals["total_tokens"],
            "usage_unavailable_any": usage_unavailable_any,
        },
        "billing": {
            "total_budget_usd": total_budget_usd,
            "input_price_usd_per_million": billing_ctx.get("input_price_usd_per_million"),
            "output_price_usd_per_million": billing_ctx.get("output_price_usd_per_million"),
            "run_cost_usd": run_cost_usd,
            "remaining_budget_usd": remaining_budget_usd,
        },
    }

    write_run_outputs(run_dir, predictions, metrics, metadata, raw_batches=raw_responses)
    with (run_dir / "raw_response.txt").open("w", encoding="utf-8") as f:
        f.write("\n\n===== BATCH SPLIT =====\n\n".join(raw_responses))
    with (run_dir / "prompt_user.txt").open("w", encoding="utf-8") as f:
        f.write(user)

    return {
        "run_id": spec.run_id,
        "sweep_axis": spec.sweep_axis,
        "p": spec.p,
        "r_len": spec.r_len,
        "k": spec.k,
        "example_num": spec.example_num,
        "seed": spec.seed,
        "repeat": spec.repeat,
        "metrics": metrics,
        "usage": {
            "input_tokens": usage_totals["input_tokens"],
            "output_tokens": usage_totals["output_tokens"],
            "total_tokens": usage_totals["total_tokens"],
            "usage_unavailable_any": usage_unavailable_any,
        },
        "billing": {
            "run_cost_usd": run_cost_usd,
            "remaining_budget_usd": remaining_budget_usd,
            "total_budget_usd": total_budget_usd,
            "input_price_usd_per_million": billing_ctx.get("input_price_usd_per_million"),
            "output_price_usd_per_million": billing_ctx.get("output_price_usd_per_million"),
        },
        "output_dir": str(run_dir),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", "-task", default="entity_matching", help="entity_matching")
    p.add_argument("--model", "-model", required=True, help="gpt52 / gpt4o / gf / gp / l4 / raw model")

    # Existing single-run args
    p.add_argument("--example", "-example", type=int, default=0)
    p.add_argument("--testcase", "-testcase", type=int, default=20)
    p.add_argument("--example_mode", "-example_mode", default="random", choices=["head", "random"])

    # New sweep args
    p.add_argument("--mode", choices=["single", "sweep"], default="single")
    p.add_argument("--sweep_profile", default="em_ofat_with_examples")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--repeat", type=int, default=None)

    # Optional parameter override for single mode
    p.add_argument("--p", type=float, default=None, help="true:false ratio, e.g. 0.25")
    p.add_argument("--r_len", type=float, default=None, help="query length ratio, e.g. 0.8")
    p.add_argument("--k", type=int, default=None, help="number of key attrs")

    p.add_argument("--debug", "-debug", action="store_true")
    return p.parse_args()


def ensure_default_em_sweep(cfg: dict) -> dict:
    if "em_sweep" not in cfg:
        cfg["em_sweep"] = {
            "levels": {
                "p": [0.25, 0.5, 0.75, 1.0],
                "r_len": [0.2, 0.4, 0.6, 0.8, 1.0],
                "k": [0, 1, 2, 3],
                "example_num": list(range(0, 11)),
            },
            "defaults": {
                "p": 1.0,
                "r_len": 1.0,
                "k": 3,
                "example_num": 0,
            },
            "key_attributes": ["Song_Name", "Artist_Name", "Album_Name"],
            "seeds": [42],
            "n_repeats": 1,
            "positive_label": 1,
        }
    return cfg


def build_single_spec(args: argparse.Namespace, cfg: dict, model: str) -> EMRunSpec:
    defaults = cfg["em_sweep"]["defaults"]
    seed = args.seed if args.seed is not None else int(cfg["em_sweep"].get("seeds", [42])[0])
    repeat = args.repeat if args.repeat is not None else 0

    p = args.p if args.p is not None else float(defaults.get("p", 1.0))
    r_len = args.r_len if args.r_len is not None else float(defaults.get("r_len", 1.0))
    k = args.k if args.k is not None else int(defaults.get("k", 3))

    spec_dict = {
        "axis": "single",
        "p": p,
        "r_len": r_len,
        "k": k,
        "example_num": args.example,
    }
    run_id = build_run_id(spec_dict, model=model, seed=seed, repeat=repeat)
    return EMRunSpec(
        run_id=run_id,
        sweep_axis="single",
        p=p,
        r_len=r_len,
        k=k,
        example_num=args.example,
        seed=seed,
        repeat=repeat,
    )


def write_summary_files(out_root: Path, model: str, records: list[dict]) -> None:
    if not records:
        return

    summary_df = summarize_results(records)
    summary_dir = out_root / "entity_matching" / model.replace("/", "_")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_dir / "summary.csv", index=False)

    agg = (
        summary_df.groupby(["sweep_axis", "example_num"], as_index=False)[["precision", "recall", "f1", "accuracy"]]
        .mean()
        .sort_values(["sweep_axis", "example_num"])
    )
    agg.to_csv(summary_dir / "gain_by_axis_example.csv", index=False)

    with (summary_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main() -> None:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required. Please install dependencies from requirements.txt.") from exc

    args = parse_args()
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    cfg = ensure_default_em_sweep(cfg)

    if args.task != "entity_matching":
        raise ValueError("This implementation currently supports entity_matching only")

    provider, model, api_key = map_model(args.model, cfg)
    out_root = Path(cfg.get("out_dir", "outputs"))

    records: list[dict] = []

    if args.mode == "single":
        spec = build_single_spec(args, cfg, model)
        result = run_em_once(cfg, provider, model, api_key, spec, out_root, args.debug)
        records.append(result)
        write_summary_files(out_root, model, records)
        usage = result.get("usage", {}) or {}
        billing = result.get("billing", {}) or {}
        print(
            f"usage summary | model={model} run_id={spec.run_id} "
            f"input_tokens={usage.get('input_tokens', 0)} "
            f"output_tokens={usage.get('output_tokens', 0)} "
            f"total_tokens={usage.get('total_tokens', 0)} "
            f"cost={_fmt_money(billing.get('run_cost_usd'))} "
            f"remaining={_fmt_money(billing.get('remaining_budget_usd'))} "
            f"usage_unavailable={bool(usage.get('usage_unavailable_any', False))}"
        )
        print(f"Completed single run: {spec.run_id}")
        return

    if args.sweep_profile != "em_ofat_with_examples":
        raise ValueError("Only sweep profile 'em_ofat_with_examples' is supported")

    seeds = [args.seed] if args.seed is not None else list(cfg["em_sweep"].get("seeds", [42]))
    if args.repeat is not None:
        repeats = [args.repeat]
    else:
        n_repeats = int(cfg["em_sweep"].get("n_repeats", 1))
        repeats = list(range(n_repeats))

    for seed in seeds:
        for repeat in repeats:
            specs = generate_em_ofat_specs(cfg["em_sweep"], model=model, seed=int(seed), repeat=int(repeat))
            for spec in specs:
                rec = run_em_once(cfg, provider, model, api_key, spec, out_root, args.debug)
                records.append(rec)
                usage = rec.get("usage", {}) or {}
                billing = rec.get("billing", {}) or {}
                print(
                    f"done {spec.run_id} "
                    f"f1={rec['metrics']['f1']:.4f} acc={rec['metrics']['accuracy']:.4f} "
                    f"coverage={rec['metrics']['coverage']:.4f} "
                    f"tokens={usage.get('total_tokens', 0)} "
                    f"cost={_fmt_money(billing.get('run_cost_usd'))} "
                    f"remaining={_fmt_money(billing.get('remaining_budget_usd'))}"
                )

    write_summary_files(out_root, model, records)
    total_input_tokens = sum(_as_int((r.get("usage", {}) or {}).get("input_tokens", 0), 0) for r in records)
    total_output_tokens = sum(_as_int((r.get("usage", {}) or {}).get("output_tokens", 0), 0) for r in records)
    total_tokens = sum(_as_int((r.get("usage", {}) or {}).get("total_tokens", 0), 0) for r in records)
    usage_unavailable_any = any(bool((r.get("usage", {}) or {}).get("usage_unavailable_any", False)) for r in records)
    total_cost_candidates = [(r.get("billing", {}) or {}).get("run_cost_usd") for r in records]
    total_cost_usd = sum(float(v) for v in total_cost_candidates if v is not None) if all(v is not None for v in total_cost_candidates) else None
    budget = _as_float(((cfg.get("billing", {}) or {}).get("total_budget_usd")))
    remaining_budget_usd = (budget - total_cost_usd) if (budget is not None and total_cost_usd is not None) else None
    print(
        f"sweep usage summary | runs={len(records)} model={model} "
        f"input_tokens={total_input_tokens} output_tokens={total_output_tokens} total_tokens={total_tokens} "
        f"cost={_fmt_money(total_cost_usd)} remaining={_fmt_money(remaining_budget_usd)} "
        f"usage_unavailable={usage_unavailable_any}"
    )
    print(f"Sweep completed. total_runs={len(records)}")


if __name__ == "__main__":
    main()
