from openai import OpenAI
from typing import Any, Dict, Optional
client = OpenAI()

# 示例命令行（预览prompt拼接效果）：
# python run.py -em -model gpt52 -example 3           # 顺序抽取3条example
# python run.py -em -model gpt52 -example 3 -random   # 随机抽取3条example
# python run.py -em -model gpt52 -manual              # 只用manual（prompts下user.txt）作为example
def format_entity_matching_case(ltable_row: dict, rtable_row: dict) -> str:
    # 拼接格式: Song A，{song_name} ..., {artist_name} ... ... Song B 同理
    def format_row(row, prefix):
        return f"{prefix} " + ", ".join([f"{k}: {v}" for k, v in row.items() if k != "id"])
    return f"{format_row(ltable_row, 'Song A')}; {format_row(rtable_row, 'Song B')}"


def build_prompt_preview(task, n_example=0, example_mode="head", n_testcase=20):
    import pandas as pd
    import yaml
    from pathlib import Path
    import sys
    import random
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    prompts_dir = cfg.get("prompts_dir", "prompts")
    # 1. system
    system_path = Path(prompts_dir) / task / "system.txt"
    system = read_text(str(system_path)) if system_path.exists() else ""

    # 2. example
    examples = []
    if task == "entity_matching":
        em_cfg = cfg["entity_matching"]
        tableA = pd.read_csv(em_cfg["tableA_csv"])
        tableB = pd.read_csv(em_cfg["tableB_csv"])
        train_path = em_cfg.get("train_csv")
        if n_example > 0 and train_path:
            train = pd.read_csv(train_path)
            if example_mode == "random":
                rows = train.sample(n=min(n_example, len(train)), random_state=42).iterrows()
            else:
                rows = train.head(n_example).iterrows()
            for _, row in rows:
                l_id, r_id = int(row['ltable_id']), int(row['rtable_id'])
                l_row = tableA.loc[tableA['id'] == l_id].iloc[0].to_dict()
                r_row = tableB.loc[tableB['id'] == r_id].iloc[0].to_dict()
                label = row.get('label', '')
                formatted = format_entity_matching_case(l_row, r_row)
                examples.append(f"[EXAMPLE] {formatted} (label: {label})")
        elif n_example == 0:
            user_path = Path(prompts_dir) / task / "user.txt"
            if user_path.exists():
                examples.append(read_text(str(user_path)))
    # 3. test case
    cases = []
    if task == "entity_matching":
        em_cfg = cfg["entity_matching"]
        tableA = pd.read_csv(em_cfg["tableA_csv"])
        tableB = pd.read_csv(em_cfg["tableB_csv"])
        test = pd.read_csv(em_cfg["test_csv"])
        for _, row in test.iterrows():
            l_id, r_id = int(row['ltable_id']), int(row['rtable_id'])
            l_row = tableA.loc[tableA['id'] == l_id].iloc[0].to_dict()
            r_row = tableB.loc[tableB['id'] == r_id].iloc[0].to_dict()
            cases.append(format_entity_matching_case(l_row, r_row))
            if len(cases) >= n_testcase:
                break
    # 预览和保存
    import os
    os.makedirs("test", exist_ok=True)
    out_path = os.path.join("test", "preview_cases.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"[SYSTEM]\n{system}\n\n")
        idx = 1
        if n_example > 0:
            for ex in examples:
                f.write(f"Example {idx}:\n{ex}\n\n")
                idx += 1
        for c in cases:
            f.write(f"Case {idx}:\n{c}\n\n")
            idx += 1
    print(f"[SYSTEM]\n{system}\n")
    idx = 1
    if n_example > 0:
        for ex in examples:
            print(f"Example {idx}: {ex}\n")
            idx += 1
    for c in cases:
        print(f"Case {idx}: {c}\n")
        idx += 1
    print(f"\n已保存预览到: {out_path}")

import os
import json
import csv
import time
from pathlib import Path
import sys


import yaml
import pandas as pd
import httpx
from typing import Optional


def read_text(p: str) -> str:
    return Path(p).read_text(encoding="utf-8")


def render(template: str, **kwargs) -> str:
    # 极简模板：只做 {{key}} 替换
    out = template
    for k, v in kwargs.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out


def to_jsonlines_examples(examples: list[dict]) -> str:
    # 用 JSONL 文本塞进 prompt，简单稳
    return "\n".join(json.dumps(e, ensure_ascii=False) for e in examples)


def build_examples(cfg: dict) -> list[dict]:
    ex_cfg = cfg.get("examples", {}) or {}
    examples = []

    # csv examples
    csv_path = ex_cfg.get("csv")
    if csv_path:
        n = int(ex_cfg.get("n", 5))
        df = pd.read_csv(csv_path)
        # 这里不做复杂映射：直接整行作为 input，并要求 examples.csv 里有 label / imputed 等字段
        for i in range(min(n, len(df))):
            row = df.iloc[i].to_dict()
            examples.append(row)

    # manual examples
    for m in ex_cfg.get("manual", []) or []:
        examples.append(m)

    return examples


def build_cases_from_csv(test_csv: str, id_col: str) -> list[dict]:
    df = pd.read_csv(test_csv)
    cases = []
    for i, row in df.iterrows():
        rid = str(row[id_col]) if id_col in df.columns else str(i)
        cases.append({"id": rid, "input": row.to_dict()})
    return cases


def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def _mask_key(v: str) -> str:
    if not v:
        return ""
    s = str(v)
    # show only last 4 characters
    return "****" + (s[-4:] if len(s) > 4 else s)


def _ascii_safe(v: str) -> str:
    if v is None:
        return ""
    return v.encode("ascii", "ignore").decode("ascii")


def call_openai_gpt4o_chat(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: Optional[int],
) -> str:
    kwargs: Dict[str, Any] = {}
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
    return resp.choices[0].message.content


def call_openai_compat(base_url: str, api_key: str, model: str, system: str, user: str,
                      temperature: float, max_tokens: Optional[int], debug: bool = False):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
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
    # make headers ascii-safe for httpx (remove non-ascii), but mask when printing
    safe_headers = {k: _ascii_safe(v) for k, v in headers.items()}
    if debug:
        masked = {k: (_mask_key(v) if k.lower() == "authorization" else v) for k, v in headers.items()}
        print("[DEBUG] POST", url)
        print("[DEBUG] headers:", masked, file=sys.stderr)
        print("[DEBUG] payload:", json.dumps(payload, ensure_ascii=False)[:10000], file=sys.stderr)
    with httpx.Client(timeout=60) as client:
        r = client.post(url, headers=safe_headers, json=payload)
        try:
            r.raise_for_status()
        except Exception:
            if debug:
                # print server response body to help debugging
                try:
                    print("[DEBUG] response status:", r.status_code, file=sys.stderr)
                    print("[DEBUG] response body:", r.text[:20000], file=sys.stderr)
                except Exception:
                    pass
            raise
        data = r.json()
    return data["choices"][0]["message"]["content"]


def call_gemini(model: str, api_key: str, system: str, user: str,
                temperature: float, max_tokens: Optional[int]):
    # 简化：system + user 拼成一个 text
    text = f"[SYSTEM]\n{system}\n\n[USER]\n{user}\n"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {"temperature": temperature},
    }
    if max_tokens is not None:
        payload["generationConfig"]["maxOutputTokens"] = int(max_tokens)

    with httpx.Client(timeout=60) as client:
        r = client.post(url, params={"key": api_key}, json=payload)
        r.raise_for_status()
        data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def load_prompts(prompts_dir: str, task: str):
    base = Path(prompts_dir) / task
    system = read_text(str(base / "system.txt"))
    user_t = read_text(str(base / "user.txt"))
    return system, user_t


def main():

    import sys
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    # 任务选择：em/entity_matching 或 di/data_imputation 或 su/semantic_understanding
    task_arg = None
    for i, arg in enumerate(sys.argv):
        if arg in ("-task", "--task") and i+1 < len(sys.argv):
            task_arg = sys.argv[i+1]
    if not task_arg:
        # 兼容直接 em/di 作为参数
        for arg in sys.argv:
            if arg in ("em", "entity_matching"):
                task_arg = "entity_matching"
            elif arg in ("di", "data_imputation"):
                task_arg = "data_imputation"
            elif arg in ("su", "semantic_understanding"):
                task_arg = "semantic_understanding"
    # 保证 task 一定有值
    if task_arg:
        task = task_arg
    else:
        task = cfg.get("task", "entity_matching")

    # 解析 example 参数和模式
    n_example = 0
    example_mode = "head"
    for i, arg in enumerate(sys.argv):
        if arg == "-example" and i+1 < len(sys.argv):
            try:
                n_example = int(sys.argv[i+1])
            except Exception:
                n_example = 0
        if arg == "-example_mode" and i+1 < len(sys.argv):
            example_mode = sys.argv[i+1]
    # 解析 testcase 参数
    n_testcase = 20
    for i, arg in enumerate(sys.argv):
        if arg == "-testcase" and i+1 < len(sys.argv):
            try:
                n_testcase = int(sys.argv[i+1])
            except Exception:
                n_testcase = 20

    # debug 开关（可打印 payload 和 response body 帮助定位 400）
    debug_payload = False
    for arg in sys.argv:
        if arg in ("-debug", "--debug"):
            debug_payload = True

    # 预览 prompt 拼接
    build_prompt_preview(task, n_example, example_mode, n_testcase)


    # 模型选择与映射
    model_arg = None
    for i, arg in enumerate(sys.argv):
        if arg in ("-model", "--model") and i+1 < len(sys.argv):
            model_arg = sys.argv[i+1]
    if not model_arg:
        # 兼容直接短名参数
        for arg in sys.argv:
            if arg in ("gpt52", "gpt4o", "gf", "gp", "l4"):
                model_arg = arg
    # 必须指定模型，否则报错
    if not model_arg:
        print("[Error] Missing model. Please specify with -model gpt52/gpt4o/gf/gp/l4.", file=sys.stderr)
        sys.exit(1)
    provider = cfg["provider"]
    model = cfg["model"]
    api_key_section = cfg.get("api_key", {})
    if model_arg == "gpt52":
        model = "gpt-5.2"
        provider = "openai"
        api_key = api_key_section.get("openai", "")
    elif model_arg == "gpt4o":
        model = "gpt-4o"
        provider = "openai"
        api_key = api_key_section.get("openai", "")
    elif model_arg == "gf":
        model = "gemini-1.5-flash"
        provider = "gemini"
        api_key = api_key_section.get("gemini", "")
    elif model_arg == "gp":
        model = "gemini-1.5-pro"
        provider = "gemini"
        api_key = api_key_section.get("gemini", "")
    elif model_arg == "l4":
        model = "llama-4"
        provider = "llama_compat"
        api_key = api_key_section.get("llama", "")
    else:
        model = model_arg
        api_key = ""

    prompts_dir = cfg.get("prompts_dir", "prompts")

    system, user_template = load_prompts(prompts_dir, task)

    examples = build_examples(cfg)
    if task == "entity_matching":
        cases = build_cases_from_csv(cfg["entity_matching"]["test_csv"], cfg["entity_matching"].get("id_col", "id"))
    elif task == "data_imputation":
        cases = build_cases_from_csv(cfg["data_imputation"]["test_csv"], cfg["data_imputation"].get("id_col", "id"))
    else:
        raise ValueError(f"Unknown task: {task}")

    out_dir = Path(cfg.get("out_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pred_{task}_{provider}_{model.replace('/', '_')}.jsonl"

    # 输出格式控制（jsonl/csv/json/dir）
    out_format = "jsonl"
    for i, arg in enumerate(sys.argv):
        if arg == "-out_format" and i+1 < len(sys.argv):
            out_format = sys.argv[i+1].lower()

    batch_size = int(cfg.get("batch_size", 1))
    temperature = float(cfg.get("temperature", 0))
    max_tokens = cfg.get("max_tokens", None)

    results = []
    for b in chunk(cases, batch_size):
        examples_str = to_jsonlines_examples(examples)
        cases_str = to_jsonlines_examples(b)

        user = render(user_template, examples=examples_str, cases=cases_str)

        if provider in ("openai", "llama_compat"):
            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY (not found in config.yaml or environment)")
            base_url = cfg.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            if model_arg == "gpt4o":
                text = call_openai_gpt4o_chat(client, model, system, user, temperature, max_tokens)
            elif model_arg == "gpt52":
                text = call_openai_gpt52_responses(client, model, system, user, temperature, max_tokens)
            else:
                text = call_openai_compat(base_url, api_key, model, system, user, temperature, max_tokens, debug=debug_payload)

        elif provider == "gemini":
            if not api_key:
                api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                raise RuntimeError("Missing GEMINI_API_KEY (not found in config.yaml or environment)")
            text = call_gemini(model, api_key, system, user, temperature, max_tokens)

        else:
            raise ValueError(f"Unknown provider: {provider}")

        # 不做严格安全解析：尽量 json.loads，失败就原样存
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {"_raw": text}

        # accumulate results
        if isinstance(parsed, list):
            by_id = {str(x.get("id")): x for x in parsed if isinstance(x, dict) and "id" in x}
            for c in b:
                results.append({
                    "id": c["id"],
                    "pred": by_id.get(str(c["id"]), {"_missing": True}),
                })
        else:
            if len(b) == 1:
                results.append({"id": b[0]["id"], "pred": parsed})
            else:
                for c in b:
                    results.append({"id": c["id"], "pred": {"_batch_not_list": True, "raw": parsed}})

        time.sleep(0.0)

    # write outputs in requested format
    if out_format == "jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"saved: {out_path}")

    elif out_format == "csv":
        csv_path = out_dir / f"pred_{task}_{provider}_{model.replace('/', '_')}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "pred_json"])
            for r in results:
                w.writerow([r.get("id"), json.dumps(r.get("pred"), ensure_ascii=False)])
        print(f"saved: {csv_path}")

    elif out_format == "json":
        json_path = out_dir / f"pred_{task}_{provider}_{model.replace('/', '_')}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"saved: {json_path}")

    elif out_format == "dir":
        dirp = out_dir / f"preds_{task}_{provider}_{model.replace('/', '_')}"
        dirp.mkdir(parents=True, exist_ok=True)
        for r in results:
            fid = str(r.get("id"))
            with open(dirp / f"{fid}.json", "w", encoding="utf-8") as f:
                json.dump(r.get("pred"), f, ensure_ascii=False, indent=2)
        print(f"saved dir: {dirp}")

    else:
        # fallback to jsonl
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"saved: {out_path} (fallback)")


def call_openai_gpt52_responses(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_output_tokens: Optional[int],
) -> str:
    kwargs: Dict[str, Any] = {}
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = int(max_output_tokens)

    # 修复 prompt 参数为消息对象列表
    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    resp = client.responses.create(
        model=model,
        instructions=system,
        input=user,
        temperature=temperature,
        prompt=prompt,  # 修复为对象列表
        **kwargs,
    )
    # SDK 提供 output_text，省去自己解析 output 数组
    return resp.output_text


if __name__ == "__main__":
    main()
