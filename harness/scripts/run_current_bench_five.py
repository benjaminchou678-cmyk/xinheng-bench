"""Run the current normalized benchmark on five providers concurrently, without scoring."""
import argparse
import getpass
import html
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import generate
from pipeline.common import ROOT, digest, load_config, read_jsonl, read_yaml, write_json, write_jsonl
from scripts.import_current_bench import import_file
from scripts.run_latest_six_models import PROVIDERS as ALL_PROVIDERS


PROVIDERS = [item for item in ALL_PROVIDERS if item[1] != "baichuan"]
SOURCE = ROOT / "configs/bench.md"


def write_reports(folder, cases, results, states):
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    expected = len(cases) * len(PROVIDERS)
    ok = sum(row.get("status") == "ok" for rows in results.values() for row in rows)
    title = "当前 Benchmark · 五模型原始回答"
    note = f"更新时间：{stamp}；成功回答 {ok}/{expected}。五模型并行、各模型题内顺序执行；不评分。"
    labels = {provider: label for label, provider, _, _ in PROVIDERS}
    by_provider = {p: {r["case_id"]: r for r in results.get(p, [])} for p in labels}
    md = ["# " + title, "", note]
    html_sections = []
    for case in cases:
        qid = case["metadata"]["source_question_id"]
        md.extend(["", "## 问题 " + qid, "", "### 题目", "", case["prompt"]])
        cards = []
        for provider, label in labels.items():
            row = by_provider[provider].get(case["case_id"], {})
            answer = row.get("text") or "尚无回答"
            status = row.get("status") or states[provider]
            error = row.get("error") or ""
            md.extend(["", "### " + label, "", "状态：" + status, "", answer])
            if error:
                md.extend(["", "错误：" + error])
            cards.append("<article><h3>" + html.escape(label) + "</h3><p>" + html.escape(status) +
                         "</p><pre>" + html.escape(answer) + "</pre>" +
                         (("<p class=error>错误：" + html.escape(error) + "</p>") if error else "") + "</article>")
        html_sections.append("<section><h2>问题 " + html.escape(qid) + "</h2><details><summary>查看题目</summary><pre>" +
                             html.escape(case["prompt"]) + "</pre></details><div class=grid>" + "".join(cards) + "</div></section>")
    (folder / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    style = "body{font:15px/1.7 system-ui;margin:24px;background:#f4f6f9;color:#182536}main{max-width:1600px;margin:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:16px}section{margin:32px 0}article,details{background:#fff;padding:18px;border-radius:12px}pre{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere}.error{color:#a21}.meta{position:sticky;top:0;background:#f4f6f9;padding:8px 0}"
    page = "<!doctype html><html lang=zh-CN><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>" + title + "</title><style>" + style + "</style><main><h1>" + title + "</h1><p class=meta>" + html.escape(note) + "</p>" + "".join(html_sections) + "</main></html>"
    (folder / "report.html").write_text(page, encoding="utf-8")
    write_json(folder / "progress.json", {"updated_at": stamp, "successful": ok, "expected": expected, "states": states})


def execute(provider, config, run_id):
    started = time.monotonic()
    try:
        rows = read_jsonl(generate.run(config, run_id))
    except Exception as exc:
        rows = [{"case_id": case["case_id"], "status": "error", "text": "",
                 "error": "Runner failed: " + type(exc).__name__, "metadata": {}}
                for case in read_jsonl(config["cases"])]
    return rows, round(time.monotonic() - started, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    cases = import_file(SOURCE, ROOT / "datasets/cases/current_bench.jsonl")
    configs = []
    for label, provider, env, path in PROVIDERS:
        config = load_config(ROOT / path)
        spec = read_yaml(ROOT / config["models_config"])
        configs.append((label, provider, env, config, spec))
    if args.check:
        print(f"检查通过：{len(cases)}题；5模型；共{len(cases)*5}次请求；4个追问继承指定父轮；镜像题独立；不评分。")
        print("模型：" + ", ".join(spec["models"][0]["parameters"]["model"] for _, _, _, _, spec in configs))
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(os.getpid())
    folder = ROOT / "outputs/reports" / ("current-bench-five-" + stamp)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "bench.md").write_bytes(SOURCE.read_bytes())
    write_jsonl(folder / "cases.jsonl", cases)
    (folder / "prompts/neutral").mkdir(parents=True)
    (folder / "prompts/neutral/template.txt").write_text("{prompt}\n", encoding="utf-8")
    inserted = {}
    results = {}
    states = {p: "等待密钥输入" for _, p, _, _, _ in configs}
    pool = None
    write_reports(folder, cases, results, states)
    try:
        print(f"新版 Benchmark：{len(cases)}题，五模型共{len(cases)*5}次请求，不评分。", flush=True)
        print("请依次粘贴五个 API Key；输入隐藏且不保存。收齐后五模型并行运行。", flush=True)
        for label, provider, env, config, spec in configs:
            while not os.environ.get(env, "").strip():
                key = getpass.getpass(label + " API Key（隐藏输入）：").strip()
                if not key or any(ch.isspace() for ch in key) or (len(key) % 2 == 0 and key[:len(key)//2] == key[len(key)//2:]):
                    print("输入为空、包含空白或疑似重复粘贴，请重新输入。", flush=True)
                    continue
                inserted[env] = os.environ.get(env)
                os.environ[env] = key
                del key
            params = spec["models"][0]["parameters"]
            params.update(max_output_tokens=None, timeout=None)
            if provider == "doubao":
                default = os.environ.get("ARK_MODEL_ID") or "ep-20260911140941-pdkdz"
                endpoint = input("豆包接入点 ID（回车沿用 " + default + "）：").strip() or default
                inserted["ARK_MODEL_ID"] = os.environ.get("ARK_MODEL_ID")
                os.environ["ARK_MODEL_ID"] = endpoint
                # A long benchmark can cross Ark's request-per-minute boundary.
                # Retry transient 429 responses with increasing waits instead of
                # treating the rest of the model run as permanently unavailable.
                params.update(model=endpoint, rate_limit_retries=6, retry_wait_seconds=20)
            for name, model_env in (("kimi", "KIMI_MODEL"), ("hunyuan", "HUNYUAN_MODEL")):
                if provider == name and os.environ.get(model_env):
                    params["model"] = os.environ[model_env].strip()
            write_json(folder / ("models." + provider + ".json"), spec)
            config.update(cases=str(folder / "cases.jsonl"), models_config=str(folder / ("models." + provider + ".json")),
                          prompt_dir=str(folder / "prompts"), native_messages=True, response_instruction="", pilot_case_limit=None)
        write_json(folder / "manifest.json", {"created_at": datetime.now().astimezone().isoformat(),
                   "bench_sha256": digest(folder / "bench.md"), "providers": [p for _, p, _, _, _ in configs],
                   "question_count": len(cases), "api_requests_planned": len(cases) * 5, "scoring": False,
                   "parallel_providers": True, "follow_up_edges": 4, "mirror_questions_independent": True})
        states = {p: "请求中" for _, p, _, _, _ in configs}
        write_reports(folder, cases, results, states)
        print("开始五模型并行测试。运行中报告：" + str(folder / "report.html"), flush=True)
        pool = ThreadPoolExecutor(max_workers=5)
        pending = {pool.submit(execute, p, config, "current-" + stamp + "-" + p): (label, p)
                   for label, p, _, config, _ in configs}
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                label, provider = pending.pop(future)
                rows, elapsed = future.result()
                results[provider] = rows
                write_jsonl(folder / (provider + "-responses.jsonl"), rows)
                count = sum(row.get("status") == "ok" for row in rows)
                states[provider] = f"完成 {count}/{len(cases)}"
                write_json(folder / (provider + "-timing.json"), {"elapsed_seconds": elapsed})
                write_reports(folder, cases, results, states)
                print(f"{label}：{states[provider]}；耗时 {elapsed:.0f} 秒。", flush=True)
            if pending:
                print("仍在运行：" + "、".join(label for label, _ in pending.values()), flush=True)
        print("测试结束。报告：" + str(folder / "report.html"), flush=True)
        return int(any(not state.startswith("完成 28/28") for state in states.values()))
    except (KeyboardInterrupt, EOFError):
        write_reports(folder, cases, results, states)
        print("\n已中断；已有结果已保存，密钥未保存。", flush=True)
        if pool is not None:
            os._exit(130)
        return 130
    finally:
        if pool is not None:
            pool.shutdown(wait=False)
        for name, value in inserted.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    raise SystemExit(main())
