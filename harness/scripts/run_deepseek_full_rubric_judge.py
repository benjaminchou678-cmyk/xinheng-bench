"""Judge the current five-model benchmark from scratch with DeepSeek."""
import argparse
import getpass
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.common import ROOT, write_json
from scripts.run_glm_healthbench_bias import judge_unit, load_scorer, write_report


RUBRIC_ROOT = ROOT / "rubrics/healthbench_bias_v1"
SOURCE_POINTER = ROOT / "outputs/reports/latest_updated_bench_run.txt"
SUCCESS_POINTER = ROOT / "outputs/reports/latest_deepseek_judge_run.txt"
FAILED_POINTER = ROOT / "outputs/reports/latest_deepseek_judge_failed_run.txt"
PROVIDERS = [
    ("DeepSeek", "deepseek"),
    ("MiniMax", "minimax"),
    ("豆包", "doubao"),
    ("混元", "hunyuan"),
    ("Kimi", "kimi"),
]
JUDGE_NOTE = "DeepSeek同时评价DeepSeek自身回答，可能存在自评偏好；应与独立裁判或人工盲评结果并列解释。"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_run():
    if not SOURCE_POINTER.exists():
        raise ValueError("未找到当前五模型完整回答目录。")
    path = Path(SOURCE_POINTER.read_text(encoding="utf-8").strip())
    if not path.is_dir():
        raise ValueError("当前五模型回答目录不存在。")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    run = source_run()
    config_path = RUBRIC_ROOT / "configs/bias_healthbench_v1.json"
    prompt_path = RUBRIC_ROOT / "docs/grader_prompt_zh.md"
    scorer_path = RUBRIC_ROOT / "evaluation/score_bias_healthbench.py"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256(ROOT / "configs/bench.md") != config["source"]["sha256"]:
        raise ValueError("当前bench与rubric指纹不一致。")
    if sha256(run / "bench.md") != config["source"]["sha256"]:
        raise ValueError("五模型回答所用bench与rubric指纹不一致。")
    ids = [criterion["id"] for unit in config["units"] for criterion in unit["criteria"]]
    if len(config["units"]) != 11 or len(ids) != 50 or len(ids) != len(set(ids)):
        raise ValueError("rubric必须包含11个单元和50个唯一条目。")

    cases = read_jsonl(run / "cases.jsonl")
    case_by_qid = {case["metadata"]["source_question_id"]: case for case in cases}
    model_rows = {}
    for _, provider in PROVIDERS:
        rows = read_jsonl(run / f"{provider}-responses.jsonl")
        if len(rows) != 28 or not all(row.get("status") == "ok" for row in rows):
            raise ValueError(provider + "回答不是28/28完整状态。")
        model_rows[provider] = {row["case_id"]: row["text"] for row in rows}

    print("检查通过：DeepSeek从头评审5模型×11单元，共55次请求；不复用任何既有裁判结果。", flush=True)
    if args.check:
        return 0

    secret = getpass.getpass("DeepSeek API Key（隐藏输入且不保存）：").strip()
    if not secret or any(ch.isspace() for ch in secret):
        print("密钥为空或包含空白，已取消。", flush=True)
        return 1
    model = input("DeepSeek Judge模型ID（回车使用 deepseek-v4-flash）：").strip() or "deepseek-v4-flash"
    from openai import OpenAI
    client = OpenAI(api_key=secret, base_url="https://api.deepseek.com", timeout=240, max_retries=0)
    del secret

    print("正在预检DeepSeek Key、接口和模型权限……", flush=True)
    try:
        probe = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "只回复OK"}],
            temperature=0,
            max_tokens=8,
            stream=False,
        )
        if not getattr(probe, "choices", None):
            raise ValueError("预检返回空choices")
    except Exception as exc:
        code = getattr(exc, "status_code", None)
        suffix = f"（HTTP {code}）" if isinstance(code, int) else ""
        print("预检失败：" + type(exc).__name__ + suffix, flush=True)
        print("请确认该Key属于api.deepseek.com，并确认账户有权限调用模型：" + model, flush=True)
        return 2
    print("预检成功，开始五模型并行全量评分。", flush=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(os.getpid())
    folder = run / ("deepseek-full-judge-" + stamp)
    folder.mkdir(parents=True)
    snapshot = folder / "rubric_snapshot"
    snapshot.mkdir()
    for source in (config_path, prompt_path, RUBRIC_ROOT / "docs/bias_healthbench_v1.md", scorer_path, ROOT / "configs/bench.md"):
        shutil.copy2(source, snapshot / source.name)
    write_json(folder / "manifest.json", {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_run": str(run),
        "judge_provider": "deepseek",
        "judge_model": model,
        "planned_calls": 55,
        "reused_calls": 0,
        "parallel_model_lanes": 5,
        "rubric_sha256": sha256(config_path),
        "bench_sha256": sha256(run / "bench.md"),
        "rubric_version": config["rubric_version"],
        "methodological_limit": JUDGE_NOTE,
    })

    grader_prompt = prompt_path.read_text(encoding="utf-8")
    score_fn = load_scorer(scorer_path)
    statuses = {provider: "请求中 0/11" for _, provider in PROVIDERS}
    scores = {}
    write_report(folder, config, scores, statuses, model, judge_label="DeepSeek", extra_note=JUDGE_NOTE)

    def run_provider(label, provider):
        units = []
        errors = 0
        for index, unit in enumerate(config["units"], 1):
            questions = {qid: case_by_qid[qid]["prompt"] for qid in unit["question_ids"]}
            responses = {qid: model_rows[provider][case_by_qid[qid]["case_id"]] for qid in unit["question_ids"]}
            result, status = judge_unit(
                client, model, grader_prompt, unit, questions, responses,
                folder / f"{provider}-{unit['id']}-judge.json",
            )
            units.append(result)
            errors += status != "ok"
            statuses[provider] = f"请求中 {index}/11（校验失败 {errors}）"
            write_json(folder / f"{provider}-progress.json", {
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "completed_units": index,
                "total_units": 11,
                "validation_failures": errors,
                "last_unit": unit["id"],
            })
            write_json(folder / f"{provider}-judgments.partial.json", {
                "rubric_version": config["rubric_version"],
                "source_sha256": config["source"]["sha256"],
                "model_id": provider,
                "run_id": run.name,
                "units": units,
            })
            print(f"{label} · {unit['id']} · {index}/11：{status}", flush=True)

        data = {
            "rubric_version": config["rubric_version"],
            "source_sha256": config["source"]["sha256"],
            "model_id": provider,
            "run_id": run.name,
            "units": units,
        }
        write_json(folder / f"{provider}-judgments.json", data)
        score = score_fn(config, data)
        write_json(folder / f"{provider}-score.json", score)
        return score, errors

    pool = ThreadPoolExecutor(max_workers=5)
    try:
        pending = {pool.submit(run_provider, label, provider): (label, provider) for label, provider in PROVIDERS}
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                label, provider = pending.pop(future)
                try:
                    score, errors = future.result()
                    scores[provider] = score
                    statuses[provider] = f"完成 11/11（校验失败 {errors}）"
                except Exception as exc:
                    statuses[provider] = "运行失败：" + type(exc).__name__
                    write_json(folder / f"{provider}-fatal.json", {"error_type": type(exc).__name__, "error": str(exc)})
                write_report(folder, config, scores, statuses, model, judge_label="DeepSeek", extra_note=JUDGE_NOTE)
                print(label + "：" + statuses[provider], flush=True)
            if pending:
                print("仍在评分：" + "、".join(label for label, _ in pending.values()), flush=True)
    except (KeyboardInterrupt, EOFError):
        write_report(folder, config, scores, statuses, model, judge_label="DeepSeek", extra_note=JUDGE_NOTE)
        print("\n已中断；现有结果已保存，密钥未保存。", flush=True)
        os._exit(130)
    finally:
        pool.shutdown(wait=False)

    write_report(folder, config, scores, statuses, model, judge_label="DeepSeek", extra_note=JUDGE_NOTE)
    complete = all(state == "完成 11/11（校验失败 0）" for state in statuses.values())
    pointer = SUCCESS_POINTER if complete else FAILED_POINTER
    pointer.write_text(str(folder) + "\n", encoding="utf-8")
    print(("评分完成。" if complete else "评分未完整成功。") + "报告：" + str(folder / "report.html"), flush=True)
    return int(not complete)


if __name__ == "__main__":
    raise SystemExit(main())
