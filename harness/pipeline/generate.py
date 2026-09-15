from uuid import uuid4
from models.registry import create_model
from pipeline.common import (artifact, now, project_path, read_yaml, write_json, write_jsonl,
                             validate_rows, digest)
from pipeline.load_cases import load_cases


def build_prompt(case, config):
    path = project_path(config["prompt_dir"]) / case["prompt_type"] / "template.txt"
    # Only the template is formatted; braces inside case text remain literal.
    return path.read_text(encoding="utf-8").format(prompt=case["prompt"])


def run(config, run_id):
    target = artifact(config, "raw_responses", run_id, "responses.jsonl")
    cases = load_cases(config)
    model_config = read_yaml(project_path(config["models_config"]))
    evaluation_config = read_yaml(project_path(config["evaluation_config"]))
    specs = model_config["models"]
    if not specs or len({s["id"] for s in specs}) != len(specs):
        raise ValueError("Provide at least one model with unique model ids.")
    if config.get("phase") == "pilot" and len(specs) != 1:
        raise ValueError("Initial pilot supports exactly one model per run.")
    models = [create_model(spec) for spec in specs]
    prompts = {c["case_id"]: build_prompt(c, config) for c in cases}
    template_paths = {c["prompt_type"]: project_path(config["prompt_dir"]) / c["prompt_type"] / "template.txt" for c in cases}
    target.parent.mkdir(parents=True, exist_ok=False)
    write_json(target.parent / "manifest.json", {
        "run_id": run_id, "created_at": now(), "harness_version": "0.1.1",
        "benchmark_config": config, "models_config": model_config,
        "evaluation_config": evaluation_config, "case_sha256": digest(project_path(config["cases"])),
        "template_sha256": {name: digest(path) for name, path in template_paths.items()},
    })
    write_jsonl(target.parent / "cases.jsonl", cases)
    responses = []
    session_history = {}
    message_history = {}
    completed_by_case = {}
    stop_reason = None
    response_instruction = config.get("response_instruction", "").strip()
    for case in cases:
        for model in models:
            current_prompt = prompts[case["case_id"]]
            prompt = current_prompt
            session_id = case.get("metadata", {}).get("session_id")
            parent_case_id = case.get("metadata", {}).get("parent_case_id")
            parent = completed_by_case.get((parent_case_id, model.model_id)) if parent_case_id else None
            parent_failed = bool(parent_case_id and (not parent or parent.get("status") != "ok"))
            if parent:
                parent_prompt = prompts[parent_case_id]
                prompt = "\n\n".join(["用户：\n" + parent_prompt,
                                      "助手：\n" + parent["text"], "用户：\n" + current_prompt])
            elif session_id and session_history.get(session_id):
                prompt = "\n\n".join(session_history[session_id] + ["用户：\n" + current_prompt])
            if response_instruction:
                prompt = response_instruction + "\n\n" + prompt
            messages = message_history.get((session_id, model.model_id), []) + [{"role": "user", "content": current_prompt}]
            if parent and not parent_failed and config.get("native_messages"):
                # Fork from the exact saved parent transcript, never from a sibling branch.
                parent_messages = parent.get("metadata", {}).get("request_messages")
                if parent_messages is None:
                    raise ValueError("Native-message parent transcript is missing.")
                messages = parent_messages + [{"role": "assistant", "content": parent["text"]},
                                               {"role": "user", "content": current_prompt}]
            if config.get("native_messages") and parent:
                # Build from the explicit parent, not the last processed sibling.
                parent_messages = parent.get("metadata", {}).get("request_messages")
                if parent_messages is None:
                    parent_messages = [{"role": "user", "content": prompts[parent_case_id]}]
                messages = list(parent_messages) + [
                    {"role": "assistant", "content": parent["text"]},
                    {"role": "user", "content": current_prompt},
                ]
            row = dict(schema_version="0.1.0", run_id=run_id, response_id=uuid4().hex,
                       case_id=case["case_id"], model_id=model.model_id, created_at=now(), prompt=prompt)
            if parent_failed:
                row.update(text="", status="error", error="Not attempted because the branch parent response is unavailable.",
                           metadata={"request_attempted": False})
            elif stop_reason:
                row.update(text="", status="error", error=stop_reason,
                           metadata={"request_attempted": False})
            else:
                try:
                    result = model.generate(prompt, messages=messages) if config.get("native_messages") else model.generate(prompt)
                    if config.get("native_messages"):
                        result.metadata["request_messages"] = messages
                    row.update(text=result.text, status=result.status, error=result.error, metadata=result.metadata)
                    if result.metadata.get("stop_run"):
                        stop_reason = "Not attempted after provider authentication, permission, or quota failure."
                    validate_rows([row], project_path(config["response_schema"]))
                except Exception as exc:
                    row.update(text="", status="error", error="Generation failed: " + type(exc).__name__, metadata={})
            if config.get("phase") == "pilot":
                print(f"{config.get('progress_label', model.model_id)} · {case['case_id']} · {len(responses)+1}/{len(cases)}: {row['status']}", flush=True)
            responses.append(row)
            completed_by_case[(case["case_id"], model.model_id)] = row
            if session_id:
                if row["status"] == "ok":
                    message_history[(session_id, model.model_id)] = messages + [{"role": "assistant", "content": row["text"]}]
                session_history.setdefault(session_id, []).append("用户：\n" + current_prompt)
                if row["status"] == "ok" and row["text"]:
                    session_history[session_id].append("助手：\n" + row["text"])
            write_jsonl(target, responses)
    validate_rows(responses, project_path(config["response_schema"]))
    write_jsonl(target, responses)
    return target
