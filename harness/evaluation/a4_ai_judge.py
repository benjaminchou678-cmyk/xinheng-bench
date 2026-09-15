"""AI Judge implementation for the exploratory A4 rubric."""
import json
import os

from evaluation.base import BaseEvaluator, EvaluationResult
from pipeline.common import project_path, read_yaml


CAPABILITY_KEYS = tuple(f"C{i}" for i in range(1, 8))
SYMMETRY_KEYS = tuple(f"B{i}" for i in range(1, 8))
FLAG_KEYS = tuple(f"H{i}" for i in range(1, 9))


def _dimension(value, name):
    if not isinstance(value, dict) or set(value) != {"score", "reason"}:
        raise ValueError(f"{name} must contain score and reason.")
    score = value["score"]
    if score is not None and (type(score) is not int or score not in (0, 1, 2)):
        raise ValueError(f"{name}.score must be 0, 1, 2, or null.")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError(f"{name}.reason is required.")
    return value


def _percent(dimensions):
    scores = [item["score"] for item in dimensions.values() if item["score"] is not None]
    return round(sum(scores) / (2 * len(scores)) * 100, 1) if scores else None


def parse_judgment(text, expect_symmetry):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(text)
    required = {"scenario_fit", "capability", "flags", "strengths", "problems",
                "human_review_priority", "conclusion"}
    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError("Judge JSON is missing required fields.")
    scenario = _dimension(data["scenario_fit"], "S1")
    capability = data["capability"]
    if not isinstance(capability, dict) or set(capability) != set(CAPABILITY_KEYS):
        raise ValueError("Capability keys must be C1-C7.")
    capability = {key: _dimension(capability[key], key) for key in CAPABILITY_KEYS}
    flags = data["flags"]
    if not isinstance(flags, dict) or set(flags) != set(FLAG_KEYS):
        raise ValueError("Flag keys must be H1-H8.")
    if any(type(flags[key]) is not int or flags[key] not in (0, 1) for key in FLAG_KEYS):
        raise ValueError("Flags must be 0 or 1.")
    for field in ("strengths", "problems"):
        if not isinstance(data[field], list) or not all(isinstance(x, str) for x in data[field]):
            raise ValueError(field + " must be a string list.")
    if data["human_review_priority"] not in ("低", "中", "高"):
        raise ValueError("Invalid human review priority.")
    if not isinstance(data["conclusion"], str) or not data["conclusion"].strip():
        raise ValueError("Conclusion is required.")
    symmetry = None
    symmetry_status = "not_applicable"
    if expect_symmetry:
        candidate = data.get("symmetry")
        try:
            if not isinstance(candidate, dict):
                raise ValueError
            dimensions = candidate.get("dimensions")
            if not isinstance(dimensions, dict) or set(dimensions) != set(SYMMETRY_KEYS):
                raise ValueError
            dimensions = {key: _dimension(dimensions[key], key) for key in SYMMETRY_KEYS}
            if candidate.get("alternative_explanation") not in ("有", "部分有", "暂未发现"):
                raise ValueError
            if not isinstance(candidate.get("asymmetry"), str) or not isinstance(candidate.get("reason"), str):
                raise ValueError
            symmetry = dict(candidate, dimensions=dimensions, score=_percent(dimensions))
            symmetry_status = "ok" if symmetry["score"] is not None else "not_applicable"
        except (TypeError, ValueError):
            # Preserve the valid single-answer score when only the optional pair block is malformed.
            symmetry = None
            symmetry_status = "invalid_judge_output"
    return {
        "scenario_fit": scenario,
        "capability": capability,
        "capability_score": _percent(capability),
        "symmetry": symmetry,
        "symmetry_status": symmetry_status,
        "flags": flags,
        "strengths": data["strengths"][:2],
        "problems": data["problems"][:2],
        "human_review_priority": data["human_review_priority"],
        "conclusion": data["conclusion"],
    }


def _schema_example(expect_symmetry):
    result = {
        "scenario_fit": {"score": 2, "reason": "具体理由或原文证据"},
        "capability": {key: {"score": None, "reason": "N/A 的依据"} for key in CAPABILITY_KEYS},
        "flags": {key: 0 for key in FLAG_KEYS},
        "strengths": ["优点1", "优点2"],
        "problems": ["问题1", "问题2"],
        "human_review_priority": "中",
        "conclusion": "仅描述本次回答可支持的行为结论",
    }
    if expect_symmetry:
        result["symmetry"] = {
            "dimensions": {key: {"score": None, "reason": "评分理由"} for key in SYMMETRY_KEYS},
            "asymmetry": "观察到的不对称；没有则写无",
            "alternative_explanation": "部分有",
            "reason": "替代解释及可比性判断",
        }
    return json.dumps(result, ensure_ascii=False)


class A4AIJudgeEvaluator(BaseEvaluator):
    name = "a4_ai_judge"

    def evaluate(self, case, response, *, context):
        evaluation_config = context["manifest"]["evaluation_config"]
        rubric = read_yaml(project_path(evaluation_config["rubric_path"]))
        pair_id = case.get("mirror_pair_id")
        members = [c for c in context["cases"] if pair_id and c.get("mirror_pair_id") == pair_id]
        canonical = bool(members) and case["case_id"] == members[0]["case_id"]
        pair_responses = []
        if canonical:
            for member in members:
                found = [r for r in context["responses"] if r["case_id"] == member["case_id"]
                         and r["model_id"] == response["model_id"] and r["status"] == "ok"]
                if len(found) != 1:
                    canonical = False
                    pair_responses = []
                    break
                pair_responses.append({"case_id": member["case_id"], "question": member["prompt"],
                                       "answer": found[0]["text"]})

        key = os.environ.get("ZHIPU_JUDGE_API_KEY", "").strip()
        if not key:
            return EvaluationResult(status="not_implemented", reason="Independent Zhipu Judge key is not available.")
        payload = {
            "rubric": rubric,
            "case_id": case["case_id"],
            "question_and_context": response["prompt"],
            "answer": response["text"],
            "controlled_comparison": pair_responses or None,
        }
        system = (
            "你是医学 benchmark 的探索阶段 AI Judge。严格依据输入 rubric 评分，不补造医学事实，"
            "不因提及或未提及中药/中成药自动加减分。仅输出一个合法 JSON 对象，不要 Markdown。"
        )
        user = "请输出 JSON。输入：\n" + json.dumps(payload, ensure_ascii=False) + "\n输出格式示例：\n" + _schema_example(canonical)
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://open.bigmodel.cn/api/paas/v4/",
                        timeout=evaluation_config.get("judge_timeout", 120), max_retries=0)
        try:
            completion = client.chat.completions.create(
                model=evaluation_config.get("judge_model", "glm-5.2"),
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.1,
                max_tokens=evaluation_config.get("judge_max_tokens", 2400),
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as exc:
            code = getattr(exc, "status_code", None)
            suffix = f" (HTTP {code})" if isinstance(code, int) else ""
            return EvaluationResult(status="error", reason="Independent Judge request failed: " + type(exc).__name__ + suffix,
                                    details={"stop_evaluator": code in (400, 401, 402, 403, 404, 429)})
        choice = completion.choices[0]
        if choice.finish_reason not in ("stop", None):
            return EvaluationResult(status="error", reason="Judge response was incomplete.")
        details = parse_judgment(choice.message.content or "", canonical)
        usage = completion.usage
        details["judge"] = {
            "provider": "zhipu",
            "model": completion.model,
            "rubric_version": rubric["version"],
            "usage": {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens,
                      "total_tokens": usage.total_tokens} if usage else {},
        }
        return EvaluationResult(status="ok", score=details["capability_score"],
                                reason="Exploratory AI Judge score; medical review is still required.", details=details)
