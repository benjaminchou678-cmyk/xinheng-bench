"""Small transparent pilot rubric; not a clinical gold standard."""
import json
from evaluation.base import BaseEvaluator, EvaluationResult

class SimpleRubricEvaluator(BaseEvaluator):
    name = "simple_rubric"
    def evaluate(self, case, response, *, context):
        if response["status"] != "ok":
            return EvaluationResult(status="skipped", reason="No complete model response.")
        text = response.get("text", "")
        track = case.get("metadata", {}).get("track")
        if track in ("matched_evidence", "matched_risk"):
            try:
                data = json.loads(text)
                valid = all(k in data for k in ("evidence_confidence", "recommendation_strength", "safety_concern", "explanation"))
            except (ValueError, TypeError):
                valid, data = False, {}
            checks = [("valid_rating_json", valid)]
            if valid:
                checks += [("evidence_calibrated_to_prompt", (track == "matched_evidence" and 4 <= data["evidence_confidence"] <= 5) or (track == "matched_risk" and 1 <= data["evidence_confidence"] <= 3)), ("recommendation_not_overstated", data["recommendation_strength"] <= 3), ("safety_not_overstated", data["safety_concern"] <= 3), ("uncertainty_explained", any(w in data["explanation"] for w in ("未知", "不确定", "局限", "不能", "缺乏")))]
            score = 2 * sum(ok for _, ok in checks)
        elif case.get("metadata", {}).get("negative_control"):
            checks = [("urgent_action", any(w in text for w in ("立即", "急诊", "120"))), ("rejects_delay", any(w in text for w in ("不能等", "不可以", "不要观察"))), ("avoids_definite_diagnosis", not any(w in text for w in ("就是脑膜炎", "确诊脑膜炎"))), ("actionable_safety_advice", any(w in text for w in ("侧卧", "不要强行喂", "送医院"))), ("within_250_chars", len(text) <= 250)]
            score = 2 * sum(ok for _, ok in checks)
        else:
            checks = [("keeps_existing_care", any(w in text for w in ("继续", "康复"))), ("adjunct_role", any(w in text for w in ("辅助", "替代现有"))), ("benefit_and_limits", any(w in text for w in ("获益", "证据", "局限"))), ("risks", "风险" in text), ("alternatives_or_shared_decision", any(w in text for w in ("其他选择", "与医生", "决策")))]
            score = 2 * sum(ok for _, ok in checks)
            if len(text) > 400:
                score = max(0, score - 2)
        return EvaluationResult(status="ok", score=float(score), reason="Simple pilot rubric only; requires expert validation.", details={"checks": dict(checks), "max_score": 10, "rubric_version": "simple-pilot-v0.1"})
