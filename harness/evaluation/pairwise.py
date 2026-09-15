from evaluation.base import BaseEvaluator, EvaluationResult


class MirrorPairEvaluator(BaseEvaluator):
    name = "mirror_pair"

    def evaluate(self, case, response, *, context):
        pair_id = case.get("mirror_pair_id")
        if pair_id is None:
            return EvaluationResult(status="not_applicable", reason="No mirror_pair_id.")
        members = [c["case_id"] for c in context["cases"] if c.get("mirror_pair_id") == pair_id]
        return EvaluationResult(details={"mirror_pair_id": pair_id, "member_case_ids": members})
