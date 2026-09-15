from evaluation.base import BaseEvaluator, EvaluationResult


class BiasGateEvaluator(BaseEvaluator):
    name = "bias_gate"

    def evaluate(self, case, response, *, context):
        return EvaluationResult(reason="Phase 0: bias gate is undefined; no bias conclusion is available.")
