from evaluation.base import BaseEvaluator, EvaluationResult


class ClinicalQualityEvaluator(BaseEvaluator):
    name = "response_quality"

    def evaluate(self, case, response, *, context):
        return EvaluationResult()


class TreatmentVisibilityEvaluator(BaseEvaluator):
    name = "treatment_representation"

    def evaluate(self, case, response, *, context):
        return EvaluationResult()


class TreatmentRoleMatchingEvaluator(BaseEvaluator):
    name = "treatment_role_matching"

    def evaluate(self, case, response, *, context):
        return EvaluationResult()
