from evaluation.rubric import (ClinicalQualityEvaluator, TreatmentVisibilityEvaluator,
                               TreatmentRoleMatchingEvaluator)
from evaluation.pairwise import MirrorPairEvaluator
from evaluation.bias_gate import BiasGateEvaluator

from evaluation.pilot import PilotObservationEvaluator
from evaluation.simple_rubric import SimpleRubricEvaluator
from evaluation.a4_ai_judge import A4AIJudgeEvaluator

EVALUATORS = {cls.name: cls for cls in [ClinicalQualityEvaluator, TreatmentVisibilityEvaluator,
             TreatmentRoleMatchingEvaluator, MirrorPairEvaluator, BiasGateEvaluator,
             PilotObservationEvaluator, SimpleRubricEvaluator, A4AIJudgeEvaluator]}


def register_evaluator(name, evaluator_class):
    if name in EVALUATORS:
        raise ValueError("Evaluator already registered: " + name)
    EVALUATORS[name] = evaluator_class


def create_evaluator(name):
    if name not in EVALUATORS:
        raise ValueError("Unknown evaluator: " + name)
    return EVALUATORS[name]()
