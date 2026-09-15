from dataclasses import asdict
from evaluation.base import EvaluationResult
from evaluation.registry import create_evaluator
from pipeline.common import artifact, read_json, read_jsonl, write_jsonl, validate_rows, project_path


def run(config, run_id):
    source = artifact(config, "raw_responses", run_id, "responses.jsonl")
    cases = read_jsonl(source.parent / "cases.jsonl")
    responses = read_jsonl(source)
    validate_rows(cases, project_path(config["case_schema"]))
    validate_rows(responses, project_path(config["response_schema"]))
    manifest = read_json(source.parent / "manifest.json")
    evaluators = [create_evaluator(name) for name in manifest["evaluation_config"]["evaluators"]]
    by_id = {case["case_id"]: case for case in cases}
    context = {"cases": cases, "responses": responses, "manifest": manifest}
    annotations = []
    stopped_evaluators = set()
    for response in responses:
        if response["run_id"] != run_id or response["case_id"] not in by_id:
            raise ValueError("Response does not belong to this run or case snapshot.")
        for evaluator in evaluators:
            identity = dict(run_id=run_id, case_id=response["case_id"],
                            response_id=response["response_id"], evaluator=evaluator.name)
            try:
                if evaluator.name in stopped_evaluators:
                    result = EvaluationResult(status="skipped", reason="Not attempted after Judge authentication, permission, quota, or model failure.")
                elif response["status"] == "error":
                    result = EvaluationResult(status="skipped", reason="Model generation failed.")
                else:
                    result = evaluator.evaluate(by_id[response["case_id"]], response, context=context)
                    if result.details.get("stop_evaluator"):
                        stopped_evaluators.add(evaluator.name)
                annotation = dict(identity, **asdict(result))
                validate_rows([annotation], project_path(config["annotation_schema"]))
            except Exception as exc:
                annotation = dict(identity, **asdict(EvaluationResult(status="error", reason="Evaluator failed: " + type(exc).__name__)))
            annotations.append(annotation)
    target = artifact(config, "annotations", run_id, "annotations.jsonl")
    write_jsonl(target, annotations)
    return target
