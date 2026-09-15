from collections import Counter
from evaluation.metrics import summarize
from evaluation.pilot import pair_gaps
import json
from pipeline.common import artifact, read_json, read_jsonl, write_json, validate_rows, project_path


def quoted(text):
    return '\n'.join('> ' + line for line in text.splitlines())


def run(config, run_id):
    source = artifact(config, 'raw_responses', run_id, 'responses.jsonl')
    responses = read_jsonl(source)
    manifest = read_json(source.parent / 'manifest.json')
    phase = manifest['benchmark_config']['phase']
    annotations = read_jsonl(artifact(config, 'annotations', run_id, 'annotations.jsonl'))
    validate_rows(responses, project_path(config['response_schema']))
    validate_rows(annotations, project_path(config['annotation_schema']))
    if any(row['run_id'] != run_id for row in responses + annotations):
        raise ValueError('Artifacts contain another run id.')
    summary = dict(run_id=run_id, phase=phase, response_count=len(responses),
                   response_status_counts=dict(Counter(r['status'] for r in responses)),
                   metrics=summarize(annotations))
    summary['token_usage'] = {
        field: sum(r['metadata'].get('usage', {}).get(field, 0) for r in responses)
        for field in ['input_tokens', 'output_tokens', 'total_tokens']
    }
    if phase == "pilot":
        cases = read_jsonl(source.parent / "cases.jsonl")
        summary["pair_observations"] = pair_gaps(cases, responses, annotations)
    write_json(artifact(config, 'scores', run_id, 'summary.json'), summary)
    report = artifact(config, 'reports', run_id, 'report.md')
    report.parent.mkdir(parents=True, exist_ok=True)
    mode = 'Phase 0 占位流程' if phase == 0 else 'Pilot 试跑'
    rubric_version = manifest['evaluation_config'].get('rubric_version')
    scored_a4 = summary['metrics'].get('a4_ai_judge')
    status_note = ('已应用探索阶段 AI Judge rubric；自动评分不等于医生审定，不构成系统性 bias 结论。'
                   if scored_a4 else '尚未应用最终评分标准；逐题回答供检查，不构成医疗质量或偏差结论。')
    sections = [f'# 中药/中成药治疗呈现偏差 — {mode}报告', f'Run: `{run_id}`', status_note,
                f'- 响应数：{len(responses)}\n- 注释数：{len(annotations)}\n'
                f'- 响应状态：{summary["response_status_counts"]}\n'
                f'- 评估状态：{summary["metrics"]["status_counts"]}\n'
                f'- Token 用量：{summary["token_usage"]}\n- Rubric：{rubric_version or "未指定"}']
    if scored_a4:
        sections.extend(['## AI Judge 汇总',
                         f'- Capability 平均分：{scored_a4["capability_mean"]} / 100（{scored_a4["capability_count"]}题）\n'
                         f'- Symmetry 平均分：{scored_a4["symmetry_mean"] if scored_a4["symmetry_mean"] is not None else "N/A"} / 100（{scored_a4["symmetry_count"]}组）\n'
                         f'- 错误标记计数：{scored_a4["flag_counts"]}',
                         '这些分数只用于探索阶段筛查，需结合原文证据和医学专家复核。'])
    elif phase == "pilot":
        sections.extend(["## 配对观察（中药/中成药减现代医学干预）",
            "以下是模型给出的情境判断差，不是临床评分或偏差定论；unavailable 表示无可用真实配对。",
            "```json\n" + json.dumps(summary["pair_observations"], ensure_ascii=False, indent=2) + "\n```",
            "单次小样本试跑，未作统计推断；偏好题和安全对照须人工复核。" ])
    for response in responses:
        sections.extend([f'## {response["case_id"]} · {response["model_id"]}',
                         '状态：' + response['status'],
                         '问题：\n\n' + quoted(response['prompt']),
                         '回答：\n\n' + quoted(response['text'] or '（无回答）')])
        reviews = [a['details'].get('manual_review', []) for a in annotations
                   if a['response_id'] == response['response_id'] and a['evaluator'] == 'pilot_observation']
        for checklist in reviews:
            if checklist:
                sections.append('待人工复核（未打分）：\n\n' + '\n'.join('- ' + item for item in checklist))
        judgments = [a for a in annotations if a['response_id'] == response['response_id']
                     and a['evaluator'] == 'a4_ai_judge']
        if judgments:
            judgment = judgments[0]
            if judgment['status'] == 'ok':
                details = judgment['details']
                lines = [f'Capability Score：{details["capability_score"]} / 100',
                         f'S1：{details["scenario_fit"]["score"]} / 2 — {details["scenario_fit"]["reason"]}']
                lines.extend(f'{key}：{value["score"] if value["score"] is not None else "N/A"} / 2 — {value["reason"]}'
                             for key, value in details['capability'].items())
                if details.get('symmetry'):
                    lines.append(f'Symmetry Score：{details["symmetry"]["score"]} / 100')
                    lines.extend(f'{key}：{value["score"] if value["score"] is not None else "N/A"} / 2 — {value["reason"]}'
                                 for key, value in details['symmetry']['dimensions'].items())
                    lines.append('不对称观察：' + details['symmetry']['asymmetry'])
                    lines.append('合理替代解释：' + details['symmetry']['alternative_explanation'] + ' — ' + details['symmetry']['reason'])
                lines.append('错误标记：' + json.dumps(details['flags'], ensure_ascii=False))
                lines.append('人工复核优先级：' + details['human_review_priority'])
                lines.append('当前最高结论：' + details['conclusion'])
                sections.append('AI Judge：\n\n' + '\n\n'.join(lines))
            else:
                sections.append('AI Judge 状态：' + judgment['status'] + ' — ' + judgment['reason'])
        if response['error']:
            sections.append('错误：' + response['error'])
    report.write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    return report
