"""Descriptive observations only; no clinical or bias scores."""
import json
from evaluation.base import BaseEvaluator, EvaluationResult

FIELDS = ('evidence_confidence', 'recommendation_strength', 'safety_concern')


def parse_ratings(text):
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != set(FIELDS) | {'explanation'}:
        raise ValueError('Unexpected rating fields.')
    if any(type(data[k]) is not int or not 1 <= data[k] <= 5 for k in FIELDS):
        raise ValueError('Ratings must be integers from 1 to 5.')
    if not isinstance(data['explanation'], str) or not data['explanation'].strip():
        raise ValueError('Explanation is required.')
    return data


class PilotObservationEvaluator(BaseEvaluator):
    name = 'pilot_observation'

    def evaluate(self, case, response, *, context):
        if response['status'] != 'ok':
            return EvaluationResult(status='skipped', reason='No complete real model response.')
        if not case['mirror_pair_id']:
            return EvaluationResult(status='not_implemented', reason='Manual review required.',
                                    details={'manual_review': case['metadata']['manual_review']})
        try:
            values = parse_ratings(response['text'])
        except (ValueError, TypeError):
            return EvaluationResult(status='error', reason='Rating JSON is invalid; no values inferred.')
        return EvaluationResult(status='ok', score=None,
                                reason='Extracted model judgments only; not a quality or bias score.',
                                details={'ratings': values})


def pair_gaps(cases, responses, annotations):
    observations = {a['response_id']: a['details'].get('ratings') for a in annotations
                    if a['evaluator'] == 'pilot_observation' and a['status'] == 'ok'}
    output = []
    for group in sorted({c['mirror_pair_id'] for c in cases if c['mirror_pair_id']}):
        members = [c for c in cases if c['mirror_pair_id'] == group]
        for model in sorted({r['model_id'] for r in responses}):
            values = {}
            for case in members:
                found = [r for r in responses if r['case_id'] == case['case_id'] and r['model_id'] == model]
                if len(found) == 1 and found[0]['status'] == 'ok':
                    label = case.get('metadata', {}).get('system_label')
                    if label:
                        values[label] = observations.get(found[0]['response_id'])
            valid = len(members) == 2 and set(values) == {'CHM', 'modern'} and all(values.values())
            output.append({'pair_id': group, 'model_id': model,
                           'status': 'observed' if valid else 'unavailable',
                           'chm_minus_modern': {k: values['CHM'][k] - values['modern'][k] for k in FIELDS} if valid else None})
    return output
