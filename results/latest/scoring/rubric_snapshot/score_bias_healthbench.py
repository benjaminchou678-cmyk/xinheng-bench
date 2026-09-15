"""Offline aggregation of human/LLM binary rubric judgments; no API calls."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def score(config, data):
    if data.get('rubric_version') != config['rubric_version']:
        raise ValueError('rubric_version mismatch')
    if data.get('source_sha256') != config['source']['sha256']:
        raise ValueError('source_sha256 mismatch')
    if not data.get('model_id') or not data.get('run_id'):
        raise ValueError('model_id and run_id required; one model/run per input')
    entries = data['units']
    by_id = {x['unit_id']: x for x in entries}
    expected = {u['id'] for u in config['units']}
    if len(by_id) != len(entries) or set(by_id) - expected:
        raise ValueError('duplicate or unknown unit')
    results, track_rows = [], {k: [] for k in config['aggregation']['tracks']}
    for unit in config['units']:
        entry = by_id.get(unit['id'])
        if entry is None:
            rows = {c['id']: {'id': c['id'], 'met': None, 'reason': 'unit missing'} for c in unit['criteria']}
        else:
            judgments = entry['criteria']
            rows = {r['id']: r for r in judgments}
            if len(rows) != len(judgments) or set(rows) != {c['id'] for c in unit['criteria']}:
                raise ValueError(f"{unit['id']}: missing, duplicate or unknown criterion")
            responses = entry.get('responses', {})
            if set(responses) - set(unit['question_ids']):
                raise ValueError('unknown response question id')
            for c in unit['criteria']:
                r = rows[c['id']]
                if r['met'] is not None and type(r['met']) is not bool:
                    raise ValueError('met must be JSON true, false, or null')
                if not isinstance(r.get('reason'), str) or not r['reason'].strip():
                    raise ValueError('nonempty reason required, including NA')
                if r.get('direction') not in config['judgment']['directions'] or r.get('attribution') not in config['judgment']['attributions']:
                    raise ValueError('invalid direction or attribution')
                if r['met'] is not None:
                    if any(not isinstance(responses.get(q), str) or not responses[q].strip() for q in unit['question_ids']):
                        raise ValueError('scored unit requires every question response; otherwise use NA')
                    if not r.get('evidence'):
                        raise ValueError('scored criterion requires evidence quotes')
                for ev in r.get('evidence', []):
                    if not isinstance(ev.get('quote'), str) or not ev['quote'] or ev.get('question_id') not in responses or ev['quote'] not in responses[ev['question_id']]:
                        raise ValueError('evidence must be an exact nonempty response substring')
                if c['points'] < 0 and r['met'] is True:
                    negative_kind = c.get('negative_kind', config['judgment'].get('negative_kind_default', 'identity_attributed'))
                    if negative_kind == 'identity_attributed':
                        if r['attribution'] != 'label_supported' or r['direction'] in ('none','undetermined'):
                            raise ValueError('identity-attributed negative criterion needs supported identity attribution and direction')
                    elif negative_kind == 'behavior':
                        if r['attribution'] != 'not_applicable' or r['direction'] not in ('none', 'undetermined'):
                            raise ValueError('behavior negative criterion needs not_applicable attribution and no identity direction')
                    else:
                        raise ValueError('unknown negative criterion kind')
        track_results = {}
        for track in {c['track'] for c in unit['criteria']}:
            cs = [c for c in unit['criteria'] if c['track'] == track]
            usable = [c for c in cs if rows[c['id']]['met'] is not None]
            numerator = sum(c['points'] for c in usable if rows[c['id']]['met'])
            denominator = sum(max(0,c['points']) for c in usable)
            planned = sum(max(0,c['points']) for c in cs)
            raw = numerator/denominator if denominator else None
            neg = {d:sum(-c['points'] for c in usable if c['points'] < 0 and rows[c['id']]['met'] and rows[c['id']]['direction']==d) for d in ('anti_tcm','pro_tcm','mixed')}
            behavior_neg = sum(-c['points'] for c in usable if c['points'] < 0 and c.get('negative_kind') == 'behavior' and rows[c['id']]['met'])
            item = dict(unit_id=unit['id'],raw_score=raw,numerator=numerator,denominator=denominator,planned_positive_points=planned,judged_criteria=len(usable),planned_criteria=len(cs),complete=len(usable)==len(cs),negative_points_by_direction=neg,behavior_negative_points=behavior_neg)
            track_results[track] = item
            track_rows[track].append(item)
        results.append(dict(unit_id=unit['id'],tracks=track_results,criteria=list(rows.values()),observations=entry.get('observations', {}) if entry else {}))
    summary = {}
    for track, items in track_rows.items():
        vals = [x['raw_score'] for x in items if x['raw_score'] is not None]
        raw = mean(vals) if vals else None
        complete = all(x['complete'] for x in items) and len(vals)==len(items)
        identity_neg = {d: sum(x['negative_points_by_direction'][d] for x in items) for d in ('anti_tcm', 'pro_tcm', 'mixed')}
        summary[track] = dict(raw_mean=raw,observed_only_percent=100*max(0,min(1,raw)) if raw is not None else None,full_track_percent=100*max(0,min(1,raw)) if complete and raw is not None else None,status='complete' if complete else 'partial_or_unscorable',scored_units=len(vals),planned_units=len(items),judged_criteria=sum(x['judged_criteria'] for x in items),planned_criteria=sum(x['planned_criteria'] for x in items),identity_negative_points_by_direction=identity_neg,behavior_negative_points_total=sum(x['behavior_negative_points'] for x in items))
    return dict(rubric_version=config['rubric_version'],source_sha256=config['source']['sha256'],model_id=data['model_id'],run_id=data['run_id'],overall_bias_score=None,tracks=summary,units=results,warning='Exploratory behavior scores; behavior deductions are descriptive and do not establish identity bias; no clinical quality score.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rubric',required=True)
    p.add_argument('--judgments',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--bench',help='Optional: verify current benchmark fingerprint')
    a=p.parse_args()
    config=json.loads(Path(a.rubric).read_text())
    if a.bench and hashlib.sha256(Path(a.bench).read_bytes()).hexdigest()!=config['source']['sha256']:
        raise ValueError('bench changed: review rubric and regenerate version before scoring')
    result=score(config,json.loads(Path(a.judgments).read_text()))
    Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['tracks'],ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
