"""Offline hierarchical audit; never provides evaluator evidence to a selector."""
import json
from collections import Counter
from pathlib import Path


def summarize(folder):
    folder=Path(folder)
    events=[json.loads(l) for l in (folder/'executive.jsonl').read_text().splitlines()] if (folder/'executive.jsonl').exists() else []
    manifest=json.loads((folder/'manifest.json').read_text())
    totals=Counter()
    for e in events:
        if e['kind']=='state_duration':totals[e['cause']]+=e['duration_wall_s']
    counts=Counter(e['kind'] for e in events)
    result=dict(benchmark='hierarchical_destination_selection',selector=manifest.get('selector'),
        diagnostic=manifest.get('diagnostic'),event_counts=dict(counts),idle_wall_s_by_cause=dict(totals),
        selection_requests=counts['decision_requested'] if manifest.get('selector')!='classical' else None,goal_selections=counts['goal_selected'] if manifest.get('selector')!='classical' else None,
        goal_assignments=counts['goal_assigned'],
        arrivals=counts['goal_reached'],failures=counts['failure'],changed_path_replans=counts['path_replanned'],
        recovery_events=counts['recovery_requested'],
        actual_model_inference_requests=0 if manifest.get('selector')!='llm' else None,
        model_request_count_note='Mock/classical uses no model. LLM turn count is NOT internal inference count; cap is 20 externally submitted turns, no token/monetary guarantee.',
        request_to_commit_latency_proxy_s=[e['request_to_response_s'] for e in events if e['kind']=='goal_selected'],
        latency_note='Executive request -> validated goal commit includes snapshot preparation, queue and next-map revalidation; not pure inference latency.',
        token_usage=None,termination=manifest['termination'],cleanup_complete=manifest.get('cleanup_complete'))
    # Count transport turns separately from actual model inference attempts.
    rpc=folder/'driver_rpc.jsonl'
    result['submitted_decision_turns']=0
    result['observable_turn_roundtrip_wall_s']=[]
    result['raw_token_usage_notifications']=[]
    result['exposed_retry_compaction_events']=[]
    result['exposed_response_ids']=[]
    if rpc.exists():
        requests={};turns={}
        for line in rpc.read_text().splitlines():
            row=json.loads(line);msg=row['message']
            if row['direction']=='sent' and msg.get('method')=='turn/start':
                requests[msg['id']]=row['wall_s'];result['submitted_decision_turns']+=1
            if row['direction']=='received' and msg.get('id') in requests:
                turn=msg.get('result',{}).get('turn',{})
                if turn.get('id'):turns[turn['id']]=requests[msg['id']]
            if msg.get('method')=='turn/completed':
                tid=msg.get('params',{}).get('turn',{}).get('id')
                if tid in turns:result['observable_turn_roundtrip_wall_s'].append(row['wall_s']-turns[tid])
            if row['direction']=='received' and any(term in msg.get('method','').lower() for term in ('retry','compact')):
                result['exposed_retry_compaction_events'].append(row)
            def response_ids(value):
                if isinstance(value,dict):
                    for k,v in value.items():
                        if k in ('response_id','responseId') and isinstance(v,str):yield v
                        else:yield from response_ids(v)
                elif isinstance(value,list):
                    for item in value:yield from response_ids(item)
            result['exposed_response_ids'].extend(response_ids(msg))
            if 'tokenUsage' in msg.get('method',''):
                result['raw_token_usage_notifications'].append(row)
    result['exposed_response_ids']=sorted(set(result['exposed_response_ids']))
    result['internal_request_note']='Internal requests may exceed externally submitted turns. Exposed events/IDs may be incomplete; absence does not prove no retry/compaction. No token or monetary cap.'
    result['turn_latency_note']='Monotonic turn/start submission to matching turn/completed notification, when exposed; includes transport and any internal inference, not pure model inference.'
    (folder/'hierarchical_results.json').write_text(json.dumps(result,indent=2)+'\n')
    original=json.loads((folder/'results.json').read_text())
    original['hierarchical']=result
    (folder/'results.json').write_text(json.dumps(original,indent=2)+'\n')
    return result


def report(root):
    root=Path(root);folder=root/'classical'
    r=summarize(folder);base=json.loads((folder/'results.json').read_text());m=base['manifest']
    data={'hierarchical_'+str(r['selector']):base};(root/'results.json').write_text(json.dumps(data,indent=2)+'\n')
    (root/'report.md').write_text(f'''# Hierarchical {r['selector']} diagnostic

Diagnostic only; no model-performance evidence. One episode, no retries.
Termination: {m['termination']}. Wall/simulation: {m.get('episode_wall_s')} / {m.get('episode_sim_s')} s.
Path: {base.get('path_length_m')} m. Known grid (not accessible area): {base.get('final_coverage')}.
Selections/arrivals: {r['goal_selections']} / {r['arrivals']}; changed paths: {r['changed_path_replans']}.
Actual model calls: {r['actual_model_inference_requests']}; token usage N/A.
Idle durations by cause: {json.dumps(r['idle_wall_s_by_cause'])}.

Raw bag, executive and exact goal observation logs are under classical/ (legacy
runner's shared local-controller slot; this folder name does not label the selector).
See classical/hierarchical_results.json, manifest.json, bag_verification.json and
cleanup.json. Source/configuration snapshots are in source/ and source_hashes.json.
No contact sensor: obstacle interventions are not collisions.
''')

if __name__=='__main__':
    import sys
    report(sys.argv[1])
