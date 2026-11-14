"""Save compact metrics, explicit calibration decisions and lossless traces."""
import argparse
import gzip
import json
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
root=Path(args.run)
rows=[json.loads(p.read_text()) for p in sorted((root/'episodes').glob('*.json'))]
compact=[{k:v for k,v in r.items() if k!='trace'} for r in rows]
decisions=[]
for r in rows:
    for event in r['result'].get('calibration_events',[]):
        if 'source' not in event:continue
        event=dict(event)
        if event['source']=='table' and event.get('accepted') and 'reason' not in event:
            # Historical run: this is the sole branch that accepts the table.
            event['reason']='consistent_references_within_bound'
            event['reason_provenance']='derived from recorded selector source'
        decisions.append({'condition':r['condition']['name'],'case':r['case']['id'],
                          'method':r['method'],'decision':event})
for name,data in [('episodes_compact.json',compact),('calibration_decisions.json',decisions)]:
    with (root/name).open('x') as f:json.dump(data,f,indent=2)
with (root/'episodes_full.json.gz').open('xb') as f:
    with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as g:
        g.write(json.dumps(rows,separators=(',',':')).encode())
print('preserved',len(rows),'episodes and',len(decisions),'calibration decisions')
