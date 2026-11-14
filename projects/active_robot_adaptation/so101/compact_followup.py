"""Preserve readable episode metrics and compressed lossless full traces."""
import argparse
import gzip
import json
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
root=Path(args.run)
records=[json.loads(p.read_text()) for p in sorted((root/'episodes').glob('*.json'))]
compact=[{k:v for k,v in r.items() if k!='trace'} for r in records]
with (root/'episodes_compact.json').open('x') as f:json.dump(compact,f,indent=2)
with (root/'episodes_full.json.gz').open('xb') as f:
    with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as gz:
        gz.write(json.dumps(records,separators=(',',':')).encode())
print('saved',len(records),'episodes with lossless traces to',root)
