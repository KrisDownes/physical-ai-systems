"""Frozen fixture adapter; corrected runtime and independent judge unchanged."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
import run
import reserved_layout


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixture-condition',required=True)
    args,rest=p.parse_known_args()
    j=rest.index('--map-from-mission')+1
    old=rest[j:j+3];rest[j:j+3]=[format(Decimal(x),'f') for x in old]
    assert [float(x).hex() for x in old]==[float(x).hex() for x in rest[j:j+3]]
    run.make=lambda out,condition,remedy:reserved_layout.make(out,args.fixture_condition,remedy)
    sys.argv=['reserved/run.py',*rest,'--layout','cell','--condition','unchanged','--arrival-tolerance','.03','--startup-readiness','--slam-pose-source','tf']
    out=Path(rest[rest.index('--out')+1])
    try:run.main()
    finally:
        if out.exists():(out/'reserved_invocation.json').write_text(json.dumps(dict(condition=args.fixture_condition,scope='reserved prospective comparison; labels in numerical scorer remain development',requirements_changed=False)))


if __name__=='__main__':main()
