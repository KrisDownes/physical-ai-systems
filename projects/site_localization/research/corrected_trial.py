"""Combined conventional corrections; the independent evaluator is unchanged."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import sys
import run
import transfer_layouts


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixture-layout',required=True)
    p.add_argument('--fixture-condition',required=True)
    p.add_argument('--split',default='correction_development')
    p.add_argument('--frame-policy',default='dock_anchor',choices=['dock_anchor','surface_registration','surface_fit_diagnostic','original_commission_map'])
    a,remaining=p.parse_known_args()
    if '--map-from-mission' in remaining:
        j=remaining.index('--map-from-mission')+1
        old=remaining[j:j+3];new=[format(Decimal(x),'f') for x in old]
        assert [float(x).hex() for x in old]==[float(x).hex() for x in new]
        remaining[j:j+3]=new
    run.make=lambda out,condition,remedy:transfer_layouts.make(out,a.fixture_layout,a.fixture_condition,remedy)
    out=Path(remaining[remaining.index('--out')+1]).resolve()
    sys.argv=['corrected/run.py',*remaining,'--layout','cell','--condition','unchanged',
              '--arrival-tolerance','.03','--startup-readiness','--slam-pose-source','tf']
    try:run.main()
    finally:
        if out.exists():(out/'correction_invocation.json').write_text(json.dumps(dict(
            split=a.split,layout=a.fixture_layout,condition=a.fixture_condition,
            frame_policy=a.frame_policy,
            corrections=['pose subscriber readiness before physics','3cm internal arrival tolerance',
                         'explicit declared map frame; surface registration consistency audit',
                         'official map-odom TF at exact wheel timestamps; no second pose interpolation'],
            physical_requirements_changed=False,truth_used_by_method=False),indent=2)+'\n')


if __name__=='__main__':main()
