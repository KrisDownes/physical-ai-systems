"""Use the unchanged numerical reporter with the correct changed-site context."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from register_maps import occupied_points, transform
from review_pair_report import report


def main(root):
    report(root)
    path=root/'paired_result.json'
    data=json.loads(path.read_text())
    data['scope']='One matched development pair on one reused rearrangement map. Additions selections agree exactly and have no new missions. Visits are correlated.'
    data['context_adapter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data['selection_record']='selections_before_outcomes.json'
    path.write_text(json.dumps(data,indent=2)+'\n')
    support=[]
    choices=json.loads((root/'selections_before_outcomes.json').read_text())
    for case in choices['cases']:
        folder=root/case['condition']
        source,_=occupied_points(folder/'source_surface.txt')
        target,_=occupied_points(folder/'target_surface.txt')
        tree=cKDTree(target)
        distances={k:tree.query(transform(source,np.array(s['pose'])))[0]
                   for k,s in case['selections'].items()}
        ids=np.argsort(distances['trimmed'],kind='stable')
        cut=int(np.ceil(.7*len(ids)))
        support.append(dict(condition=case['condition'],source_cells=len(source),
            on_same_trimmed_winner_subsets={k:dict(
                retained_rms_m=float(np.sqrt(np.mean(d[ids[:cut]]**2))),
                discarded_rms_m=float(np.sqrt(np.mean(d[ids[cut:]]**2))),
                beyond_cap_fraction=float(np.mean(d>.3))) for k,d in distances.items()},
            note='Source-to-reference geometry only; high residuals are not labels of actual changed surfaces.'))
    (root/'support_diagnostics.json').write_text(json.dumps(support,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root.resolve())
