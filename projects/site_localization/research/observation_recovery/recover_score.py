"""Replay a retained run's frozen scorer with NumPy scalar JSON serialization.

No criteria or computations change. No simulation/verifier runs. Original
scorer, stderr, run record and first three outcomes are preserved byte-for-byte.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import types

import numpy as np


def scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__+' is not JSON serializable')


def main(run,out):
    out.mkdir(parents=True,exist_ok=False)
    inputs=['judge.jsonl','cell.json','integration_done.json','evaluation_contract.json',
            'sensor_truth.jsonl','controller.jsonl']+[p.name for p in run.glob('verifier_*.json')]
    hashes={}
    for name in inputs:
        src=run/name;hashes[name]=hashlib.sha256(src.read_bytes()).hexdigest()
        shutil.copyfile(src,out/name)
    scorer=Path(__file__).with_name('score.py')
    spec=importlib.util.spec_from_file_location('frozen_integration_scorer',scorer)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Only the integration scorer's encoder gets a JSON-native scalar adapter.
    module.json=types.SimpleNamespace(loads=json.loads,
        dumps=lambda *args,**kwargs:json.dumps(*args,default=scalar,**kwargs))
    module.main(out)
    record=dict(original_run=str(run),scorer_sha256=hashlib.sha256(scorer.read_bytes()).hexdigest(),
        original_error='numpy.bool_ physical_arrival_pass cannot be encoded by json.dumps',
        correction='Encode NumPy scalar using .item(); no arithmetic, predicates or thresholds changed',
        input_hashes=hashes,new_physics_runs=0,new_verifier_calls=0,originals_preserved=True)
    (out/'scoring_recovery.json').write_text(json.dumps(record,indent=2)+'\n')
    for name,h in hashes.items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==h


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.run.resolve(),a.out.resolve())
