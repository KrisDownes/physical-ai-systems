"""Fixture adapter around the immutable conventional runner; no method edits."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def verify(freeze):
    baseline=json.loads((freeze/'manifest.json').read_text())
    for name,digest in baseline['files'].items():
        path=Path(name[len('external:'):]) if name.startswith('external:') else freeze/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Frozen baseline changed: {name}')
    study=json.loads((freeze/'study_manifest.json').read_text())
    for name,digest in study['files'].items():
        if hashlib.sha256((freeze/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Frozen study changed: {name}')
    return baseline['revision']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--cohort-layout',required=True)
    p.add_argument('--cohort-condition',required=True)
    args,remaining=p.parse_known_args();freeze=args.freeze.resolve()
    revision=verify(freeze)
    sys.path.insert(0,str(freeze/'research'))
    generator=module('heldout_fixtures',freeze/'study/transfer_layouts.py')
    runner=module('frozen_runner',freeze/'research/run.py')
    runner.make=lambda out,condition,remedy:generator.make(out,args.cohort_layout,args.cohort_condition,remedy)
    # The old parser calls this extension through its generic cell entry point.
    # Its placeholder condition does not choose the generated physical world.
    sys.argv=['frozen/run.py',*remaining,'--layout','cell','--condition','unchanged']
    out=Path(remaining[remaining.index('--out')+1]).resolve()
    try:runner.main()
    finally:
        if out.exists():
            (out/'cohort_invocation.json').write_text(json.dumps(dict(
                baseline_revision=revision,layout=args.cohort_layout,condition=args.cohort_condition,
                split='heldout_transfer_A',method_runner_unchanged=True,
                parser_adapter='Frozen generic-cell generation replaced by prospectively hashed fixture factory only',
                raw_score_metadata='Frozen scorer hardcodes development labels; cohort identity is in this envelope. Numerical judging is unchanged.'),indent=2)+'\n')


if __name__=='__main__':main()
