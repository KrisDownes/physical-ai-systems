"""Transport-only retry: decimal notation preserves identical float values."""
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import sys


def main():
    args=sys.argv[1:];index=args.index('--map-from-mission')+1
    original=args[index:index+3];fixed=[format(Decimal(token),'f') for token in original]
    assert len(original)==3 and [float(x).hex() for x in original]==[float(x).hex() for x in fixed]
    args[index:index+3]=fixed
    freeze=Path(args[args.index('--freeze')+1]);out=Path(args[args.index('--out')+1])
    spec=importlib.util.spec_from_file_location('original_frozen_adapter',freeze/'study/transfer_trial.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    sys.argv=[sys.argv[0],*args]
    try:base.main()
    finally:
        if out.exists():(out/'numeric_transport.json').write_text(json.dumps(dict(
            adapter='v1.1 transport-only',original_tokens=original,transport_tokens=fixed,
            binary64_identical=True,float_hex=[float(x).hex() for x in original],
            note='No estimator, transform, seed, map, route, predicate or acceptance-rule change.'),indent=2)+'\n')


if __name__=='__main__':main()
