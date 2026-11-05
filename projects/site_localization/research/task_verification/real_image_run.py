"""Single fixed-reader pass. No evaluation labels are read by this program."""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np


def main(root, images):
    assert not (root/'predictions.json').exists(), 'Preserve earlier executions.'
    freeze = json.loads((root/'pre_execution_freeze.json').read_text())
    for name in ['frozen_reader.py', 'method_inputs.json', 'selection.json']:
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == freeze['files'][name]
    spec = importlib.util.spec_from_file_location('fixed_reader', root/'frozen_reader.py')
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    cv2.setNumThreads(1)
    configs = {r['index']: r for r in json.loads((root/'method_inputs.json').read_text())}
    output = []
    with (root/'attempts.jsonl').open('x') as ledger:
        for row in json.loads((root/'selection.json').read_text()):
            i, name = row['index'], row['file']
            if row['strict_decode'] != 'ok':
                item = {'index': i, 'file': name, 'execution': 'source_unavailable',
                        'reason': row['strict_decode'], 'reader_called': False}
                ledger.write(json.dumps(item)+'\n'); ledger.flush(); output.append(item)
                continue
            ledger.write(json.dumps({'index': i, 'file': name, 'event': 'start'})+'\n'); ledger.flush()
            image = cv2.imread(str(images/name), cv2.IMREAD_GRAYSCALE)
            start, cpu = time.perf_counter(), time.process_time()
            try:
                assert image is not None, 'OpenCV decode failed'
                pred = reader.read_needle(image, np.eye(3), configs[i])
                item = {'index': i, 'file': name, 'execution': 'complete', 'reader_called': True,
                        'prediction': pred, 'wall_s': time.perf_counter()-start,
                        'cpu_s': time.process_time()-cpu}
            except Exception as error:
                item = {'index': i, 'file': name, 'execution': 'runtime_failure',
                        'reader_called': True, 'error': repr(error)}
            ledger.write(json.dumps(item)+'\n'); ledger.flush(); output.append(item)
    (root/'predictions.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--images', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.images)
