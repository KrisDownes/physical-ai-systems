"""Post-opening conventional surface-representation diagnostic; no ROS or truth.

Keep occupied cells with a 4-connected FREE neighbour. Occupied interior cells
become UNKNOWN, never free. Unknown neighbours and raster edges do not count
as free evidence. Both maps receive the same transformation. This is a known
representation correction, not a new registration or acceptance method.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CELLS = ('l_workshop__unchanged', 'l_workshop__additions', 'maintenance_bays__unchanged')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boundary_map(source, output):
    with Path(source).open() as stream:
        header_text = stream.readline()
        header = np.fromstring(header_text, sep=' ')
        if len(header) != 5:
            raise ValueError('Native map requires five header fields')
        width, height = map(int, header[:2])
        values = np.loadtxt(stream, dtype=int).reshape(height, width)
    if not np.isin(values, (-1, 0, 1)).all():
        raise ValueError('Unexpected native state code')
    # Confirmed against slam_bridge.export_map: ROS free=0 -> native -1;
    # ROS occupied=100 -> native 1; everything else -> native unknown=0.
    free = values == -1
    adjacent_free = np.zeros(values.shape, dtype=bool)
    adjacent_free[1:] |= free[:-1]
    adjacent_free[:-1] |= free[1:]
    adjacent_free[:, 1:] |= free[:, :-1]
    adjacent_free[:, :-1] |= free[:, 1:]
    occupied = values == 1
    retained = occupied & adjacent_free
    result = values.copy()
    result[occupied & ~retained] = 0
    with Path(output).open('w') as stream:
        stream.write(header_text)
        np.savetxt(stream, result, fmt='%d')
    return dict(source=str(source), output=str(output), source_sha256=digest(source),
                output_sha256=digest(output), native_states=dict(free=-1, unknown=0, occupied=1),
                input_counts={str(v): int((values == v).sum()) for v in (-1, 0, 1)},
                output_counts={str(v): int((result == v).sum()) for v in (-1, 0, 1)},
                retained_occupied=int(retained.sum()), removed_occupied=int((occupied & ~retained).sum()),
                rule='occupied AND at least one 4-connected free neighbour; removed occupied -> unknown',
                unknown_or_outside_counts_as_free=False, grid_frame_changed=False)


def run(cohort, out):
    freeze = json.loads((ROOT/'docs/transfer/frozen_revision.json').read_text())
    expected_hash = freeze['files']['research/register_maps.py']
    config = freeze['config']['alignment']
    if config['stable_region'] is not None:
        raise ValueError('Diagnostic expects common dock prior only')
    results = []
    for prefix in CELLS:
        acquisition = cohort/(prefix+'__full_acquisition')
        old = json.loads((acquisition/'alignment.json').read_text())
        code = acquisition/'alignment.source.py'
        if digest(code) != expected_hash:
            raise ValueError(f'Frozen implementation hash mismatch: {code}')
        for key, value in config.items():
            if old['parameters'][key] != value:
                raise ValueError(f'Frozen configuration mismatch: {prefix} {key}')
        dest = out/prefix
        dest.mkdir(parents=True, exist_ok=True)
        source_map, target_map = dest/'source_boundary.txt', dest/'target_boundary.txt'
        metadata = dict(diagnostic='conventional occupied/free interface extraction',
                        status='post-opening development diagnosis, not held-out validation',
                        prior=old['parameters']['prior'], prior_source='existing operational dock correspondence',
                        source=boundary_map(Path(old['parameters']['source']), source_map),
                        target=boundary_map(Path(old['parameters']['target']), target_map),
                        frozen_revision=freeze['revision'], registration_sha256=expected_hash,
                        no_ground_truth_inputs=True)
        command = [sys.executable, str(code), '--source', str(source_map), '--target', str(target_map),
                   '--output', str(dest/'alignment.json'), '--prior', *map(str, metadata['prior'])]
        for key, value in config.items():
            if value is not None:
                command.extend(['--'+key.replace('_', '-'), str(value)])
        metadata['command'] = command
        (dest/'extraction.json').write_text(json.dumps(metadata, indent=2)+'\n')
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        (dest/'registration.stdout').write_text(completed.stdout)
        (dest/'registration.stderr').write_text(completed.stderr)
        if completed.returncode:
            results.append(dict(cell=prefix, registration_failed=True,
                                source_retained=metadata['source']['retained_occupied'],
                                target_retained=metadata['target']['retained_occupied'],
                                reason=completed.stderr.strip()))
            continue
        alignment = json.loads((dest/'alignment.json').read_text())
        results.append(dict(cell=prefix, target_from_source=alignment['target_from_source'],
                            source_retained=metadata['source']['retained_occupied'],
                            target_retained=metadata['target']['retained_occupied'],
                            elapsed_wall_s=alignment['elapsed_wall_s']))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, default=ROOT/'results/transfer_20261022_A')
    parser.add_argument('--out', type=Path, default=ROOT/'results/transfer_diagnosis_20261022/surface')
    args = parser.parse_args()
    print(json.dumps(run(args.cohort.resolve(), args.out.resolve()), indent=2))
