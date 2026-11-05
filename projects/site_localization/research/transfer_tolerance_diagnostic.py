"""One post-cohort conventional controller error-budget diagnostic.

The immutable baseline remains untouched. Only the controller's translational
arrival threshold changes from .09 to .03 metres; this is not a research method.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    freeze, cohort, root = args.freeze.resolve(), args.cohort.resolve(), args.root.resolve()
    sys.path.insert(0, str(freeze / 'research'))
    adapter = load('frozen_adapter_for_tolerance', freeze / 'study/transfer_trial.py')
    revision = adapter.verify(freeze)
    original = freeze / 'research/controller.py'
    body = original.read_text()
    assert body.count('if distance>.09:') == 1
    changed = body.replace('if distance>.09:', 'if distance>.03:')
    root.mkdir(parents=True, exist_ok=True)
    support = root / 'tolerance_adapter'
    support.mkdir(exist_ok=False)
    derivative = support / 'controller_derivative.py'
    derivative.write_text(changed)
    wrapper = support / 'controller.py'
    wrapper.write_text('import sys\nfrom pathlib import Path\n'
        + f'sys.path.insert(0, {str(freeze / "research")!r})\n'
        + f'body = Path({str(derivative)!r}).read_text()\n'
        + f'exec(compile(body, {str(derivative)!r}, "exec"), '
        + f'{{"__name__": "__main__", "__file__": {str(original)!r}}})\n')
    source = cohort / 'maintenance_bays__unchanged__full_acquisition'
    deployment = cohort / 'maintenance_bays__unchanged__full_deployment'
    previous = json.loads((deployment / 'controller_config.json').read_text())
    delta = dict(kind='post-cohort engineering diagnosis; not held-out validation',
        baseline_revision=revision, original_controller=str(original),
        original_controller_sha256=sha(original), derivative_sha256=sha(derivative),
        wrapper_sha256=sha(wrapper), adapter_sha256=sha(Path(__file__)),
        replacement_count=1, before='if distance>.09:', after='if distance>.03:',
        interpretation='Tighten control arrival tolerance to reserve room for localization error; conventional remedy, no novelty.',
        transport='Wrapper preserves frozen __file__ and frozen frames import; same native AMCL binary.',
        map_sha256=sha(source / 'mapper/map.txt'),
        map_from_mission=previous['map_from_mission'], seed=3000, domain=224,
        frozen_evaluator_unchanged=True, physical_target_tolerances_unchanged=True)
    (root / 'tolerance_delta_prelaunch.json').write_text(json.dumps(delta, indent=2) + '\n')
    generator = load('frozen_generator_for_tolerance', freeze / 'study/transfer_layouts.py')
    runner = load('frozen_runner_for_tolerance', freeze / 'research/run.py')
    runner.make = lambda out, condition, remedy: generator.make(out, 'maintenance_bays', 'unchanged', remedy)
    actual_popen = subprocess.Popen
    interceptions = []

    def intercept(cmd, *positional, **keyword):
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[1] == str(original):
            assert not interceptions
            interceptions.append(True)
            cmd[1] = str(wrapper)
        return actual_popen(cmd, *positional, **keyword)

    subprocess.Popen = intercept
    from decimal import Decimal
    pose = [format(Decimal(str(x)), 'f') for x in previous['map_from_mission']]
    sys.argv = ['frozen/run.py', '--out', str(root / 'tolerance'),
        '--layout', 'cell', '--condition', 'unchanged', '--remedy', 'acquired_map',
        '--map', str(source / 'mapper/map.txt'), '--map-from-mission', *pose,
        '--localizer', 'amcl', '--mapping', 'off', '--seed', '3000',
        '--domain', '224', '--wall-limit', '620']
    try:
        runner.main()
    finally:
        subprocess.Popen = actual_popen
        assert len(interceptions) == 1
        adapter.verify(freeze)
        (root / 'tolerance' / 'diagnostic_delta.json').write_text(json.dumps(delta, indent=2) + '\n')


if __name__ == '__main__':
    main()
