"""Run attributed stock AMCL-core tracking baselines on development data only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2])
    parser.add_argument('--models', nargs='+', choices=['likelihood_field','likelihood_field_prob_noskip'],
                        default=['likelihood_field'])
    args = parser.parse_args()
    result_root = ROOT/'results/baseline'
    result_root.mkdir(parents=True,exist_ok=True)
    for sequence in ['dynamics_0','lt_changes_0']:
        data = ROOT/'data'/sequence
        manifest = json.loads((data/'manifest.json').read_text())
        (result_root/f'{sequence}_input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for model in args.models:
            for seed in args.seeds:
                name = f'{sequence}_{model}_seed{seed}'
                output = result_root/f'{name}.csv'
                timing = result_root/f'{name}.cost.json'
                command = [str(ROOT/'vendor/offline_amcl'),str(data/'map.txt'),
                           str(data/'frames.txt'),str(output),*map(str,manifest['initial_pose']),
                           str(seed),model,'1000','10000','0.2','0.2','100']
                before = time.perf_counter()
                # shell script is fixed, user/data strings passed only as argv.
                shell = 'source "$1"; shift; exec "$@"'
                proc = subprocess.run(['bash','-c',shell,'bash',str(ROOT/'vendor/setup_amcl.bash'),
                                       '/usr/bin/time','-f','{"max_rss_kib":%M,"cpu_user_s":%U,"cpu_system_s":%S}',
                                       '-o',str(timing),*command],capture_output=True,text=True)
                record = dict(command=command,returncode=proc.returncode,
                              wall_s=time.perf_counter()-before,stdout=proc.stdout,stderr=proc.stderr,
                              code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in [ROOT/'prepare.py',ROOT/'evaluate.py',
                                                     ROOT/'vendor/offline_amcl.cpp',ROOT/'vendor/offline_amcl']})
                (result_root/f'{name}.run.json').write_text(json.dumps(record,indent=2)+'\n')
                if proc.returncode: raise RuntimeError(proc.stderr)
                subprocess.run([sys.executable,str(ROOT/'evaluate.py'),sequence,str(output)],check=True,
                               stdout=subprocess.DEVNULL)
                print(name, f'{record["wall_s"]:.2f}s',flush=True)


if __name__=='__main__': main()
