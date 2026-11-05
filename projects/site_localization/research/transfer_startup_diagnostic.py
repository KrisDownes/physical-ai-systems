"""Development-only launch-readiness intervention; frozen method unchanged.

Do not start physics until both official /pose publisher and mission subscriber
are discovered. This tests a lost first keyframe / startup deadlock mechanism.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


READINESS = r'''
import json, time
import rclpy
rclpy.init()
n=rclpy.create_node('site_startup_readiness_diagnostic')
started=time.monotonic();stable=None;report=None
try:
    while time.monotonic()-started<40:
        rclpy.spin_once(n,timeout_sec=.1)
        subscribers=n.get_subscriptions_info_by_topic('/pose')
        publishers=n.get_publishers_info_by_topic('/pose')
        ready=any(s.node_name=='site_mission' for s in subscribers) and any(p.node_name=='slam_toolbox' for p in publishers)
        if ready:
            if stable is None:stable=time.monotonic()
            if time.monotonic()-stable>=2:
                report={'ready':True,'elapsed_wall_s':time.monotonic()-started,'stable_discovery_s':2.,
                        'subscriber_nodes':[s.node_name for s in subscribers],
                        'publisher_nodes':[p.node_name for p in publishers]}
                break
        else:stable=None
    if report is None:raise RuntimeError('No stable official /pose publisher and mission subscriber before physics')
    print(json.dumps(report))
finally:
    n.destroy_node();rclpy.shutdown()
'''


def main():
    args=sys.argv[1:]
    out=Path(args[args.index('--out')+1]).resolve()
    here=Path(__file__).resolve()
    real_popen=subprocess.Popen
    intervention=[]
    requires_readiness=('--localizer' in args and args[args.index('--localizer')+1]=='slam_toolbox')

    def guarded_popen(command,*positional,**kwargs):
        if requires_readiness and isinstance(command,(list,tuple)) and list(command[:2])==['gz','sim']:
            before=time.monotonic()
            reply=subprocess.run(['/usr/bin/python3','-c',READINESS],env=kwargs.get('env'),
                capture_output=True,text=True,timeout=45,check=False)
            entry=dict(returncode=reply.returncode,wall_s=time.monotonic()-before,
                       stdout=reply.stdout,stderr=reply.stderr)
            intervention.append(entry)
            (out/'startup_readiness.json').write_text(json.dumps(entry,indent=2)+'\n')
            if reply.returncode:raise RuntimeError('Startup diagnostic readiness barrier failed before physics')
        return real_popen(command,*positional,**kwargs)

    spec=importlib.util.spec_from_file_location('numeric_transport_adapter',here.with_name('transfer_trial_fixed.py'))
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    subprocess.Popen=guarded_popen
    try:adapter.main()
    finally:
        subprocess.Popen=real_popen
        if out.exists():
            (out/'startup_diagnostic.json').write_text(json.dumps(dict(
                split='development_diagnosis_after_transfer_A',
                mechanism='Prevent physics until official /pose publisher and mission subscriber discovery is stable for two wall seconds',
                method_core_changed=False,truth_inputs=False,
                adapter_sha256=hashlib.sha256(here.read_bytes()).hexdigest(),
                numeric_transport_sha256=hashlib.sha256(here.with_name('transfer_trial_fixed.py').read_bytes()).hexdigest(),
                readiness_attempts=intervention,
                limitations='DDS discovery barrier is a startup intervention, not a localization or map-repair contribution; physical success is not guaranteed.'
            ),indent=2)+'\n')


if __name__=='__main__':main()
