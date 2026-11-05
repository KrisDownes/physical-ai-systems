"""Sequential inspection-only matched handoffs. Experimental simulator resets."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
OUT = ROOT / "results/inspection_aware_20261028"


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def verify(manifest):
    changed = [name for name, expected in manifest["files"].items() if sha(REPO / name) != expected]
    if changed:
        raise RuntimeError("Frozen input changed: " + repr(changed))


def no_server():
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = [x.decode(errors="replace") for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
        except OSError:
            continue
        if any(Path(x).name == "gz" and args[i + 1:i + 3] == ["sim", "-s"] for i, x in enumerate(args)):
            raise RuntimeError("Other simulator active: " + str(proc))


def run(case, arm, manifest, suffix=""):
    verify(manifest)
    no_server()
    path = OUT / ("mission_" + case + "_" + arm + suffix)
    shutil.copytree(OUT / "templates" / case, path)
    inspection = path / "inspection"
    public = inspection / "public"
    shutil.copyfile(public / ("policy_" + arm + ".json"), public / "policy.json")
    for name in ("policy_nearest.json", "policy_gain.json", "policy_aware.json"):
        (public / name).unlink()
    env = os.environ.copy()
    env.update(ROS_DOMAIN_ID="216", GZ_PARTITION="matched_" + uuid.uuid4().hex,
               ROS_LOG_DIR=str(path / "ros_logs"), GZ_LOG_PATH=str(path / "gz_logs"),
               LIBGL_ALWAYS_SOFTWARE="1", MESA_LOADER_DRIVER_OVERRIDE="llvmpipe", EGL_PLATFORM="surfaceless")
    bridges = [
        "/site/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
        "/site/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
        "/site/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
        "/judge/truth@nav_msgs/msg/Odometry[gz.msgs.Odometry",
        "/site/contacts@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts",
        "/judge/wheel_contacts@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts",
        "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
    ]
    commands = {
        "bridge": ["/opt/ros/jazzy/lib/ros_gz_bridge/parameter_bridge", *bridges],
        "judge": ["/usr/bin/python3", str(ROOT / "sim/judge_node.py"), str(path)],
        "camera": ["/usr/bin/python3", str(ROOT / "research/moved_asset/camera.py"), str(inspection)],
        "inspector": ["/usr/bin/python3", str(ROOT / "research/inspection_aware/inspection.py" if arm == "aware" else ROOT / "research/continuous_asset/inspection.py"), str(inspection)],
        "server": ["gz", "sim", "-s", "-r", "--headless-rendering", "-v", "3", str(path / "private/world.sdf")],
    }
    children = {}
    handles = []
    started = time.monotonic()
    reason = "wall_guard"
    finished = None
    ledger = (OUT / "attempts.jsonl").open("a")

    def event(kind, **kwargs):
        record = dict(event=kind, utc=utc(), case=case, arm=arm, output=str(path), **kwargs)
        ledger.write(json.dumps(record) + "\n")
        ledger.flush()
        with (path / "stages.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def start(name):
        cmd = commands[name]
        stdout = (path / (name + ".stdout")).open("w")
        stderr = (path / (name + ".stderr")).open("w")
        handles.extend((stdout, stderr))
        children[name] = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        event("process_started", name=name, pid=children[name].pid, command=cmd)

    event("attempt_started", diagnostic_simulator_reset=True, continuous_mission=False,
          map_sha256=sha(inspection / "map.txt"), selected_policy_sha256=sha(public / "policy.json"))
    try:
        for name in ("bridge", "judge", "camera", "inspector"):
            start(name)
        ready = subprocess.run(["/usr/bin/python3", str(ROOT / "research/post_repair_connected/inspection_readiness.py"),
                                "--output", str(path / "startup_readiness.json")], env=env,
                               capture_output=True, text=True, timeout=45)
        if ready.returncode:
            save(path / "startup_readiness_failure.json", dict(returncode=ready.returncode,
                                                                 stdout=ready.stdout, stderr=ready.stderr))
            raise RuntimeError("startup_readiness_failed")
        start("server")
        while time.monotonic() - started < manifest["wall_guard_s"]:
            exited = [name for name, child in children.items() if child.poll() is not None]
            if exited:
                reason = "child_exit:" + ",".join(exited)
                break
            if (inspection / "task_done.json").exists():
                if finished is None:
                    finished = time.monotonic()
                if time.monotonic() - finished >= 2:
                    reason = "inspection_finished"
                    break
            time.sleep(.1)
    except KeyboardInterrupt:
        reason = "operator_abort"
    except Exception as error:
        reason = "infrastructure_exception:" + str(error)
        save(path / "exception.json", dict(type=type(error).__name__, message=str(error)))
    finally:
        for child in children.values():
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        for child in children.values():
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for handle in handles:
            handle.close()
        run_record = dict(stop_reason=reason, wall_s=time.monotonic() - started,
                          commands=commands, simulator_launches=int("server" in children),
                          diagnostic_simulator_reset=True, continuous_mission=False,
                          reset_count=1, interventions=int(reason == "operator_abort"),
                          all_owned_processes_reaped=all(child.poll() is not None for child in children.values()),
                          exits={name: child.returncode for name, child in children.items()},
                          partition=env["GZ_PARTITION"])
        save(path / "run.json", run_record)
        event("attempt_finished", run=run_record)
        ledger.close()
    if reason == "inspection_finished":
        reply = subprocess.run(["/usr/bin/python3", str(ROOT / "research/continuous_asset/inspection_judge.py"),
                                str(inspection)], capture_output=True, text=True)
        (path / "score.stdout").write_text(reply.stdout)
        (path / "score.stderr").write_text(reply.stderr)
        if reply.returncode:
            reason = "scoring_failure"
    return path, reason


def main(args):
    manifest = load(OUT / args.manifest)
    with open("/tmp/site_localization_domain_216.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for case, arm in manifest["execution_order"]:
            if args.single is not None and [case, arm] != args.single:
                continue
            path, reason = run(case, arm, manifest, args.suffix)
            print(case, arm, reason, flush=True)
            if reason != "inspection_finished":
                raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", nargs=2, metavar=("CASE", "ARM"))
    parser.add_argument("--suffix", default="")
    parser.add_argument("--manifest", default="manifest.json")
    main(parser.parse_args())
