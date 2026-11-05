#!/usr/bin/env python3
"""Reproduce isolated official package extraction; never installs system packages.

Requires curl, apt metadata hashes recorded in amcl_packages.json, and dpkg-deb.
Run in an environment with authorized network access. Existing correct downloads
are reused. No package scripts are executed.
"""
import hashlib
import json
from pathlib import Path
import subprocess

vendor = Path(__file__).resolve().parent
packages = json.loads((vendor / "amcl_packages.json").read_text())
deb_dir = vendor / "debs"
deb_dir.mkdir(exist_ok=True)
for package in packages:
    target = deb_dir / (package["Package"] + ".deb")
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != package["SHA256"]:
        partial = target.with_suffix(".deb.partial")
        subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", package["url"], "--output", str(partial)], check=True)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != package["SHA256"]:
            raise RuntimeError("SHA256 mismatch for " + package["Package"])
        partial.replace(target)
    subprocess.run(["dpkg-deb", "--extract", str(target), str(vendor / "amcl_root")], check=True)
    print(package["Package"], package["Version"], "SHA256 verified")
