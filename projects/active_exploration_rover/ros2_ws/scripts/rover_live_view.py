"""Read-only desktop telemetry; consumes supervisor records, never commands."""
import json
import sys
import time
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer

app=QApplication(sys.argv)
window=QWidget(); window.setWindowTitle('Rover live coverage and latency')
layout=QVBoxLayout(window); label=QLabel('Waiting for telemetry'); label.setWordWrap(True); layout.addWidget(label)
folder=Path(sys.argv[1]); events=[]; stream=None

def update():
    global stream
    if (folder/'manifest.json').exists():
        try:
            if json.loads((folder/'manifest.json').read_text()).get('ended_unix_s'): app.quit(); return
        except ValueError: pass
    if stream is None:
        try: stream=(folder/'events.jsonl').open()
        except FileNotFoundError: return
    for line in stream:
        try: events.append(json.loads(line))
        except ValueError: pass
    timing=[e['data'] for e in events if e['kind']=='timing']
    obs={e['observation_id']:e for e in timing if e['kind']=='observation_ready'}
    requests=[e for e in timing if e['kind']=='request']
    decisions=[e['wall_s']-obs[e['observation_id']]['wall_s'] for e in requests if e.get('observation_id') in obs]
    pubs={e['id']:e for e in timing if e['kind']=='command_published'}
    dispatch=[e['wall_s']-pubs[e['id']]['wall_s'] for e in timing if e['kind']=='executor_received' and e['id'] in pubs]
    def stats(v):
        if not v: return 'unavailable (n=0)'
        s=sorted(v)
        def q(p):
            r=(len(s)-1)*p; a=int(r); b=min(a+1,len(s)-1); return s[a]+(s[b]-s[a])*(r-a)
        return f'latest {v[-1]:.4f}s | median {q(.5):.4f}s | p95 {q(.95):.4f}s | max {max(s):.4f}s | n={len(s)}'
    text=(folder/'live-status.txt').read_text() if (folder/'live-status.txt').exists() else 'Starting'
    text+='\n\nObservation ready → movement request (includes harness/network/model):\n'+stats(decisions)
    text+='\n\nROS dispatch (publish → executor receive):\n'+stats(dispatch)
    text+='\n\nMCP actual return/harness receipt unavailable; ready is a pre-return proxy.'
    label.setText(text)
window.resize(900,350); window.show(); timer=QTimer();timer.timeout.connect(update);timer.start(1000);app.exec()
