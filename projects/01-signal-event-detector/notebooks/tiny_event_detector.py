import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

#Create fake signal
fs = 1000
duration = 10
n = int(fs * duration)
t = np.arange(n) / fs

# Fake random noise
rng = np.random.default_rng(seed=0)
noise = rng.normal(0, 0.5, size=n)

x = noise.copy()

# Add fake event 
event_start_sec = 2
event_end_sec = 2.5
event_start_idx = int(event_start_sec * fs)
event_end_idx = int(event_end_sec * fs)
x[event_start_idx:event_end_idx] += 3

# Convert to energy 
energy = x**2

# Smooth energy with moving average to get envelope
window_ms = 200
window_samples = int(window_ms * fs / 1000)

kernel = np.ones(window_samples) / window_samples
envelope = np.convolve(energy, kernel, mode='same')

#Estimate noise floor 
median = np.median(envelope)
mad = np.median(np.abs(envelope - median))
sigma = mad * 1.4826
threshold_std = 6.0
threshold = median + threshold_std * sigma

is_event = envelope > threshold
event_idx = np.where(is_event)[0]
events = []

if len(event_idx) > 0:
    start_idx = event_idx[0]
    end_idx = event_idx[-1]
    event = {
        "sensor_id": "simulated_sensor_0",
        "measurement_type": "energy_event",
        "start_sample": int(start_idx),
        "end_sample": int(end_idx),
        "start_time_sec": float(start_idx / fs),
        "end_time_sec": float(end_idx / fs),
        "duration_sec": float((end_idx - start_idx) / fs),
        "score": float(np.max(envelope)),
        "threshold": float(threshold),
        "confidence": float(min(1.0, np.max(envelope) / (threshold + 1e-12) / 3.0)),
    }

    events.append(event)



output_dir = Path("outputs/tiny_event_detector")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "events.jsonl", "w") as f:
    for event in events:
        f.write(json.dumps(event) + "\n")

# Plotting for visualization
plt.figure(figsize=(12, 6))
plt.plot(t, x, label="raw signal")
plt.plot(t, envelope, label="energy envelope")
plt.axhline(threshold, linestyle="--", label="threshold")

for event in events:
    plt.axvspan(
        event["start_time_sec"],
        event["end_time_sec"],
        alpha=0.2,
        label="detected event",
    )

plt.xlabel("Time (seconds)")
plt.ylabel("Signal / Energy")
plt.title("Tiny Event Detector")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "tiny_event_detector.png")
plt.show()

print(f"Detected {len(events)} event(s)")
for event in events:
    print(event)
