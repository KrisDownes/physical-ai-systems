import json
from pathlib import Path
import numpy as np

path = Path("output/tiny_kalman_1d/states.jsonl")

records = []

with open(path, "r") as f:
    for line in f:
        records.append(json.loads(line))

errors = np.array([record["position_error"] for record in records])
uncertainty = np.array([record["position_uncertainty"] for record in records])
rmse = np.sqrt(np.mean(errors**2))
mean_uncertainty = np.mean(uncertainty)

print(f"Loaded {len(records)} records from {path}")
print(f"RMSE of position estimates: {rmse:.3f}")
print(f"Mean position uncertainty (σ): {mean_uncertainty:.3f}")

worst = max(records, key=lambda r: abs(r["position_error"]))
print("\nWorst estimate:")
print(f"Time: {worst['time_sec']:.2f}s")
print(f"True position: {worst['true_position']:.3f}")
print(f"Measurement: {worst['measured_position']:.3f}")
print(f"Estimate: {worst['estimated_position']:.3f}")
print(f"Error: {worst['position_error']:.3f}")