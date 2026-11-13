"""Agent-authored deterministic smoke run; run with PYTHONPATH=src."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest  # Verify the test dependency is importable too.
from active_adaptation.plant import PlanarPlant


def rollout():
    plant = PlanarPlant()
    tips = [plant.reset((0.2, -0.4))]
    steps = []
    for command in ([0.05, 0.02], [0.2, -0.04], [-0.03, 0.06]):
        step = plant.step(command)
        tips.append(step["tip_m"])
        steps.append(step)
    return np.array(tips), steps


def main():
    tips, steps = rollout()
    repeat, _ = rollout()
    np.testing.assert_array_equal(tips, repeat)
    output = Path("results")
    output.mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(tips[:, 0], tips[:, 1], "o-")
    ax.set(xlabel="base-frame x (m)", ylabel="base-frame y (m)", title="Deterministic smoke trajectory")
    ax.axis("equal")
    fig.savefig(output / "smoke.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    summary = {"tips_m": tips.tolist(), "saturated_steps": sum(s["saturated"] for s in steps),
               "elapsed_s": steps[-1]["time_s"], "deterministic_repeat": True}
    (output / "smoke.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"numpy {np.__version__}; matplotlib {matplotlib.__version__}; pytest {pytest.__version__}")
    print("Saved results/smoke.png and results/smoke.json")


if __name__ == "__main__":
    main()
