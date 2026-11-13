"""Behavioral checks authored by the agent."""
import numpy as np
from active_adaptation.plant import PlanarPlant


def test_command_and_joint_limits_and_executed_motion():
    plant = PlanarPlant()
    plant.reset((np.pi - 0.01, 0))
    result = plant.step((1, -1))
    np.testing.assert_allclose(result["bounded_command_rad"], (0.1, -0.1))
    np.testing.assert_allclose(result["executed_rad"], (0.01, -0.08))
    assert result["saturated"]
    assert result["time_s"] == 0.1
