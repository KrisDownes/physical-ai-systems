"""Optional interactive viewer; requires a working graphical display."""
import mujoco
import mujoco.viewer
from environment import load_model

model=load_model()
data=mujoco.MjData(model)
mujoco.viewer.launch(model,data)
