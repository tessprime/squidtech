import time
import mujoco

# Minimal MJCF: plane + free sphere above it.
# The solref trick below is from the MuJoCo docs' restitution example.
XML = r"""
<mujoco model="drop_bounce">
  <option gravity="0 0 -9.81" timestep="0.0002"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.1" solref="-100000 0"/>
    </body>
  </worldbody>
</mujoco>
"""

def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # Optional: give it a tiny lateral velocity so you can see motion.
    # Freejoint DOFs are 7 qpos (pos+quat) and 6 qvel (linvel+angvel).
    data.qvel[0] = 0.2  # x linear velocity

    # If you have a GUI context, this opens an interactive viewer.
    try:
      from mujoco import viewer
      with viewer.launch_passive(model, data) as v:
        i = 0
        print("Timestep", model.opt.timestep)
        while v.is_running():
          i+=1
          mujoco.mj_step(model, data)
          v.sync()
          #print(f"Step {i}!")
          # Run roughly realtime (viewer also has its own sync, but this is fine).
          time.sleep(model.opt.timestep)
    except Exception as e:
      # Headless fallback: simulate for 3 seconds and print height samples.
      print("Viewer not available, running headless:", e)
      for i in range(int(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if i % 200 == 0:
          # ball body position in world coordinates is in data.xpos[body_id]
          ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
          z = data.xpos[ball_id][2]
          print(f"t={data.time:6.3f}  z={z:7.4f}")
