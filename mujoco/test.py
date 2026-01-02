import time
import mujoco
import imageio
import sys

import argparse

# Minimal MJCF: plane + free sphere above it.
# The solref trick below is from the MuJoCo docs' restitution example.
XML = r"""
<mujoco model="drop_bounce">
  <option gravity="0 0 -9.81" timestep="0.0002"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
  </visual>
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interactive", action="store_true", help="Run with interactive viewer if available")
    parser.add_argument("-o", "--output", type=str, default="output.mp4", help="Output video file")
    parser.add_argument("--max-time", type=float, default=3.0, help="Maximum simulation time in seconds")
    options = parser.parse_args()

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # Optional: give it a tiny lateral velocity so you can see motion.
    # Freejoint DOFs are 7 qpos (pos+quat) and 6 qvel (linvel+angvel).
    data.qvel[0] = 0.2  # x linear velocity

    # If you have a GUI context, this opens an interactive viewer.
    if options.interactive:
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
        except ImportError:
            print("mujoco.viewer not available; exiting")
            sys.exit(1)
    else:

        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = model.stat.center
        cam.distance = 2.0 * model.stat.extent
        cam.azimuth = 90
        cam.elevation = -20

        renderer = mujoco.Renderer(model, height=1080, width=1920)

        with imageio.get_writer(options.output, fps=60) as video:
            next_frame_time = 0.0
            for i in range(int(options.max_time / model.opt.timestep)):
                mujoco.mj_step(model, data)
                if (i * model.opt.timestep) > next_frame_time:
                    print("yolo")
                    renderer.update_scene(data, cam)
                    img = renderer.render()
                    video.append_data(img)
                    next_frame_time += 1 / 60.0

main()
