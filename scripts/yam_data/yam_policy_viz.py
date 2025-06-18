"""
Visualize YAM robot policy (single arm) joint angle control.
"""

try:
    import viser
except ImportError:
    print("ImportError: viser not found, for now:")
    print("pip install viser==0.2.23")
    exit()

import numpy as np
import time

try:
    from robot_descriptions.loaders.yourdfpy import load_robot_description
except ImportError:
    print("ImportError: robot_descriptions not found, for now:")
    print("pip install git+https://github.com/uynitsuj/robot_descriptions.py.git")
    print("[INFO] Will be changed to official repo once YAM and RBY1 are added and released to upstream")
    exit()
from viser.extras import ViserUrdf


class YAMVis:
    """
    Joint angle control visualization for YAM robot policy (single arm).
    """

    def __init__(self, action_horizon: int = 10):
        self.urdf = load_robot_description("yam_description")
        self.server = viser.ViserServer()
        self.server.scene.add_grid("/ground", width=2, height=2)
        self.action_horizon = action_horizon
        self.urdf_vis = []
        for i in range(self.action_horizon + 1):
            if i == 0:  # Current configuration is green
                viser_urdf = ViserUrdf(
                    self.server, self.urdf, root_node_name=f"/base{i}", mesh_color_override=(0.3, 0.8, 0.3)
                )
            else:  # Future time steps in the action chunk appear more purple
                viser_urdf = ViserUrdf(
                    self.server,
                    self.urdf,
                    root_node_name=f"/base{i}",
                    mesh_color_override=(0.5, (i / self.action_horizon), 0.9),
                )
            for mesh in viser_urdf._meshes:
                # Future time steps in the action chunk appear more transparent
                mesh.opacity = 1 - (i / self.action_horizon) * 0.8
            self.urdf_vis.append(viser_urdf)

    def update_configuration(self, current_cfg, predicted_action_chunk):
        """
        action_chunk: (N, action_dim), order of action_dim is determined by the urdf -- check this with function self.joint_names()

        *** Assumes joint angle configuration and predicted action chunks are:
        - in the same order as the urdf
        - absolute joint angles
        - in radians
        """
        self.urdf_vis[0].update_cfg(current_cfg)

        for i in range(predicted_action_chunk.shape[0]):
            self.urdf_vis[i + 1].update_cfg(predicted_action_chunk[i])

    def print_joint_order(self):
        print(self.urdf.joint_names)

    def run(self):
        self.server.run()


if __name__ == "__main__":
    vis = YAMVis(action_horizon=10)
    print("joint order:")
    vis.print_joint_order()

    while True:
        current_cfg = np.random.rand(6)

        action_chunk = np.zeros((10, 6))
        perturbations = np.random.randn(10, 6) * 0.1
        # Progressively add the randn to the current cfg
        for i in range(action_chunk.shape[0]):
            if i == 0:
                action_chunk[i] = current_cfg + perturbations[i]
            else:
                action_chunk[i] = action_chunk[i - 1] + perturbations[i]
        vis.update_configuration(current_cfg, action_chunk)
        time.sleep(1)