"""Register the pinned Menagerie RBY1 right arm with robosuite 1.4."""

from __future__ import annotations

from pathlib import Path


def register_rby1_right(model_dir: str | Path) -> None:
    """Register one fixed-base RBY1-A right arm and its stock gripper."""
    import numpy as np
    from robosuite.models.grippers import GRIPPER_MAPPING
    from robosuite.models.grippers.gripper_model import GripperModel
    from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
    from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
    from robosuite.robots import ROBOT_CLASS_MAPPING
    from robosuite.robots.single_arm import SingleArm

    if "RBY1Right" in REGISTERED_ROBOTS:
        return

    model_dir = Path(model_dir)

    class RBY1Gripper(GripperModel):
        def __init__(self, idn=0):
            super().__init__(str(model_dir / "rby1_gripper.xml"), idn=idn)

        def format_action(self, action):
            assert len(action) == 1
            self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
            return self.current_action

        @property
        def dof(self):
            return 1

        @property
        def speed(self):
            # 40 mm/s finger speed across a 50 mm joint stroke at 20 Hz.
            return 0.08

        @property
        def init_qpos(self):
            return np.array([-0.05, 0.05])

        @property
        def _important_geoms(self):
            return {
                "left_finger": ["finger_r1_collision_0", "finger_r1_collision_1"],
                "right_finger": ["finger_r2_collision_0", "finger_r2_collision_1"],
                "left_fingerpad": ["finger_r1_collision_1"],
                "right_fingerpad": ["finger_r2_collision_1"],
            }

    GRIPPER_MAPPING["RBY1Gripper"] = RBY1Gripper

    class RBY1Right(ManipulatorModel):
        def __init__(self, idn=0):
            super().__init__(str(model_dir / "rby1_right.xml"), idn=idn)

        @property
        def default_mount(self):
            return None

        @property
        def default_gripper(self):
            return "RBY1Gripper"

        @property
        def default_controller_config(self):
            return "default_panda"

        @property
        def init_qpos(self):
            # Rainbow Robotics' documented ready pose for the right arm.
            return np.array([0.0, -0.087266, 0.0, -2.094395, 0.0, 1.22173, 0.0])

        @property
        def base_xpos_offset(self):
            # Keep the fixed torso below the LIBERO workspace and place the
            # right shoulder behind the tabletop objects.
            offset = (-0.25, 0.25, -0.75)

            def table(_):
                return offset

            return {
                "bins": offset,
                "empty": offset,
                "table": table,
                "study_table": table,
                "kitchen_table": table,
                "coffee_table": table,
                "living_room_table": table,
            }

        @property
        def top_offset(self):
            return np.array((0.0, 0.0, 1.75))

        @property
        def _horizontal_radius(self):
            return 0.8

        @property
        def _eef_name(self):
            return "link_right_arm_6"

        @property
        def arm_type(self):
            return "single"

    class MountedRBY1Right(RBY1Right):
        pass

    class OnTheGroundRBY1Right(RBY1Right):
        pass

    for name in ("RBY1Right", "MountedRBY1Right", "OnTheGroundRBY1Right"):
        ROBOT_CLASS_MAPPING[name] = SingleArm
