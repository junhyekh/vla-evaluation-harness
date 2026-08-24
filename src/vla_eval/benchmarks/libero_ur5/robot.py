"""Register robosuite's native UR5e for LIBERO's two mounting modes."""

from __future__ import annotations


def register_ur5e() -> None:
    """Register mounted and floor UR5e aliases with a LIBERO-speed Robotiq85."""
    from robosuite.models.grippers import GRIPPER_MAPPING
    from robosuite.models.grippers.robotiq_85_gripper import Robotiq85Gripper
    from robosuite.models.robots.manipulators.ur5e_robot import UR5e
    from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
    from robosuite.robots import ROBOT_CLASS_MAPPING
    from robosuite.robots.single_arm import SingleArm

    if "MountedUR5e" in REGISTERED_ROBOTS and "OnTheGroundUR5e" in REGISTERED_ROBOTS:
        return

    class LIBERORobotiq85Gripper(Robotiq85Gripper):
        @property
        def speed(self):
            # Match Panda's binary-gripper response time while retaining the
            # stock Robotiq85 geometry, joints, actuator, and contact model.
            return 0.2

    GRIPPER_MAPPING["LIBERORobotiq85Gripper"] = LIBERORobotiq85Gripper

    class MountedUR5e(UR5e):
        @property
        def default_gripper(self):
            return "LIBERORobotiq85Gripper"

        @property
        def base_xpos_offset(self):
            return {
                "bins": (-0.5, -0.1, 0),
                "empty": (-0.6, 0, 0),
                "table": lambda length: (-0.16 - length / 2, 0, 0),
                "study_table": lambda length: (-0.25 - length / 2, 0, 0),
                "kitchen_table": lambda length: (-0.16 - length / 2, 0, 0),
            }

    class OnTheGroundUR5e(MountedUR5e):
        @property
        def default_mount(self):
            return None

        @property
        def base_xpos_offset(self):
            return {
                "bins": (-0.5, -0.1, 0),
                "empty": (-0.6, 0, 0),
                "table": lambda length: (-0.16 - length / 2, 0, 0),
                "coffee_table": lambda length: (-0.16 - length / 2, 0, 0.41),
                "living_room_table": lambda length: (-0.16 - length / 2, 0, 0.42),
            }

    for name in ("MountedUR5e", "OnTheGroundUR5e"):
        ROBOT_CLASS_MAPPING[name] = SingleArm
