"""Convert the pinned Menagerie RBY1-A model into a robosuite single arm."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


GRIPPER_XML = """<mujoco model="rby1_gripper">
  <asset>
    <mesh name="EE_BODY" file="assets/EE_BODY.obj"/>
    <mesh name="EE_FINGER_0" file="assets/EE_FINGER_0.obj"/>
    <mesh name="EE_FINGER_1" file="assets/EE_FINGER_1.obj"/>
  </asset>
  <actuator>
    <position name="right_finger_act" joint="finger_r1" ctrllimited="true" ctrlrange="-0.05 0" kp="1500" kv="120" forcelimited="true" forcerange="-80 80"/>
  </actuator>
  <worldbody>
    <body name="rby1_gripper">
      <inertial pos="0.00212703 0.00000615 -0.16897176" mass="0.46704998" fullinertia="0.00032067 0.00059559 0.00052732 0.00002383 -0.00000785 -0.00000007"/>
      <site name="ft_frame" size="0.01" rgba="1 0 0 0"/>
      <geom name="hand_visual" type="mesh" mesh="EE_BODY" pos="0 0 -0.1261" contype="0" conaffinity="0" group="1" rgba="0.27 0.27 0.27 1"/>
      <geom name="hand_collision" type="mesh" mesh="EE_BODY" pos="0 0 -0.1261" group="0"/>
      <body name="finger_r1" pos="0.003 0 -0.2001">
        <inertial pos="0.00278346 -0.00000329 -0.0257011" quat="0.999784 0 0.0208044 0" mass="0.03327" diaginertia="1.23275e-5 1.066e-5 2.22251e-6"/>
        <joint name="finger_r1" axis="-1 0 0" type="slide" range="-0.05 0" damping="5" armature="0.01"/>
        <geom name="finger_r1_visual_0" type="mesh" mesh="EE_FINGER_0" contype="0" conaffinity="0" group="1" rgba="0.75 0.75 0.75 1"/>
        <geom name="finger_r1_visual_1" type="mesh" mesh="EE_FINGER_1" contype="0" conaffinity="0" group="1" rgba="0.27 0.27 0.27 1"/>
        <geom name="finger_r1_collision_0" type="mesh" mesh="EE_FINGER_0" group="0" condim="4" friction="2 0.05 0.01"/>
        <geom name="finger_r1_collision_1" type="mesh" mesh="EE_FINGER_1" group="0" condim="4" friction="2 0.05 0.01"/>
      </body>
      <body name="finger_r2" pos="-0.003 0 -0.2001" quat="0 0 0 1">
        <inertial pos="0.00278346 -0.00000329 -0.0257011" quat="0.999784 0 0.0208044 0" mass="0.03327" diaginertia="1.23275e-5 1.066e-5 2.22251e-6"/>
        <joint name="finger_r2" axis="1 0 0" type="slide" range="0 0.05" damping="5" armature="0.01"/>
        <geom name="finger_r2_visual_0" type="mesh" mesh="EE_FINGER_0" contype="0" conaffinity="0" group="1" rgba="0.75 0.75 0.75 1"/>
        <geom name="finger_r2_visual_1" type="mesh" mesh="EE_FINGER_1" contype="0" conaffinity="0" group="1" rgba="0.27 0.27 0.27 1"/>
        <geom name="finger_r2_collision_0" type="mesh" mesh="EE_FINGER_0" group="0" condim="4" friction="2 0.05 0.01"/>
        <geom name="finger_r2_collision_1" type="mesh" mesh="EE_FINGER_1" group="0" condim="4" friction="2 0.05 0.01"/>
      </body>
      <body name="eef" pos="0 0 -0.2306">
        <site name="grip_site" size="0.01" rgba="1 0 0 0"/>
        <site name="grip_site_cylinder" size="0.005 0.1" rgba="0 1 0 0" type="cylinder"/>
        <site name="ee_x" pos="0.1 0 0" size="0.005 0.1" quat="0.707105 0 0.707108 0" rgba="1 0 0 0" type="cylinder"/>
        <site name="ee_y" pos="0 0.1 0" size="0.005 0.1" quat="0.707105 0.707108 0 0" rgba="0 1 0 0" type="cylinder"/>
        <site name="ee_z" pos="0 0 0.1" size="0.005 0.1" rgba="0 0 1 0" type="cylinder"/>
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="finger_r1" body2="finger_r2"/>
  </contact>
  <sensor>
    <force name="force_ee" site="ft_frame"/>
    <torque name="torque_ee" site="ft_frame"/>
  </sensor>
  <equality>
    <joint name="finger_coupling" joint1="finger_r1" joint2="finger_r2" polycoef="0 -1 0 0 0"/>
  </equality>
</mujoco>
"""


def prepare(model_dir: Path) -> None:
    source = model_dir / "rby1a_1.2_no_gripper.xml"
    root = ET.parse(source).getroot()
    root.set("model", "rby1_right")

    compiler = root.find("compiler")
    assert compiler is not None
    compiler.attrib.pop("meshdir", None)
    for mesh in root.iter("mesh"):
        mesh.set("file", f"assets/{mesh.get('file')}")

    defaults = root.find("default")
    if defaults is not None:
        root.remove(defaults)

    class_attrs = {
        "visual": {"group": "1", "type": "mesh", "contype": "0", "conaffinity": "0"},
        "collision": {"group": "0", "type": "mesh"},
        "in-model-collision": {"group": "0", "contype": "1", "conaffinity": "1"},
    }
    for geom in root.iter("geom"):
        class_name = geom.attrib.pop("class", None)
        for key, value in class_attrs.get(class_name, {}).items():
            geom.attrib.setdefault(key, value)
        if geom.get("mesh") is not None:
            geom.attrib.setdefault("type", "mesh")

    parent_by_child = {child: parent for parent in root.iter() for child in parent}
    for joint in list(root.iter("joint")):
        if not joint.get("name", "").startswith("right_arm_"):
            parent_by_child[joint].remove(joint)
        else:
            joint.attrib.setdefault("armature", "0.01")
            joint.attrib.setdefault("damping", "5")

    actuator = root.find("actuator")
    assert actuator is not None
    torque_limits = {
        element.get("joint"): element.get("forcerange")
        for element in actuator
        if element.get("joint", "").startswith("right_arm_")
    }
    actuator.clear()
    for index in range(7):
        joint = f"right_arm_{index}"
        force_range = torque_limits[joint]
        assert force_range is not None
        ET.SubElement(
            actuator,
            "motor",
            name=f"right_arm_{index}_torque",
            joint=joint,
            ctrllimited="true",
            ctrlrange=force_range,
        )

    eef = next(body for body in root.iter("body") if body.get("name") == "link_right_arm_6")
    ET.SubElement(
        eef,
        "camera",
        name="eye_in_hand",
        mode="fixed",
        pos="0.08 0 -0.12",
        quat="1 0 0 0",
        fovy="75",
    )

    output = model_dir / "rby1_right.xml"
    ET.ElementTree(root).write(output, encoding="unicode")
    (model_dir / "rby1_gripper.xml").write_text(GRIPPER_XML)

    check = ET.parse(output).getroot()
    assert [joint.get("name") for joint in check.iter("joint")] == [f"right_arm_{i}" for i in range(7)]
    assert len(list(check.find("actuator") or ())) == 7
    assert any(camera.get("name") == "eye_in_hand" for camera in check.iter("camera"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    prepare(parser.parse_args().model_dir)
