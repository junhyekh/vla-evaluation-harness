"""RBY1-right cross-embodiment suite contract."""

import runpy
import xml.etree.ElementTree as ET
from pathlib import Path

from vla_eval.cli.config_loader import load_config


ROOT = Path(__file__).parents[1]


def test_rby1_gripper_uses_mirrored_mesh_jaws() -> None:
    module = runpy.run_path(str(ROOT / "docker/prepare_rby1_right.py"))
    root = ET.fromstring(module["GRIPPER_XML"])

    mesh_geoms = [geom for geom in root.iter("geom") if geom.get("mesh")]
    assert mesh_geoms and all(geom.get("type") == "mesh" for geom in mesh_geoms)
    finger_r2 = root.find(".//body[@name='finger_r2']")
    assert finger_r2 is not None and finger_r2.get("quat") == "0 0 0 1"
    assert finger_r2.find("joint").get("range") == "0 0.05"
    assert root.find("equality/joint").get("polycoef") == "0 -1 0 0 0"
    assert root.find("contact/exclude").attrib == {"body1": "finger_r1", "body2": "finger_r2"}
    assert root.find(".//body[@name='eef']").get("pos") == "0 0 -0.2306"


def test_libero_rby1_object_order() -> None:
    config = load_config(str(ROOT / "configs/benchmarks/libero_rby1/object.yaml"))
    benchmark = config["benchmarks"][0]
    assert benchmark["episodes_per_task"] == 50
    assert benchmark["params"]["suite"] == "libero_object"
    assert "max_tasks" not in benchmark


def test_libero_pro_rby1_object_order() -> None:
    config = load_config(str(ROOT / "configs/benchmarks/libero_pro_rby1/object.yaml"))
    benchmarks = config["benchmarks"]
    assert [entry["params"]["suite"] for entry in benchmarks] == [
        "libero_object_object",
        "libero_object_swap",
        "libero_object_lan",
        "libero_object_task",
    ]
    assert all(entry["episodes_per_task"] == 50 for entry in benchmarks)
    assert all(entry["params"]["max_steps"] == 280 for entry in benchmarks)
