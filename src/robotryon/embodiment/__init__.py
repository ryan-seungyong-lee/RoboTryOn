"""Robot embodiments: kinematics, cameras, silhouettes and hand retargeting from a YAML spec.

``load_embodiment("my_robot.yaml")`` returns a :class:`MujocoEmbodiment`; see
``configs/embodiments/gr1.yaml`` for every field.
"""

from .base import OFFSCREEN_TOLERANCE, Embodiment
from .camera import CameraModel
from .spec import EmbodimentSpec, load_spec, parse_spec


def load_embodiment(path_or_name):
    """Embodiment from a YAML path, or from ``configs/embodiments/<name>.yaml`` (relative to
    the working directory) when given a bare name."""
    from pathlib import Path

    from .mujoco_embodiment import MujocoEmbodiment   # mujoco is an optional dependency
    p = Path(path_or_name)
    if not p.exists() and p.suffix == "":
        p = Path("configs/embodiments") / f"{path_or_name}.yaml"
    return MujocoEmbodiment(load_spec(p))


__all__ = ["Embodiment", "CameraModel", "EmbodimentSpec", "OFFSCREEN_TOLERANCE",
           "load_embodiment", "load_spec", "parse_spec"]
