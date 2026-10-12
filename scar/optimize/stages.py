"""Declarative description of the staged optimization (priors, kernels, solver)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PRESET_DIRECTORY = Path(__file__).resolve().parent.parent / "config" / "presets"

CALIBRATION_PARAMETERS = ("fx", "fy", "skew", "cx", "cy", "k1", "k2", "p1", "p2")  # GTSAM Cal3DS2 order


def _triple(value, name: str) -> tuple[float, float, float]:
    values = tuple(float(v) for v in value)
    if len(values) != 3:
        raise ValueError(f"{name} needs three values")
    return values


@dataclass(frozen=True)
class RobustKernel:
    type: str | None = None  # 'huber', 'cauchy', 'geman_mcclure' or None
    param: float = 1.0

    def __post_init__(self) -> None:
        if self.type not in (None, "huber", "cauchy", "geman_mcclure"):
            raise ValueError(f"Unknown robust kernel {self.type!r}")


@dataclass(frozen=True)
class PosePriorSpec:
    """Prior on camera poses, centered at the stage's input poses.

    ``axes='camera'`` interprets the sigmas in camera axes (GTSAM's local pose coordinates; the
    legacy behaviour, which makes 'z' the optical axis). ``axes='world'`` interprets them in world
    axes. With ``add_navigation_covariance`` the per-frame navigation covariance is added.
    """

    rotation_sigma_deg: tuple[float, float, float]
    position_sigma_m: tuple[float, float, float]
    axes: str = "camera"
    add_navigation_covariance: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation_sigma_deg", _triple(self.rotation_sigma_deg, "rotation_sigma_deg"))
        object.__setattr__(self, "position_sigma_m", _triple(self.position_sigma_m, "position_sigma_m"))
        if self.axes not in ("camera", "world"):
            raise ValueError("Pose prior axes must be 'camera' or 'world'")


@dataclass(frozen=True)
class LandmarkPriorSpec:
    """Prior on landmarks (world axes).

    ``center``: ``state`` = the stage's input landmarks (staged schedule), ``anchors`` = the measured
    anchor positions. ``use_landmark_covariance`` uses the per-anchor covariances instead of ``sigma_m``.
    """

    sigma_m: tuple[float, float, float] | None = None
    use_landmark_covariance: bool = False
    center: str = "state"

    def __post_init__(self) -> None:
        if self.sigma_m is None and not self.use_landmark_covariance:
            raise ValueError("A landmark prior needs sigma_m or use_landmark_covariance")
        if self.center not in ("state", "anchors"):
            raise ValueError("Landmark prior center must be 'state' or 'anchors'")
        if self.sigma_m is not None:
            object.__setattr__(self, "sigma_m", _triple(self.sigma_m, "sigma_m"))


@dataclass(frozen=True)
class StageSpec:
    name: str
    kernel: RobustKernel = field(default_factory=RobustKernel)
    reprojection_sigma_px: tuple[float, float] = (1.0, 1.0)
    pose_prior: PosePriorSpec | None = None
    landmark_prior: LandmarkPriorSpec | None = None
    calibration_prior: dict[str, float] | None = None
    solver: str = "dogleg"
    compute_calibration_covariance: bool = False
    # joint stage: body poses B_t (with ``pose_prior`` around the navigation poses) and the extrinsics
    # E are variables, tied to the camera poses by C_t = B_t E
    joint_extrinsics: bool = False
    extrinsics_prior: PosePriorSpec | None = None

    def __post_init__(self) -> None:
        if self.solver not in ("dogleg", "lm", "gn"):
            raise ValueError(f"Unknown solver {self.solver!r}")
        if self.joint_extrinsics and (self.pose_prior is None or self.extrinsics_prior is None):
            raise ValueError(f"Joint stage '{self.name}' needs pose_prior (navigation) and extrinsics_prior")
        if self.calibration_prior is not None:
            missing = set(CALIBRATION_PARAMETERS) - set(self.calibration_prior)
            if missing:
                raise ValueError(f"Calibration prior of stage '{self.name}' lacks {sorted(missing)}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageSpec:
        data = dict(data)
        kernel = data.pop("kernel", None)
        pose_prior = data.pop("pose_prior", None)
        landmark_prior = data.pop("landmark_prior", None)
        extrinsics_prior = data.pop("extrinsics_prior", None)
        return cls(
            kernel=RobustKernel(**kernel) if kernel else RobustKernel(),
            pose_prior=PosePriorSpec(**pose_prior) if pose_prior else None,
            extrinsics_prior=PosePriorSpec(**extrinsics_prior) if extrinsics_prior else None,
            landmark_prior=LandmarkPriorSpec(**landmark_prior) if landmark_prior else None,
            reprojection_sigma_px=tuple(data.pop("reprojection_sigma_px", (1.0, 1.0))),
            **data,
        )


@dataclass(frozen=True)
class ExtrinsicsSpec:
    """Recovery of ``T_body_camera`` from optimized camera poses and navigation poses."""

    method: str = "least_squares"
    solver: dict[str, Any] = field(default_factory=lambda: {"method": "trf", "loss": "huber", "jac": "3-point"})
    weight_translation: float = 1.0
    weight_rotation: float = 1.0

    def __post_init__(self) -> None:
        if self.method != "least_squares":
            raise ValueError(f"Unknown extrinsics method {self.method!r}")


@dataclass(frozen=True)
class OptimizationPreset:
    name: str
    stages: tuple[StageSpec, ...]
    min_point_observations: int = 1
    min_frame_features: int = 1
    extrinsics: ExtrinsicsSpec | None = field(default_factory=ExtrinsicsSpec)
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationPreset:
        filtering = data.get("filtering", {})
        extrinsics = data.get("extrinsics", {})
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            min_point_observations=int(filtering.get("min_point_observations", 1)),
            min_frame_features=int(filtering.get("min_frame_features", 1)),
            stages=tuple(StageSpec.from_dict(stage) for stage in data["stages"]),
            extrinsics=None if extrinsics is None else ExtrinsicsSpec(**extrinsics),
        )


def _preset_data(name_or_path: str | Path) -> dict[str, Any]:
    path = Path(name_or_path)
    if not path.suffix:
        path = PRESET_DIRECTORY / f"{name_or_path}.yaml"
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    base = data.pop("extends", None)
    if base is None:
        return data
    # 'extends: <preset>' takes the base preset and appends 'stages'; other keys override
    merged = _preset_data(base)
    merged["stages"] = list(merged["stages"]) + list(data.pop("stages", []))
    merged.update(data)
    return merged


def load_preset(name_or_path: str | Path) -> OptimizationPreset:
    """Load a bundled preset by name (e.g. ``cvprw2026``) or a preset YAML file."""
    return OptimizationPreset.from_dict(_preset_data(name_or_path))
