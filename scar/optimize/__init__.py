"""Staged factor-graph optimization of intrinsics, camera poses and anchors."""

from .extrinsics import ExtrinsicsEstimate, estimate_extrinsics
from .problem import CalibrationProblem, CalibrationState
from .runner import CalibrationResult, StageResult, run_calibration, run_stage
from .stages import (
    ExtrinsicsSpec,
    LandmarkPriorSpec,
    OptimizationPreset,
    PosePriorSpec,
    RobustKernel,
    StageSpec,
    load_preset,
)

__all__ = [
    "CalibrationProblem",
    "CalibrationResult",
    "CalibrationState",
    "ExtrinsicsEstimate",
    "ExtrinsicsSpec",
    "LandmarkPriorSpec",
    "OptimizationPreset",
    "PosePriorSpec",
    "RobustKernel",
    "StageResult",
    "StageSpec",
    "estimate_extrinsics",
    "load_preset",
    "run_calibration",
    "run_stage",
]
