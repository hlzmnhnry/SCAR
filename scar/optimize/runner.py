"""Run a staged calibration on a sequence."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np

from scar.core.camera import CameraModel
from scar.core.types import Extrinsics, Sequence

from .extrinsics import ExtrinsicsEstimate, estimate_extrinsics, extrinsics_fit
from .graph import (
    SolverReport,
    build_stage_graph,
    calibration_covariance,
    extract_state,
    solve,
)
from .problem import CalibrationProblem, CalibrationState
from .stages import OptimizationPreset, StageSpec

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageResult:
    name: str
    state: CalibrationState
    report: SolverReport
    calibration_covariance: np.ndarray | None = None
    extrinsics_covariance: np.ndarray | None = None


@dataclass(frozen=True)
class CalibrationResult:
    problem: CalibrationProblem
    stages: tuple[StageResult, ...]
    extrinsics: ExtrinsicsEstimate | None

    @property
    def camera(self) -> CameraModel:
        return self.stages[-1].state.camera

    @property
    def T_body_camera(self) -> np.ndarray:
        if self.extrinsics is None:
            return self.problem.extrinsics.T_body_camera
        return self.extrinsics.T_body_camera

    @property
    def calibrated_extrinsics(self) -> Extrinsics:
        return Extrinsics(self.T_body_camera)

    def stage(self, name: str) -> StageResult:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)


def run_stage(problem: CalibrationProblem, state: CalibrationState, stage: StageSpec) -> StageResult:
    graph, values = build_stage_graph(problem, state, stage)
    result, report = solve(graph, values, stage.solver)
    covariance, extrinsics_covariance = (calibration_covariance(graph, result, stage.joint_extrinsics)
        if stage.compute_calibration_covariance else (None, None))

    new_state = extract_state(problem, result, template=state.camera, joint=stage.joint_extrinsics)
    log.info("stage %s: error %.6g -> %.6g in %d iterations", stage.name, report.initial_error,
        report.final_error, report.iterations)
    return StageResult(name=stage.name, state=new_state, report=report, calibration_covariance=covariance,
        extrinsics_covariance=extrinsics_covariance)


def run_calibration(sequence: Sequence, preset: OptimizationPreset) -> CalibrationResult:
    """Filter the sequence's anchors, run all stages in order and recover the extrinsics."""
    problem = CalibrationProblem.from_sequence(sequence, min_point_observations=preset.min_point_observations,
        min_frame_features=preset.min_frame_features)
    log.info("optimizing %d frames, %d landmarks, %d observations", problem.num_frames, problem.num_landmarks,
        problem.num_observations)

    state = CalibrationState.initial(problem)
    stages = []
    for stage in preset.stages:
        if stage.joint_extrinsics and state.T_body_camera is None:
            # initialize the extrinsics from the camera poses estimated so far
            initial = estimate_extrinsics(problem.T_world_body, state.T_world_camera, problem.extrinsics.T_body_camera,
                preset.extrinsics)
            state = replace(state, T_body_camera=initial.T_body_camera)
        result = run_stage(problem, state, stage)
        stages.append(result)
        state = result.state

    extrinsics = None
    if state.T_body_camera is not None:
        # the last stage estimated the extrinsics in the graph; report the fit like the LS recovery
        extrinsics = extrinsics_fit(problem.T_world_body, state.T_world_camera, state.T_body_camera)
    elif preset.extrinsics is not None:
        extrinsics = estimate_extrinsics(problem.T_world_body, state.T_world_camera,
            problem.extrinsics.T_body_camera, preset.extrinsics)

    return CalibrationResult(problem=problem, stages=tuple(stages), extrinsics=extrinsics)
