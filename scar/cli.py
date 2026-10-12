"""Command line interface: ``scar init | check | collect | optimize | run | export``."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parent / "config" / "templates"


def _viewer(config, enabled: bool):
    if not (enabled or config.viewer.enabled):
        return None
    from scar.viz.rerun_viewer import RerunViewer

    save = str(config.path(config.viewer.save)) if config.viewer.save else None
    return RerunViewer(application_id=config.name, spawn=config.viewer.spawn, save=save)


def cmd_init(args) -> int:
    source = TEMPLATES / f"{args.template}.yaml"
    target = Path(args.path)
    if target.exists() and not args.force:
        print(f"{target} exists (use --force to overwrite)", file=sys.stderr)
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    print(f"wrote {target}; edit the paths and run 'scar check {target}'")
    return 0


def cmd_check(args) -> int:
    import numpy as np

    from scar.config.schema import load_run_config
    from scar.pipeline import prepare_sequence

    config = load_run_config(args.config)
    prepared = prepare_sequence(config)
    sequence = prepared.sequence
    frames = sequence.frames
    positions = np.asarray([f.pose_prior.T_world_body[:3, 3] for f in frames])
    duration = (frames[-1].timestamp_ns - frames[0].timestamp_ns) * 1e-9
    camera = sequence.camera

    print(f"sequence   {len(frames)} frames over {duration:.1f} s, path {np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.0f} m")
    print(f"camera     {camera.width}x{camera.height}, f = ({camera.fx:.1f}, {camera.fy:.1f}), c = ({camera.cx:.1f}, {camera.cy:.1f}), "
          f"distortion {[round(float(d), 5) for d in camera.distortion]}")
    print(f"world      {sequence.world_frame.crs}, heights {sequence.world_frame.height.kind}"
          f"{' (' + sequence.world_frame.height.name + ')' if sequence.world_frame.height.name else ''}, "
          f"meridian convergence {'on' if sequence.world_frame.apply_meridian_convergence else 'off'}")
    print(f"positions  x {positions[:, 0].min():.1f}..{positions[:, 0].max():.1f}, y {positions[:, 1].min():.1f}..{positions[:, 1].max():.1f}, "
          f"height {positions[:, 2].min():.1f}..{positions[:, 2].max():.1f}")
    missing = [f.image_path for f in frames[:: max(1, len(frames) // 20)] if not prepared.images.resolve(f.image_path).exists()]
    if missing:
        print(f"images     MISSING, e.g. {prepared.images.resolve(missing[0])}")
        return 1
    print("images     found")

    if args.render:
        import cv2

        from scar.pipeline import reference_provider

        references = reference_provider(config, sequence.world_frame)
        output = config.path(config.output) / "check"
        output.mkdir(parents=True, exist_ok=True)
        for frame in (frames[0], frames[len(frames) // 2], frames[-1]):
            view = references.render(frame.pose_prior.T_world_camera(sequence.extrinsics), camera)
            image = cv2.cvtColor(prepared.read_image(frame), cv2.COLOR_GRAY2BGR)
            reference = cv2.cvtColor(view.image, cv2.COLOR_RGB2BGR)
            scale = image.shape[0] / reference.shape[0]
            reference = cv2.resize(reference, (int(reference.shape[1] * scale), image.shape[0]))
            path = output / f"frame_{frame.frame_id:05d}.png"
            cv2.imwrite(str(path), np.hstack([image, reference]))
            print(f"reference  frame {frame.frame_id}: {view.image.shape[1]}x{view.image.shape[0]} px at "
                  f"{view.metadata['pixels_per_meter']} px/m -> {path}")
    return 0


def _collect(config, args):
    from scar.pipeline import prepare_sequence, run_collect

    prepared = prepare_sequence(config)
    viewer = _viewer(config, args.viewer)
    on_frame = None
    if viewer is not None:
        sequence = prepared.sequence

        def on_frame(report):
            viewer.on_frame(report, report.frame.pose_prior.T_world_camera(sequence.extrinsics), sequence.camera)
    return run_collect(config, resume=args.resume, on_frame=on_frame, prepared=prepared), viewer


def cmd_collect(args) -> int:
    from scar.config.schema import load_run_config

    config = load_run_config(args.config)
    result, _ = _collect(config, args)
    print(f"{len(result.landmarks)} anchors, {len(result.observations)} observations -> "
          f"{config.path(config.output) / 'anchors.json'}")
    return 0


def _report_calibration(result, config) -> None:
    from scar.core.se3 import rotation_angle

    camera = result.camera
    problem = result.problem
    change = rotation_angle(result.T_body_camera[:3, :3], problem.extrinsics.T_body_camera[:3, :3])
    print(f"optimized  {problem.num_frames} frames, {problem.num_landmarks} anchors, {problem.num_observations} observations")
    print(f"intrinsics f = ({camera.fx:.2f}, {camera.fy:.2f}), c = ({camera.cx:.2f}, {camera.cy:.2f}), "
          f"distortion {[round(float(d), 5) for d in camera.distortion]}")
    print(f"extrinsics rotation changed by {change * 57.29577951308232:.3f} deg")
    print(f"written    {config.path(config.output) / 'calibration' / 'calibration.yaml'}")


def cmd_optimize(args) -> int:
    from scar.config.schema import load_run_config
    from scar.pipeline import run_optimize

    config = load_run_config(args.config)
    result = run_optimize(config)
    viewer = _viewer(config, args.viewer)
    if viewer is not None:
        viewer.on_calibration(result)
    _report_calibration(result, config)
    return 0


def cmd_run(args) -> int:
    from scar.config.schema import load_run_config
    from scar.pipeline import run_optimize

    config = load_run_config(args.config)
    anchors, viewer = _collect(config, args)
    result = run_optimize(config, anchors)
    if viewer is not None:
        viewer.on_calibration(result)
    _report_calibration(result, config)
    return 0


def cmd_evaluate(args) -> int:
    import numpy as np

    from scar.evaluate import (
        ReprojectionStatistics,
        estimate_pose_offset,
        localize,
        reprojection_errors,
    )
    from scar.io import load_camera, load_extrinsics, load_sequence

    anchors = load_sequence(args.anchors)
    camera, T = load_camera(args.calibration), load_extrinsics(args.calibration).T_body_camera
    print(f"anchors    {len(anchors.frames)} frames, {len(anchors.landmarks)} anchors, {len(anchors.observations)} observations")
    print(f"reprojection {ReprojectionStatistics.of(reprojection_errors(anchors, camera, T))}")
    if args.offsets:
        offset = estimate_pose_offset(anchors, camera, T, estimate_rotation=True)
        print(f"offsets    translation {np.round(offset.delta, 3).tolist()} m, time {1000 * offset.dt:+.1f} ms, "
              f"camera rotation {np.round(offset.omega_deg, 3).tolist()} deg (reprojection then {offset.after})")
    if args.localize:
        summary = localize(anchors, camera, T).summary()
        accuracy = ", ".join(f"{key} {value:.3f}" for key, value in summary["accuracy"].items())
        print(f"localization {summary['localized']}/{summary['frames']} frames, median rotation "
              f"{summary['median_rotation_error_deg']:.3f} deg, translation {summary['median_translation_error_m']:.2f} m; {accuracy}")
    return 0


def cmd_export(args) -> int:
    from scar.io import export_calibration, load_camera, load_extrinsics

    files = export_calibration(args.output, load_camera(args.calibration), load_extrinsics(args.calibration), args.format)
    for path in files:
        print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scar", description="Satellite-based calibration for aerial recordings")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress (INFO)")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="write a configuration template")
    init.add_argument("path")
    init.add_argument("--template", choices=sorted(p.stem for p in TEMPLATES.glob("*.yaml")), default="csv")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    check = commands.add_parser("check", help="validate a configuration and its inputs")
    check.add_argument("config")
    check.add_argument("--render", action="store_true", help="render references for three frames (needs georay)")
    check.set_defaults(func=cmd_check)

    for name, func, help_text in (("collect", cmd_collect, "collect anchors"),
            ("optimize", cmd_optimize, "optimize collected anchors"), ("run", cmd_run, "collect and optimize")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("config")
        command.add_argument("--viewer", action="store_true", help="show progress in rerun")
        if name != "optimize":
            command.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
        command.set_defaults(func=func)

    evaluate = commands.add_parser("evaluate", help="evaluate a calibration on independent anchors")
    evaluate.add_argument("anchors", help="anchors.json of a test sequence (collected with the same pose processing)")
    evaluate.add_argument("--calibration", required=True, help="calibration.yaml to evaluate")
    evaluate.add_argument("--localize", action="store_true", help="visual localization (PnP) per frame; needs "
        "per-frame anchors, e.g. collected with tracking disabled")
    evaluate.add_argument("--offsets", action="store_true", help="estimate translation/time/rotation offsets")
    evaluate.set_defaults(func=cmd_evaluate)

    export = commands.add_parser("export", help="write a calibration for other tools")
    export.add_argument("calibration", help="calibration.yaml of a run")
    export.add_argument("--format", choices=["kalibr", "opencv", "ltc"], required=True)
    export.add_argument("--output", required=True)
    export.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(name)s: %(message)s")
    try:
        return args.func(args)
    except Exception as error:
        if args.verbose:
            raise
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
