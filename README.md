<table align="center"><tr><td align="center" width="9999">
<img src="assets/scar_icon.png" align="center" width="150" alt="SCAR icon">

# SCAR

**S**atellite-based **C**alibration for **A**erial **R**ecordings
</td></tr></table>

This repository contains a tool for fine-tuning / optimizing a calibration of a visual-inertial sensor system in an aerial scenario based on satellite images. The tool is based on factor graphs for optimization and this repository contains code for the whole workflow: from data collection to optimization.

⚠️ As of now, the repository is published in an **initial version** and is currently refined. ⚠️

More will follow within the next days... 

## What it does

- Collect feature correspondences between aerial image sequences and georeferenced satellite orthophotos.
- Estimate camera intrinsics/extrinsics and GNSS/IMU alignment using factor-graph optimization (GTSAM).
- Validate the resulting calibration via PnP, reprojection error analysis, and structure-from-motion refinements.

## Contributing

The codebase is still being refined. After refinement, contributions via issues or PRs are welcome.

## Citation

If you use SCAR in academic work, please cite the accompanying paper.
