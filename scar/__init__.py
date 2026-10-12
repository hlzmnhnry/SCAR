"""SCAR: satellite-based calibration for aerial recordings.

Typical use goes through a run configuration (``scar init``/``scar run``); the library API:

- :mod:`scar.core` data model (camera, poses, frames, anchors), :mod:`scar.geodesy` world frames,
- :mod:`scar.datasets` / :mod:`scar.sources` inputs, :mod:`scar.references` reference data,
- :mod:`scar.collect` anchor collection, :mod:`scar.optimize` calibration, :mod:`scar.evaluate` metrics.
"""

__version__ = "2.0.0"
