# How to export environment

**TODO**: update environment and include installation of `pygeon`.

## Full export

1. `conda env export > environment.yml`
2. remove confusing dependency for `_x86_64-microarch-level=4=2_icelake`
3. remove circular dependency to `scar`
4. remove dependency to `image-matching-models==1.0.0` and replace by `"image-matching-models[all] @ git+https://github.com/alexstoken/image-matching-models.git@28168d251ca46253a52d306161e7a13bb0456cfc"`

## Minimum configuration

`conda env export --from-history > environment_min.yml` only lists packages that have been installed explicitly.
