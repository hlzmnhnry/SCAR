# Reference data

For every selected image SCAR asks a reference provider for a *reference view*: an orthophoto and a
surface model (DSM) on a common pixel grid around the camera footprint, plus the affine map from pixels to
world coordinates. The georay provider computes the footprint by ray marching the camera frustum
against the DSM and renders the view from raster tiles.

## German federal states (georaffer)

```yaml
world:
  crs: EPSG:25832
  height: geoid
  geoid_grid: data/quasigeoid/GCG2016v2023.tif   # BKG quasigeoid, see examples/README.md
  height_name: DHHN2016
reference:
  regions: [nrw, rlp]
  imagery_from: [2018, null]      # orthophoto years to consider
  tile_cache_dir: ~/georay/tiles
  preferred_date: 2022-10-19      # rank tiles by season/year closeness to the flight
```

Needs the `germany` extra (or `pixi install -e gpu`). Tiles are downloaded once and cached; the
orthophoto closest in season and year to the flight is preferred.

## Your own rasters (any country)

```yaml
world:
  crs: EPSG:32617                 # projected CRS used for the calibration
  height: ellipsoidal             # vertical reference of the DSM
reference:
  image_files: [reference/orthophoto.tif]
  dsm_files: [reference/dsm.tif]
```

Any GDAL-readable raster in any CRS works (GeoTIFF, COG, VRT); rasters in another CRS are reprojected
to `world.crs`. Multiple files are mosaicked, NaN or nodata cells are treated as missing. The DSM must
be a *surface* model (buildings and vegetation included) whose heights are in `world.height`.
Anchors are only as accurate as the orthophoto's georeferencing and the DSM.

Possible open data sources (not tested with SCAR yet): NAIP orthophotos with USGS 3DEP lidar DSMs (US), swisstopo SWISSIMAGE
with swissSURFACE3D (CH), national orthophoto and lidar programmes of most European countries.

## Your own provider

Anything that implements `scar.references.ReferenceProvider` can be passed to
`scar.collect.AnchorCollector`:

```python
class MyProvider:
    def render(self, T_world_camera, camera, *, pixels_per_meter=None, timestamp_ns=None):
        ...
        return ReferenceView(image=ortho_uint8, dsm=heights_float, pixel_to_world=A_3x3)
```

`pixel_to_world` maps continuous pixel coordinates (area convention, see
[coordinate_systems.md](coordinate_systems.md)) to world x/y; DSM heights are in the world frame's
vertical reference.

## Licenses

Reference data keep the licenses of their providers, e.g. NRW (dl-de/zero-2.0), RLP and the BKG
quasigeoid (dl-de/by-2.0). Attribute them in publications and do not redistribute tiles unless the
license allows it.
