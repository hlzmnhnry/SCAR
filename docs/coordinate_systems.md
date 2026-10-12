# Coordinate systems, datums and conventions

Calibrating a camera against map data only works if the INS poses and the reference data are expressed
in the same frame. Errors here look exactly like calibration errors: a heading convention error
becomes a yaw offset of the extrinsics, a height datum error a focal length or lever-arm error. This
page lists every convention SCAR uses and what you have to state about your data.

## Frames

| Frame | Axes | Notes |
| --- | --- | --- |
| World | x = grid east, y = grid north, z = up | Projected CRS of the reference data (`world.crs`); the API supports an optional local origin (`WorldFrame.origin`) |
| Navigation | NED or ENU, **true north** | Frame the INS orientation refers to (`navigation_frame`) |
| Body | as defined by your INS | The frame the INS poses describe (IMU or vehicle frame) |
| Camera | x right, y down, z forward (OpenCV) | |

Poses are 4x4 transforms `T_a_b` that map coordinates in frame `b` to frame `a`:

- `T_world_body` comes from the INS,
- `T_body_camera` is the extrinsic calibration (camera to body),
- `T_world_camera = T_world_body @ T_body_camera`.

Kalibr's `T_cam_imu` is the inverse of `T_body_camera`; SCAR converts it when reading a camchain.

## Meridian convergence

INS orientations refer to true north, while a map projection has its own grid north. The angle between
them, the meridian convergence, grows with the distance from the central meridian of the projection:
in UTM zone 32N it is about -1.37 deg at 7.2 deg E / 50.6 deg N and varies by more than a degree
along a 100 km flight.

With `world.meridian_convergence: true` (default) SCAR rotates orientations into grid axes,
`R_world_nav = Rz(gamma) @ R_enu_nav`, where `gamma` is pyproj's meridian convergence at each position.
Without the rotation the error is absorbed by the estimated extrinsics as a yaw offset, which is then
only valid for flights in the same area. The legacy pipeline (preset `cvprw2026`) does not apply the correction, and the
extrinsics of the LTC dataset belong to its uncorrected pose convention (use them with uncorrected
poses). For calibrations estimated with SCAR the correction lowers the test reprojection error by 21 %
on average over six LTC campaigns (see `paper/cvprw2026/ablation_geodesy.md`).

## Horizontal datum

State the geographic CRS of the INS positions explicitly (`crs` of the pose format), e.g. `EPSG:4979`
(WGS 84) or `EPSG:4937` (ETRS89). SCAR transforms them with pyproj using the best available
transformation and refuses ballpark transformations.

The datum is determined by the GNSS corrections, not by the receiver: RTK corrections from a national
service such as SAPOS (Germany) yield ETRS89 coordinates, PPP or SBAS solutions a current ITRF/WGS 84
realization. In Europe the two differ by about 0.8 m (2024) and drift by 2.5 cm per year. PROJ treats
WGS 84 and ETRS89 as identical unless you configure a time-dependent transformation, so a mismatch is
not corrected automatically: choose the CRS that matches your corrections and the reference data.
For the LTC dataset (SAPOS) this is ETRS89.

## Heights

The world frame has an explicit vertical reference (`world.height`): `ellipsoidal`, or `geoid` with a
PROJ-readable grid (`world.geoid_grid`), for heights above a geoid or quasigeoid as in most national
surface models. INS heights can be given as

- `ellipsoidal`: above the ellipsoid of the horizontal datum,
- `geoid`: above a geoid given by a grid, `h = H + N(lat, lon)`,
- `undulation`: above the geoid model of the INS, with the undulation `N` per sample in a column
  (SBG and Novatel logs), `h = H + N`,
- `world`: already in the world's vertical reference, used unchanged (how the legacy pipeline treated
  INS heights).

German DOP/DSM data use DHHN2016 heights; converting to them needs the GCG2016 quasigeoid of the BKG
(see [examples/README.md](../examples/README.md)). Differences between geoid models are typically
decimetres; between ellipsoidal and normal heights about 50 m in Germany.

The INS position refers to its own reference point; a lever arm to the camera is part of
`T_body_camera`. A cheap sanity check of heights is a standstill on the ground: the INS height minus the
DSM height should equal the mounting height of the INS reference point.

## Pixel coordinates

- **Images:** OpenCV convention, the centre of the top-left pixel is (0, 0).
- **Reference rasters:** area convention, pixel (i, j) covers `[i, i+1) x [j, j+1)`; its centre is at
  (i + 0.5, j + 0.5). Keypoints detected in a rendered orthophoto are shifted by half a pixel before they
  are mapped to world coordinates. The legacy pipeline did not do this (finding W10 in the ablation).
- **Surface model:** heights are looked up at the anchor position, by nearest pixel (legacy) or bilinear
  interpolation between pixel centres.

## Time

Timestamps can be given in s, ms, us or ns and are stored as integer nanoseconds; decimal strings are
converted exactly. Poses are interpolated to `image time + time_offset_s`. For reconstructed event
frames, name or list the frames by the time the reconstruction refers to (usually the end of the
event window).
