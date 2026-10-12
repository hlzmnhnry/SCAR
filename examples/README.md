# Examples

| Example | What it shows | Needs |
| --- | --- | --- |
| [`custom_region_geotiff`](custom_region_geotiff) | Complete run on a synthetic flight in UTM 17N with your own orthophoto/DSM files; compares the result with the truth | nothing (data is generated), CPU only |
| [`ltc_quickstart`](ltc_quickstart) | An LTC campaign with official German reference data (georaffer) and the corrected geodesy | LTC campaign data, GPU recommended |

```bash
python examples/custom_region_geotiff/make_demo_data.py examples/custom_region_geotiff/demo
scar run examples/custom_region_geotiff/config.yaml
python examples/custom_region_geotiff/make_demo_data.py --compare examples/custom_region_geotiff/demo
```

Start your own configuration with `scar init my_flight.yaml --template csv` (or `ltc`, `event`) and
validate it with `scar check my_flight.yaml --render` before running.

## Quasigeoid for German data

The German reference data (DOP/DSM) use DHHN2016 heights. Converting INS heights to them needs the
GCG2016 quasigeoid of the BKG (open data, dl-de/by-2.0), available at
<https://gdz.bkg.bund.de/index.php/default/quasigeoid-der-bundesrepublik-deutschland-quasigeoid.html>.
Download the GeoTIFF version and point `world.geoid_grid` to it (the LTC example expects
`data/quasigeoid/de/linux/geotiff/GCG2016v2023.tif`). Configurations that use INS heights unchanged
(`height: world`) do not need it.
