import cv2
import glob
import time
import laspy
import requests
import numpy as np

from osgeo import gdal
from os import makedirs
from functools import partial
from typing import List, Callable
from scipy.interpolate import griddata
from os.path import join, isfile, basename

gdal.UseExceptions()

UTM_ZONE_NUMBER = 32

# change this if tile service changes
# currently this applies for all NRW services (DOP, DOM, DGM, bDOM)
SINGLE_TILE_WIDTH = 1_000   # one tile is 1km wide
SINGLE_TILE_HEIGHT = 1_000  # one tile is 1km high

# change this if tile service changes
DOP_RESOLUTION = 10 # 1m on ground is covered by xx pixels
BDOM_RESOLUTION = 2 # 1m on ground is covered by xx pixels

# folder to save temporary output
TEMP_FOLDER = "./data/temp"

def robust_download_file(url: str, destination: str, max_retries=3, timeout=10, wait_seconds=5,
    print_log: bool = False):
    
    for attempt in range(1, max_retries + 1):
        
        try:
            if print_log:
                print(f"[{attempt}/{max_retries}] Downloading: {url}")
            response = requests.get(url, timeout=timeout)

            if response.status_code == 404:
                if print_log:
                    print(f"File not found (404): {url}")
                return None

            response.raise_for_status()

            with open(destination, "wb") as f:
                f.write(response.content)

            if print_log:
                print(f"Successfully downloaded {basename(destination)}")

            return destination

        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"Download failed: {e}")
            if attempt < max_retries:
                print(f"Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
            else:
                print(f"Giving up after {max_retries} attempts.")
                return None

        except requests.HTTPError as e:
            print(f"HTTP error: {e}")
            return None

def download_dgm1_tiff(utm_x, utm_y, year=2022, utm_zone=32, overwrite=False,
    year_independent_return=True, print_log=False):

    east = (int(utm_x // SINGLE_TILE_WIDTH) * SINGLE_TILE_WIDTH) // 1000
    north = (int(utm_y // SINGLE_TILE_HEIGHT) * SINGLE_TILE_HEIGHT) // 1000
    
    if year_independent_return:
        filename_year_independent = f"dgm1_{utm_zone}_{east}_{north}_1_nw_*.tif"
        existing_files = glob.glob(join(TEMP_FOLDER, filename_year_independent))
        # return matching files, independent if it matches this year
        if existing_files:
            return existing_files[0]

    filename = f"dgm1_{utm_zone}_{east}_{north}_1_nw_{year}.tif"
    url = f"https://www.opengeodata.nrw.de/produkte/geobasis/hm/dgm1_tiff/dgm1_tiff/{filename}"
    destination = join(TEMP_FOLDER, filename)

    if not overwrite and isfile(destination):
        if print_log:
            print(f"File already exists: {destination}")
        return destination

    return robust_download_file(url, destination, print_log=print_log)

def download_dom1_tiff(utm_x, utm_y, year=2022, utm_zone=32, overwrite=False,
    year_independent_return=True, print_log=False):

    east = (int(utm_x // SINGLE_TILE_WIDTH) * SINGLE_TILE_WIDTH) // 1000
    north = (int(utm_y // SINGLE_TILE_HEIGHT) * SINGLE_TILE_HEIGHT) // 1000
    
    if year_independent_return:
        filename_year_independent = f"dom1_{utm_zone}_{east}_{north}_1_nw_*.tif"
        existing_files = glob.glob(join(TEMP_FOLDER, filename_year_independent))
        # return matching files, independent if it matches this year
        if existing_files:
            return existing_files[0]

    filename = f"dom1_{utm_zone}_{east}_{north}_1_nw_{year}.tif"
    url = f"https://www.opengeodata.nrw.de/produkte/geobasis/hm/dom1_tiff/dom1_tiff/{filename}"
    destination = join(TEMP_FOLDER, filename)

    if not overwrite and isfile(destination):
        if print_log:
            print(f"File already exists: {destination}")
        return destination

    return robust_download_file(url, destination, print_log=print_log)

def download_bdom50_las(utm_x, utm_y, year=2023, utm_zone=32, overwrite=False,
    year_independent_return=True, print_log=False):

    east = (int(utm_x // SINGLE_TILE_WIDTH) * SINGLE_TILE_WIDTH) // 1000
    north = (int(utm_y // SINGLE_TILE_HEIGHT) * SINGLE_TILE_HEIGHT) // 1000

    if year_independent_return:
        filename_year_independent = f"bdom50_{utm_zone}{east}_{north}_1_nw_*.laz"
        existing_files = glob.glob(join(TEMP_FOLDER, filename_year_independent))
        # return matching files, independent if it matches this year
        if existing_files:
            return existing_files[0]

    # NOTE: it seems to be the case that they forgot the '_' after the utm_zone
    filename = f"bdom50_{utm_zone}{east}_{north}_1_nw_{year}.laz"
    url = f"https://www.opengeodata.nrw.de/produkte/geobasis/hm/bdom50_las/bdom50_las/{filename}"
    destination = join(TEMP_FOLDER, filename)

    if not overwrite and isfile(destination):
        if print_log:
            print(f"File already exists: {destination}")
        return destination

    return robust_download_file(url, destination, print_log=print_log)

def download_als(utm_x, utm_y, utm_zone=32, overwrite=False, print_log=False):

    east = (int(utm_x // SINGLE_TILE_WIDTH) * SINGLE_TILE_WIDTH) // 1000
    north = (int(utm_y // SINGLE_TILE_HEIGHT) * SINGLE_TILE_HEIGHT) // 1000
    
    filename = f"3dm_{utm_zone}_{east}_{north}_1_nw.laz"
    url = f"https://www.opengeodata.nrw.de/produkte/geobasis/hm/3dm_l_las/3dm_l_las/{filename}"
    
    makedirs(TEMP_FOLDER, exist_ok=True)
    destination = join(TEMP_FOLDER, filename)

    if not overwrite and isfile(destination):
        if print_log:
            print(f"File already exists: {destination}")
        return destination

    return robust_download_file(url, destination, print_log=print_log)

def download_dop_nrw_jp2(utm_x, utm_y, year=2022, utm_zone=32, overwrite=False,
    year_independent_return=True, print_log=False):

    east = (int(utm_x // SINGLE_TILE_WIDTH) * SINGLE_TILE_WIDTH) // 1000
    north = (int(utm_y // SINGLE_TILE_HEIGHT) * SINGLE_TILE_HEIGHT) // 1000
    
    if year_independent_return:
        filename_year_independent = f"dop10rgbi_{utm_zone}_{east}_{north}_1_nw_*.jp2"
        existing_files = glob.glob(join(TEMP_FOLDER, filename_year_independent))
        # return matching files, independent if it matches this year
        if existing_files:
            return existing_files[0]

    filename = f"dop10rgbi_{utm_zone}_{east}_{north}_1_nw_{year}.jp2"
    url = f"https://www.opengeodata.nrw.de/produkte/geobasis/lusat/akt/dop/dop_jp2_f10/{filename}"
    destination = join(TEMP_FOLDER, filename)

    if not overwrite and isfile(destination):
        if print_log:
            print(f"File already exists: {destination}")
        return destination

    return robust_download_file(url, destination, print_log=print_log)

def try_download_with_different_years(func: Callable, year_from=2015, year_to=2025,
    reverse=True):

    fname = None
    year = year_to if reverse else year_from

    while fname is None and year in range(year_from, year_to + 1):
        fname = func(year=year)
        year += (-1) if reverse else 1
        
    return fname

def download_grid_floor_with_different_years(func: Callable, utm_x_base: float,
    utm_y_base: float, NX: int, NY: int, year_from=2015, year_to=2025, reverse=True,
    x_step=SINGLE_TILE_WIDTH, y_step=SINGLE_TILE_HEIGHT):

    filenames = []
    
    for nx in range(NX):
        for ny in range(NY):

            x = utm_x_base + nx * x_step
            y = utm_y_base + ny * y_step
            
            fname = try_download_with_different_years(
                partial(func, utm_x=x, utm_y=y),
                year_from=year_from, year_to=year_to, reverse=reverse)
            
            filenames.append(fname)

    assert len(filenames) == NX * NY

    return filenames

def download_grid(func: Callable, utm_x: float, utm_y: float,
    NX=3, NY=3, x_step=1_000, y_step=1_000):
    
    assert NX % 2 == 1 and NY % 2 == 1
    
    filenames = []
    
    for nx in range(-NX//2 + 1, NX//2 + 1):
        for ny in range(-NY//2 + 1, NY//2 + 1):

            x = utm_x + nx * x_step
            y = utm_y + ny * y_step
            
            fname = partial(func, utm_x=x, utm_y=y)
            
            filenames.append(fname)

    assert len(filenames) == NX * NY

    return filenames  

def extract_elevation_from_tiff(tiff_file: str, x: float, y: float):
    
    dataset = gdal.Open(tiff_file)

    if dataset is None:
        raise IOError(f"Could not open TIFF file: {tiff_file}")

    transform = dataset.GetGeoTransform()

    # origin is in the upper left corner (NOTE: y is mirrored)
    # pixel height and width indicate the true range the refer to
    # in case of a DGM/DOM 1 it is 1.0 meter
    origin_x, pixel_width, _, origin_y, _, pixel_height = transform

    tiff_width = dataset.RasterXSize
    tiff_height = dataset.RasterYSize
    
    assert origin_x <= x <= origin_x + tiff_width * pixel_width
    assert origin_y >= y >= origin_y + tiff_height * pixel_height # NOTE: pixel height neg.

    # get the pixel coordinate for given UTM coordinate
    pixel_x = round((x - origin_x) / pixel_width)
    pixel_y = round((y - origin_y) / pixel_height)

    assert pixel_x >= 0 and pixel_y >= 0, "Pixel position(s) must be positive"

    # NOTE: workaround if out of scope
    if pixel_x == tiff_width:
        pixel_x -= 1
    if pixel_y == tiff_height:
        pixel_y -= 1

    # read first channel (height)
    band = dataset.GetRasterBand(1)
    height = band.ReadAsArray()[pixel_y, pixel_x]

    return height

def fill_bdom_grid(grid, method="linear"):

    ny, nx = grid.shape
    y, x = np.mgrid[0:ny, 0:nx]
    
    mask = np.isfinite(grid)
    
    interpolated = griddata(
        points=(x[mask], y[mask]),
        values=grid[mask],
        xi=(x, y),
        method=method
    )
    
    # 'linear' might fail -> use 'nearest' as fallback
    if np.isnan(interpolated).any():
        
        interpolated_nearest = griddata(
            points=(x[mask], y[mask]),
            values=grid[mask],
            xi=(x, y),
            method='nearest'
        )
        
        nan_idx = np.isnan(interpolated)
        interpolated[nan_idx] = interpolated_nearest[nan_idx]

    filled = grid.copy()
    nan_mask = ~mask
    filled[nan_mask] = interpolated[nan_mask]
    
    return filled

def bdom_las_to_numpy(bdom_file: str, missing_cells_threshold=2000):
    
    las = laspy.read(bdom_file)
    resolution = 1 / BDOM_RESOLUTION
    
    assert las.xyz.shape[1] == 3, "Expected 3-dimensional array for 'xyz'"
    
    xmin, ymin = np.min(las.xyz[:, 0]), np.min(las.xyz[:, 1])
    
    i = np.floor((las.x - xmin) / resolution)
    j = np.floor((las.y - ymin) / resolution)
    
    assert np.all(np.equal(np.mod(i, 1), 0)) and np.all(np.equal(np.mod(j, 1), 0)), \
        "Number of rows and columns is expected to be an integer"
    
    i, j = i.astype(int), j.astype(int)
    j = j.max() - j

    nx, ny = i.max() + 1, j.max() + 1
    flat = np.full(nx * ny, -np.inf, dtype=float)
    idx = j * nx + i

    np.maximum.at(flat, idx, las.xyz[:, 2])

    grid = flat.reshape((ny, nx))
    grid[grid == -np.inf] = np.nan

    if np.isnan(grid).any():
        
        num_missing = np.isnan(grid).sum()
        print(f"Warning: Grid contains {num_missing} missing cells ({grid.size} cells in total)")
        
        if num_missing > missing_cells_threshold:
            raise ValueError(f"More than {missing_cells_threshold} cells do not have elevation data in bDOM")
        else:
            grid = fill_bdom_grid(grid)
    
    return grid

def tiff_to_numpy(tiff_file: str):
    
    dataset = gdal.Open(tiff_file)
    
    transform = dataset.GetGeoTransform()

    # origin is in the upper left corner (NOTE: y is mirrored)
    # pixel height and width indicate the true range the refer to
    # in case of a DGM/DOM 1 it is 1.0 meter
    origin_x, pixel_width, _, origin_y, _, pixel_height = transform
    
    tiff_width = dataset.RasterXSize
    tiff_height = dataset.RasterYSize

    start_x, start_y = 0, 0
    block_width = abs(int(tiff_width/pixel_width))
    block_height = abs(int(tiff_height/pixel_height))
    
    # just a sanity check for current data
    assert block_width == 1000
    assert block_height == 1000

    band = dataset.GetRasterBand(1)
    block = band.ReadAsArray(start_x, start_y, block_width, block_height)
    
    return block

def stack_to_grid(fnames: List[str], NX: int = 3, NY: int = 3):
    
    assert NX % 2 == 1 and NY % 2 == 1
    assert len(fnames) == NX * NY
    
    fnames_index = 0
    grid = []
    
    for _ in range(NX):
        column_images = []
        for _ in range(NY):
            image = cv2.imread(fnames[fnames_index])
            column_images.append(image)
            fnames_index += 1
            
        row = np.vstack(list(reversed(column_images)))
        grid.append(row)
        
    return np.hstack(grid)

def stack_to_grid_tiff(fnames: List[str], NX: int, NY: int):

    assert len(fnames) == NX * NY
    
    fnames_index = 0
    grid = []
    
    for _ in range(NX):
        column_images = []
        for _ in range(NY):
            image = tiff_to_numpy(fnames[fnames_index])
            column_images.append(image)
            fnames_index += 1
            
        row = np.vstack(list(reversed(column_images)))
        grid.append(row)
        
    return np.hstack(grid)

def stack_to_grid_las(fnames: List[str], NX: int, NY: int):

    assert len(fnames) == NX * NY
    
    fnames_index = 0
    grid = []
    
    for _ in range(NX):
        column_images = []
        for _ in range(NY):
            image = bdom_las_to_numpy(fnames[fnames_index])
            column_images.append(image)
            fnames_index += 1
            
        column = np.vstack(list(reversed(column_images)))
        grid.append(column)
        
    return np.hstack(grid)
