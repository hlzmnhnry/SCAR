import re
import numpy as np

from os.path import basename, splitext
from collections import OrderedDict
from datetime import datetime

class LRUCache:

    max_size: int
    cache: OrderedDict
    
    def __init__(self, max_size: int):
        
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key: str):
        
        value = self.cache.pop(key)
        self.cache[key] = value
        
        return value

    def put(self, key: str, array: np.ndarray):
        
        if key in self.cache:
            self.cache.pop(key)
        
        self.cache[key] = array
        
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def __contains__(self, key: str):
        return key in self.cache

    def __len__(self):
        return len(self.cache)

def round_if_close_array(arr: np.ndarray, tol=1e-10):
    
    rounded = np.round(arr)
    close_mask = np.abs(arr - rounded) < tol
    
    result = arr.copy()
    result[close_mask] = rounded[close_mask]
    
    return result

def extract_timestamp(filename: str):
    
    file_basename = basename(filename)
    
    # NOTE: 4Seasons campaigns are named yyyy-mm-dd-HH-MM-SS (also part of filename)
    #   but timestamp for file creation is yyyy-mm-dd_HH-MM-SS
    match = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", file_basename)
    
    if not match:
        raise ValueError(f"No timestamp found in filename: {filename}")

    timestamp_str = match.group(0)

    # check if this is really the edit timestamp (at the end)
    if not file_basename.endswith(timestamp_str + splitext(filename)[1]):
        raise AssertionError(f"Timestamp '{timestamp_str}' not at the end of filename: {file_basename}")

    return datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
