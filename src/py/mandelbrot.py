import os
import multiprocessing
import time
import sys
import configparser
import logging
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import matplotlib.image
from concurrent.futures import ProcessPoolExecutor
from concurrent import futures

log = logging.getLogger(__name__)


def fn_test(cx, cy, depth):
    x = 0.0
    y = 0.0
    for i in range(depth):
        # Implements the function zn = z**2 + c
        xn = x*x - y*y + cx
        yn = 2.0*x*y + cy
        # If the length of z > 2, break
        if (xn*xn + yn*yn) > 4:
            return i / depth;
        x = xn
        y = yn
    return 0


def mandelbrot(xmin:float, xmax:float, ymin:float, ymax:float, image_size:int, depth:int, n_workers:int=1, 
        tile_size:int=100):
    xx = np.linspace(xmin, xmax, image_size)
    yy = np.linspace(ymax, ymin, image_size)
    zz = compute_image(xx, yy, depth, n_workers, tile_size)
    return zz


def compute_tile(xx, yy, rmin, rmax, cmin, cmax, depth):
    """ Computes part of the image, called a tile.

    Parameters
    ----------
    xx: np.ndarray
        Array of x coordinates, representing the real values of z.
    yy: np.ndarray
        Array of y coordinates, representing the imaginary values of z.
    rmin: int
        Start Re index of the tile.
    rmax: int
        End Im index (end row) of the tile.
    """
    rows = rmax - rmin
    cols = cmax - cmin
    zz = np.empty(shape=(rows, cols))
    for i, r in enumerate(range(rmin, rmax)):
        for j, c in enumerate(range(cmin, cmax)):
            zz[i, j] = fn_test(xx[c], yy[r], depth)
    return {"zz": zz, "rmin": rmin, "rmax": rmax, "cmin": cmin, "cmax": cmax}


def compute_image(xx, yy, depth, n_workers, tile_size):
    rows = len(yy)
    cols = len(xx)
    # Divide the computational space into tiles and send it to different threads.
    row_batches = batch_ranges_with_size(rows, tile_size)
    col_batches = batch_ranges_with_size(cols, tile_size)
    # List of future objects, for looping and retrieving the results of the workers.
    future_list = []
    # List of computed image tiles.
    image_tile_list = []
    with tqdm(total=len(row_batches)*len(col_batches)) as pbar:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for row_batch in row_batches:
                for col_batch in col_batches:
                    future = executor.submit(compute_tile, xx, yy,
                                             row_batch[0], row_batch[1], 
                                             col_batch[0], col_batch[1], depth)
                    future_list.append(future)
            for future in futures.as_completed(future_list):
                image_tile_list.append(future.result())
                pbar.update(1)
    zz = np.empty(shape=(rows, cols))
    for tile in image_tile_list:
        zz[tile["rmin"]:tile["rmax"], tile["cmin"]:tile["cmax"]] = tile["zz"]
    print(np.min(zz))
    print(np.max(zz))
    return zz


def batch_ranges_with_size(N:int, batch_size:int) -> List[Tuple[int, int]]:
    """
    Computes batch ranges with given batch size and returns a list of tuples with start and end indices.

    Parameters
    ----------
    N: int
        Total amount of samples
    batch_size: int
        Batch size, e.g. 64 samples per batch

    Returns
    -------
    List[Tuple[int, int]]
        List of tuples with start and end indices, e.g. [(start0, end0), (start1, end1),...,]
    """
    rng = []
    if N <= batch_size:
        rng.append((0, N))
    else:
        n_batches = N // batch_size
        for i in range(n_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            rng.append((start, end))
        if batch_size * n_batches < N:
            start = batch_size * n_batches
            end = N
            rng.append((start, end))
    return rng


def read_config(config_path: str):
    """ Reads the config file from disk and returns a config object.

    Parameters
    ----------
    config_path: str
        Relative or full path to the config file.

    Returns
    -------
    ConfigParser
        ConfigParser holding the values from the ini file.

    """
    config = configparser.ConfigParser()
    try:
        print(f"Reading configuration from {config_path}")
        with open(config_path) as fp:
            config.read_file(fp)
            if "mandelbrot" not in config:
                log.error("[mandelbrot] section is missing from config file.")
                sys.exit(-1)
            return config["mandelbrot"]
    except Exception:
        log.exception(f"Config could not be read from {config_path}.")
        sys.exit(-1)

def save_image(data:np.ndarray, cmap:str, output_folder:str, prefix:str, filetype:str="png") -> None:
    """ Creates the resulting image on disk with a colormap of choice.

    Parameters
    ----------
    data: np.ndarray
        zz values of type float with values between 0 and 1.
    cmap: str
        Color map, as defined by matplotlib. See https://matplotlib.org/stable/tutorials/colors/colormaps.html
        for valid options.
    prefix: str
        Prefix of the filename.
    filetype: str
        Image file type. Valid options are 'jpg', 'png'.
    """
    file_path = get_file_path(output_folder, prefix, filetype)
    print(f"Writing image to {file_path}")
    matplotlib.image.imsave(file_path, data, cmap=cmap)

def save_data(data:np.ndarray, output_folder, prefix:str) -> None:
    file_path = get_file_path(output_folder, prefix, filetype="npy")
    print(f"Writing data to {file_path}")
    np.save(file_path, data)



def get_file_path(output_folder:str, prefix:str, filetype:str):
    filename = f"{prefix}.{filetype}"
    if output_folder != "":
        os.makedirs(os.path.join(output_folder), exist_ok=True)
        file_path = os.path.join(output_folder, filename)
    else:
        file_path = filename
    return file_path


if __name__ == "__main__":
    # Read the parameters from the config file.
    config = read_config("config.ini")
    xmin = config.getfloat("xmin")
    xmax = config.getfloat("xmax")
    ymin = config.getfloat("ymin")
    ymax = config.getfloat("ymax")
    depth = config.getint("depth")
    cmap = "gray"
    output_folder = config.get("output_folder")
    image_size = config.getint("image_size")
    filetype = config.get("filetype")
    # Misc settings
    timestamp = str(int(time.time()))
    prefix = f"{timestamp}_mandelbrot"
    n_cpus = multiprocessing.cpu_count()
    msg = f"=== Fractals ===\n"
    msg += f"(xmin, ymin) - (xmax, ymax) = ({xmin}, {ymin}) - ({xmax}, {ymax})\n"
    msg += f"depth: {depth}\n"
    msg += f"image size: {image_size}x{image_size}\n"
    msg += f"Number of CPUs: {n_cpus}\n"
    msg += f"Timestamp: {timestamp}\n"
    print(msg)
    # Compute the fractal.
    tic = time.time()
    zz = mandelbrot(xmin, xmax, ymin, ymax, image_size, depth, n_workers=n_cpus)
    toc = time.time()
    # Post processing.
    save_image(zz, cmap=cmap, output_folder=output_folder, prefix=prefix, filetype=filetype)
    if config.getboolean("with_raw_output", False):
        save_data(zz, output_folder, prefix)
    print(f"Computational time: {time.strftime('%H:%M:%S', time.gmtime(toc-tic))}")
