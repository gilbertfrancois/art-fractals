import time
from typing import Dict
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import avax.hydra.utils.parallel as hup
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures


T_FLOAT = np.float32
T_COMPLEX = np.complex64

def fn_test(c, max_it:int=100):
    z = 0 + 0j
    for i in range(max_it):
        zn = z**2 + c
        if np.abs(zn) > 2.0:
            return i
        z = zn
    return max_it

def get_grid(xmin, xmax, ymin, ymax, steps):
    c = np.linspace(xmin, xmax, steps)
    r = np.linspace(ymax, ymin, steps)
    _grid = []
    for i in range(steps):
        _grid.append(c + 1j*r[i])
    _grid = np.vstack(_grid)
    _grid = _grid.astype(T_COMPLEX)
    return _grid


def mandelbrot(xmin:float, xmax:float, ymin:float, ymax:float, steps:int, max_it:int, prefix:str="mandelbrot", n_workers=1, tile_size=100) -> np.ndarray:
    # Create a 2D plane with (x, y) -> (a + ib) values.
    xx = get_grid(xmin, xmax, ymin, ymax, steps)
    # Allocate memory for the output (x, y) -> z
    rows, cols = xx.shape
    if n_workers <= 1:
        out_dict = _mandelbrot_inner_loop(xx, 0, rows, 0, cols, max_it)
    else:
        out_dict = _mandelbrot_inner_loop_mp(xx, 0, rows, 0, cols, max_it, n_workers, tile_size)
    # Normalize yy
    yy = out_dict["out"]
    yy = yy / max_it
    # Save the "raw" numbers for easy postprocessing later.
    np.save(f"{prefix}.npy", yy)
    return yy


def _mandelbrot_inner_loop(xx:np.ndarray, rmin:int, rmax:int, cmin:int, cmax:int, max_it) -> Dict:
    rows = rmax - rmin
    cols = cmax - cmin
    yy = np.empty(shape=(rows, cols), dtype=T_FLOAT)
    for i, r in enumerate(range(rmin, rmax)):
        for j, c in enumerate(range(cmin, cmax)):
            yy[i,j] = fn_test(xx[r, c], max_it)
    return {"out": yy, "rmin": rmin, "rmax": rmax, "cmin": cmin, "cmax": cmax}


def _mandelbrot_inner_loop_mp(xx:np.ndarray, rmin:int, rmax:int, cmin:int, cmax:int, max_it, n_workers:int, tile_size:int) -> Dict:
    if rmin != 0:
        raise ValueError("Only the non-threaded inner loop can have a rmin=0 value.")
    # Create many slices per worker, but don't make the slices too small.
    row_batches = hup.batch_ranges_with_size(rmax, tile_size)
    col_batches = hup.batch_ranges_with_size(cmax, tile_size)
    future_list = []
    result_list = []
    with tqdm(total=len(row_batches)*len(col_batches)) as pbar:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for row_batch in row_batches:
                for col_batch in col_batches:
                    future = executor.submit(_mandelbrot_inner_loop, xx,
                                             row_batch[0], row_batch[1], col_batch[0], col_batch[1], max_it)
                    future_list.append(future)
            for future in futures.as_completed(future_list):
                result_list.append(future.result())
                pbar.update(1)
    yy = np.empty(shape=(rmax-rmin, cmax-cmin), dtype=T_FLOAT)
    for r in result_list:
        yy[r["rmin"]:r["rmax"], r["cmin"]:r["cmax"]] = r["out"]
    return {"out": yy, "rows": (rmin, rmax), "cols": (cmin, cmax)}


def save_image(image:np.ndarray, prefix:str="mandelbrot") -> None:
    for cmap in ["plasma", "prism", "magma", "CMRmap", "jet", "gist_ncar"]:
        matplotlib.image.imsave(f"{prefix}_{cmap}.png", image, cmap=cmap)


if __name__ == "__main__":
    tic = time.time()
    timestamp = str(int(time.time()))
    prefix = f"{timestamp}_mandelbrot"
    image = mandelbrot(-2, 1, -1.5, 1.5, steps=1001, max_it=100, prefix=prefix, n_workers=1)
    save_image(image, prefix=prefix)
    toc = time.time()
    print(f"Computational time: {time.strftime('%H:%M:%S', time.gmtime(toc-tic))}")
