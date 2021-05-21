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

T_FLOAT = np.float64

def fn(x, y, cx, cy):
    xn = x**2 - y**2 + cx
    yn = -2*x*y + cy
    return xn, yn


def fn_test(cx, cy, max_it):
    # Avoid implicit conversion
    x = T_FLOAT(0.0)
    y = T_FLOAT(0.0)
    a = T_FLOAT(2.0)
    for i in range(max_it):
        # Implements the function zn = z**2 + c
        xn = x*x - y*y + cx
        yn = a*x*y + cy
        # If the length of z > 2, break
        if xn*xn + yn*yn > 16:
            return i
        x = xn
        y = yn
    return max_it


def mandelbrot(xmin, xmax, ymin, ymax, steps, max_it,
        prefix="mandelbrot", n_workers=1, tile_size=100):
    xx = np.linspace(xmin, xmax, steps, dtype=T_FLOAT)
    yy = np.linspace(ymax, ymin, steps, dtype=T_FLOAT)
    if n_workers <= 1:
        zz = compute_image(xx, yy, max_it)
    else:
        zz = compute_image_mp(xx, yy, max_it, n_workers, tile_size)
    np.save(f"{prefix}.npy", zz)
    return zz


def _inner_loop(xx, yy, rmin, rmax, cmin, cmax, max_it):
    rows = rmax - rmin
    cols = cmax - cmin
    zz = np.empty(shape=(rows, cols), dtype=T_FLOAT)
    for i, r in enumerate(range(rmin, rmax)):
        for j, c in enumerate(range(cmin, cmax)):
            zz[i, j] = fn_test(xx[c], yy[r], max_it)
    return {"zz": zz, "rmin": rmin, "rmax": rmax, "cmin": cmin, "cmax": cmax}


def compute_image(xx, yy, max_it):
    rows = len(yy)
    cols = len(xx)
    out = _inner_loop(xx, yy, 0, rows, 0, cols, max_it)
    zz = out["zz"]
    # Normalize the output
    zz = zz / max_it
    return zz


def compute_image_mp(xx, yy, max_it, n_workers, tile_size):
    # Create many slices per worker, but don't make the slices too small.
    rows = len(yy)
    cols = len(xx)
    row_batches = hup.batch_ranges_with_size(rows, tile_size)
    col_batches = hup.batch_ranges_with_size(cols, tile_size)
    future_list = []
    result_list = []
    with tqdm(total=len(row_batches)*len(col_batches)) as pbar:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for row_batch in row_batches:
                for col_batch in col_batches:
                    future = executor.submit(_inner_loop, xx, yy,
                                             row_batch[0], row_batch[1], 
                                             col_batch[0], col_batch[1], max_it)
                    future_list.append(future)
            for future in futures.as_completed(future_list):
                result_list.append(future.result())
                pbar.update(1)
    zz = np.empty(shape=(rows, cols), dtype=T_FLOAT)
    for r in result_list:
        zz[r["rmin"]:r["rmax"], r["cmin"]:r["cmax"]] = r["zz"]
    # Normalize the output
    zz = zz / max_it
    return zz

def save_image(image:np.ndarray, prefix:str="mandelbrot") -> None:
    for cmap in ["prism", "magma", "flag", "jet"]:
        matplotlib.image.imsave(f"{prefix}_{cmap}.png", image, cmap=cmap)


if __name__ == "__main__":
    timestamp = str(int(time.time()))
    prefix = f"{timestamp}_mandelbrot"
    tic = time.time()
    image = mandelbrot(-2, 1, -1.5, 1.5, steps=4*1024, max_it=100, prefix=prefix, n_workers=16)
    toc = time.time()
    save_image(image, prefix=prefix)
    print(f"Computational time: {time.strftime('%H:%M:%S', time.gmtime(toc-tic))}")
