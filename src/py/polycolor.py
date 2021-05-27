import numpy as np
import matplotlib.image


class PolyColor:
    def __init__(self, order):
        self.order = order
        self.init_colors = self._init_colors()

    def _init_colors(self):
        # 3 channels (r, g, b) and per channel 'order + 1' parameters. E.g. 2nd order, has 3 parameters.
        self.color = np.random.rand(3, self.order + 1)

    def apply(self, data):
        self._init_colors()
        if data.ndim != 2:
            raise ValueError(f"Wrong dimensions. Expected 2D, actual {data.ndim}D.")
        buffer = np.empty(shape=(data.shape[0], data.shape[1], 3))
        for channel in range(3):
            buffer[:, :, channel] = self._color_map_channel(data, channel)
        buffer = self._cvtcolor(buffer)
        return buffer

    def apply_from_npy(self, src_filepath, dst_filepath):
        data = np.load(src_filepath)
        buffer = self.apply(data)
        matplotlib.image.imsave(dst_filepath, buffer)

    def _color_map_channel(self, data, channel):
        buffer = np.zeros_like(data)
        for i in range(self.order + 1):
            freq = np.power(self.color[channel][i], i) * 2 * np.pi
            buffer += np.sin(freq * data)
            buffer += self.color[channel][i] * np.power(data, i)
        buffer = self._normalize(buffer)
        return buffer

    def _normalize_gaussian(self, data):
        mean = np.mean(data)
        std = np.std(data)
        data = (data - mean) / std
        return data

    def _normalize(self, data):
        _min = np.min(data)
        _max = np.max(data)
        return (data - _min)/(_max - _min)

    def _cvtcolor(self, data):
        data = 255 * data
        data = data.astype(np.uint8)
        return data

if __name__ == "__main__":
    src_filepath = "../../images/1621645137_mandelbrot.npy"
    dst_filepath = "../../images/1621643498_mandelbrot"

    for i in range(1, 5):
        for j in range(5):
            pc = PolyColor(i)
            pc.apply_from_npy(src_filepath, f"{dst_filepath}_{i:02d}_{j:02d}.png")



