# Fractals

Implementation of the Mandelbrot fractal in different languages.



## Python

Multi-core, high precision implementation with float64 that allows deep zoom.

Installation:

```sh
cd [project folder]/src/py
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To run the program:

```sh
cd [project folder]/src/py
source .venv/bin/activate
python mandelbrot.py
```

The resulting image will be written in the folder `[project folder]/images` . To change the settings, like viewpoint center, viewpoint size, image output size, etc, you can edit the `src/py/config.ini` file.



## GLSL

By far the fastest implementation, written in GLSL shader language. On average, the computation of a HD image takes about < 17 milliseconds, meaning that it renders at 60 fps. However, since it uses float16, the image looses detail at a zoom level of 1000 and beyond. A running example can be found at 

[blitzblit.com]: https://www.blitzblit.com



To install and run, do:

```sh
cd src/webgl
npm install
npm run prod
```

Then open a browser and go to `http://localhost:1234`.



