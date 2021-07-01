precision highp float;

varying vec2  vUv;
uniform vec2  iMouse;
uniform vec2  iResolution;
uniform vec2  iViewPortMin;
uniform vec2  iViewPortMax;
uniform float iViewPortRatio;
uniform float iGlobalTime;
uniform sampler2D iChannel0;
uniform bool invert;

uniform int max_it;

vec2 add(vec2 dsa, vec2 dsb);
vec2 sub(vec2 dsa, vec2 dsb);
vec2 mul(vec2 dsa, vec2 dsb);

/**
 * Helper function: Maps the input vector with min-max range to a destination dimension space.
 *
 * @param src        Source vector.
 * @param src_min    Minimum value of the source.
 * @param src_max    Maximum value of the source.
 * @param dst_min    Minimun value of the destination.
 * @param dst_max    Maximum value of the destination.
 */
vec2 map(in vec2 src, in vec2 src_min, in vec2 src_max, in vec2 dst_min, in vec2 dst_max) {
    return (src - src_min) / (src_max - src_min) * (dst_max - dst_min) + dst_min;
}

/**
 * Helper function: Maps the input vector with range 0 ≤ src ≤ 1 to a destination dimension space.
 *
 * @param src        Source vector.
 * @param dst_min    Minimun value of the destination.
 * @param dst_max    Maximum value of the destination.
 */
vec2 mapn(in vec2 src, in vec2 dst_min, in vec2 dst_max) {
    return src * (dst_max - dst_min) + dst_min;
}

/**
 * Test if a point is inside or outside the Mandelbrot space.
 *
 * @param c   Input value (Re, Im)
 5* @returns   Value between 0 and 1, denoting the time it takes to escape the Mandelbrot space. All values == 1 are 
 *            considered outside.
 */
float f(in vec2 c) {
    vec2 z = vec2(0.0);
    vec2 zn = vec2(0.0);
    const int cmax_it = 128;
    for (int i = 0; i < cmax_it; i++) {
        zn.x = z.x * z.x - z.y * z.y + c.x;
        zn.y = 2.0 * z.x * z.y + c.y;
        if ((zn.x * zn.x + zn.y * zn.y) > 4.0) {
            float den = 1.0 * float(cmax_it);
            float nom = 1.0 * float(i);
            return nom / den;
        }
        z = zn;
    }
    return 1.0;
}

/**
 * Main
 *
 */
void main() {
    vec2 uv = gl_FragCoord.xy / iResolution.xy;
    uv = mapn(uv, iViewPortMin, iViewPortMax);
    // show checkerboard
    // vec3 cc = show_grid(uv, 10.0);
    // show working mouse interaction
    // vec3 cm = vec3(0.5*(iMouse.x + iMouse.y));
    float depth = f(uv);
    if (invert) {
        depth = 1.0 - depth;
    }
    depth = floor(depth+0.5);
    vec3 col = vec3(depth);
	gl_FragColor = vec4(col, 1.0);
}
