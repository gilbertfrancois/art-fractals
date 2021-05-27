uniform sampler2D iChannel0;
varying vec2      vUv;

void main() {
	gl_FragColor = vec4(texture2D(iChannel0, vUv).rgb, 1.0);
    /* gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); */
}

