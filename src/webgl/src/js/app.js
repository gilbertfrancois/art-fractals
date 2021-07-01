import * as THREE from "three";
import { GUI } from "./lib/dat.gui.module";
import Stats from "./lib/stats.module";

import fragment_shader_a from "./shader/fragment_a.glsl";
import fragment_shader_b from "./shader/fragment_b.glsl";
import vertex_shader from "./shader/vertex.glsl";

class ShaderCinema {
    constructor(
        vertex_shader,
        fragment_shader_a,
        fragment_shader_b,
        framebuffer_enable = true,
        framebuffer_size = 256,
        framebuffer_antialias = true,
    ) {
        // Application settings, stored in a dictionary, so that they can be modified by the DAT gui.
        this.settings = {
            framebuffer: {
                enable: framebuffer_enable,
                size: framebuffer_size,
                antialias: framebuffer_antialias,
            },
            stats: { enable: false },
            gui: { enable: false },
            debug: { log: true },
            mandelbrot: {
                depth: 128,
                zoom_factor: 1.4142,
                invert: true,
            },
        };
        this.stats = null;
        // this.render_size = new THREE.Vector2(512, 512);
        this.fragment_shader_a = fragment_shader_a;
        this.fragment_shader_b = fragment_shader_b;
        this.vertex_shader = vertex_shader;
        // Normalized coordinates, specified by its center and square minimum size (width or height).
        // The min and max values are computed and dependent on the screen and its aspect ratio.
        this.viewport = {
            size: 2.0,
            center: new THREE.Vector2(0.0, 0.0),
            min: new THREE.Vector2(-1.0, -1.0),
            max: new THREE.Vector2(1.0, 1.0),
            ratio: 1.0,
            width: 2.0,
            height: 2.0,
        };
        this.mouse = new THREE.Vector2(0.5, 0.5);
        this.mouse_start = new THREE.Vector2(0.0, 0.0);
        this.mouse_down = false;
        this.mouse_zoom_speed = 0.0;
        this.mouse_move_speed = new THREE.Vector2(0.0, 0.0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer();
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container = document.getElementById("container");
        this.container.appendChild(this.renderer.domElement);

        // Uniforms
        this.uniforms_a = {
            iGlobalTime: { type: "f", value: 0.0 },
            iResolution: { value: new THREE.Vector2() },
            iRatio: { type: "f", value: this.viewport.ratio },
            iMouse: { type: "v2", value: this.mouse },
            iViewPortMin: { type: "v2", value: this.viewport.min },
            iViewPortMax: { type: "v2", value: this.viewport.max },
            max_it: { type: "i", value: this.settings.mandelbrot.depth },
            invert: { type: "b", value: this.settings.mandelbrot.invert },
        };
        this.uniforms_b = {
            iChannel0: { type: "t", value: null },
        };

        // Render Target A
        this.scene_a = new THREE.Scene();
        this.camera_a = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        const geometry_a = new THREE.PlaneBufferGeometry(2, 2);
        const material_a = new THREE.ShaderMaterial({
            uniforms: this.uniforms_a,
            vertexShader: this.vertex_shader,
            fragmentShader: this.fragment_shader_a,
        });
        const mesh_a = new THREE.Mesh(geometry_a, material_a);
        this.scene_a.add(mesh_a);

        // Render Target B
        this._setup_render_target_a();
        this.scene_b = new THREE.Scene();
        this.camera_b = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        const geometry_b = new THREE.PlaneBufferGeometry(2, 2);
        const material_b = new THREE.ShaderMaterial({
            uniforms: this.uniforms_b,
            vertexShader: this.vertex_shader,
            fragmentShader: this.fragment_shader_b,
        });
        const mesh_b = new THREE.Mesh(geometry_b, material_b);
        this.scene_b.add(mesh_b);

        // Clock
        this.clock = new THREE.Clock();

        // Stats
        this.stats = new Stats();
        if (this.settings.stats.enable) {
            this._toggle_stats();
        }

        this._setup_listeners();
        if (this.settings.gui.enable) {
            this._setup_dat_gui();
        }

        this._onWindowResize();
    }

    _map(src, src_min, src_max, dst_min, dst_max) {
        return (
            ((src - src_min) / (src_max - src_min)) * (dst_max - dst_min) +
            dst_min
        );
    }

    run() {
        this._update_uniforms();
        if (this.settings.stats.enable) {
            this.stats.update();
        }
        if (this.settings.framebuffer.enable) {
            // render target a
            this.renderer.setRenderTarget(this.render_target_a);
            this.renderer.render(this.scene_a, this.camera_a);
            this.renderer.setRenderTarget(null);
            this.uniforms_b.iChannel0.value = this.render_target_a.texture;
            // render target b
            this.renderer.render(this.scene_b, this.camera_b);
        } else {
            this.renderer.render(this.scene_a, this.camera_a);
        }
        requestAnimationFrame(this.run.bind(this));
    }

    _onWindowResize() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        if (this.settings.framebuffer.enable) {
            this._setup_render_target_a();
        }
        this._update_viewport();
        this._update_uniforms();
    }

    _set_viewport_center(x, y, relative) {
        if (relative) {
            this.viewport.center.x -= x * 1.0 * this.viewport.width;
            this.viewport.center.y += y * 1.0 * this.viewport.height;
        } else {
            let cx = this._map(
                x,
                0,
                1,
                this.viewport.min.x,
                this.viewport.max.x,
            );
            let cy = this._map(
                y,
                0,
                1,
                this.viewport.max.y,
                this.viewport.min.y,
            );
            this.viewport.center.x = cx;
            this.viewport.center.y = cy;
        }
        this._update_viewport();
        this._update_uniforms();
    }

    _set_viewport_size(size, relative) {
        if (relative) {
            this.viewport.size += size;
        } else {
            this.viewport.size = size;
        }
        this._update_viewport();
    }

    _update_viewport() {
        this.viewport.ratio =
            this.renderer.domElement.width / this.renderer.domElement.height;
        let ratio_xy = new THREE.Vector2(1.0, 1.0);
        if (this.viewport.ratio > 1.0) {
            ratio_xy.x = this.viewport.ratio;
        } else {
            ratio_xy.y = 1.0 / this.viewport.ratio;
        }
        this.viewport.min.x =
            this.viewport.center.x - 0.5 * this.viewport.size * ratio_xy.x;
        this.viewport.min.y =
            this.viewport.center.y - 0.5 * this.viewport.size * ratio_xy.y;
        this.viewport.max.x =
            this.viewport.center.x + 0.5 * this.viewport.size * ratio_xy.x;
        this.viewport.max.y =
            this.viewport.center.y + 0.5 * this.viewport.size * ratio_xy.y;
        this.viewport.width = this.viewport.max.x - this.viewport.min.x;
        this.viewport.height = this.viewport.max.y - this.viewport.min.y;
        if (this.settings.debug.log) {
            console.log(this.viewport);
        }
    }

    _update_uniforms() {
        this.uniforms_a.iGlobalTime.value += this.clock.getDelta();
        if (this.settings.framebuffer.enable) {
            this.uniforms_a.iResolution.value.x = this.render_target_a.width;
            this.uniforms_a.iResolution.value.y = this.render_target_a.height;
        } else {
            this.uniforms_a.iResolution.value.x =
                this.renderer.domElement.width;
            this.uniforms_a.iResolution.value.y =
                this.renderer.domElement.height;
        }
        this.uniforms_a.iRatio = this.viewport.ratio;
        this.uniforms_a.iViewPortMin.value = this.viewport.min;
        this.uniforms_a.iViewPortMax.value = this.viewport.max;
        this.uniforms_a.iMouse.value = this.mouse;
        this.uniforms_a.max_it.value = this.settings.mandelbrot.depth;
        this.uniforms_a.invert.value = this.settings.mandelbrot.invert;

        if (this.settings.debug.log) {
            console.log(
                "render_target_b = (" +
                    this.renderer.domElement.width +
                    ", " +
                    this.renderer.domElement.height +
                    ")",
            );
        }
    }

    _setup_render_target_a() {
        let aspect_ratio =
            this.renderer.domElement.width / this.renderer.domElement.height;
        let width, height, params;
        if (aspect_ratio > 1.0) {
            width = this.settings.framebuffer.size;
            height = this.settings.framebuffer.size / aspect_ratio;
        } else {
            width = this.settings.framebuffer.size * aspect_ratio;
            height = this.settings.framebuffer.size;
        }
        width = Math.floor(width);
        height = Math.floor(height);
        if (this.settings.framebuffer.antialias) {
            params = {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
            };
        } else {
            params = {
                minFilter: THREE.NearestFilter,
                magFilter: THREE.NearestFilter,
            };
        }
        this.render_target_a = new THREE.WebGLRenderTarget(
            width,
            height,
            params,
        );
        if (this.settings.debug.log) {
            console.log(
                "render_target_a = (" +
                    this.render_target_a.width +
                    ", " +
                    this.render_target_a.height +
                    "), ratio = " +
                    aspect_ratio,
            );
        }
        this._update_viewport();
        this.uniforms_b.iChannel0.value = this.render_target_a.texture;
    }

    _setup_dat_gui() {
        this.gui = new GUI();
        const gui_settings = this.gui.addFolder("Settings");
        gui_settings
            .add(this.settings.framebuffer, "antialias", true)
            .onFinishChange(this._setup_render_target_a.bind(this));
        gui_settings
            .add(this.settings.stats, "enable", true)
            .name("stats")
            .onFinishChange(this._toggle_stats.bind(this));
        gui_settings.add(this.settings.debug, "log", true).name("debug output");
        gui_settings.close();
        const gui_viewport = this.gui.addFolder("Viewport");
        gui_viewport
            .add(this.viewport.center, "x", -10, 10, 0.01)
            .onFinishChange(this._update_viewport.bind(this));
        gui_viewport
            .add(this.viewport.center, "y", -10, 10, 0.01)
            .onFinishChange(this._update_viewport.bind(this));
        gui_viewport
            .add(this.viewport, "size", 0.0001, 2)
            .onFinishChange(this._update_viewport.bind(this));
        gui_viewport.open();
        const gui_mandelbrot = this.gui.addFolder("Mandelbrot");
        gui_mandelbrot.add(this.settings.mandelbrot, "depth", 1, 1000);
        gui_mandelbrot.add(this.settings.mandelbrot, "zoom_factor", 0, 2);
        gui_mandelbrot.add(this.settings.mandelbrot, "invert");
        gui_mandelbrot.open();
        this.gui.close();
    }

    _setup_listeners() {
        window.addEventListener("resize", this._onWindowResize.bind(this));

        this.container.addEventListener(
            "mousedown",
            function (event) {
                event.preventDefault();
                if (!this.mouse_down) {
                    this.mouse_start.x = event.clientX / window.innerWidth;
                    this.mouse_start.y = event.clientY / window.innerHeight;
                }
                this.mousedown = true;
            }.bind(this),
            false,
        );

        this.container.addEventListener(
            "mouseleave",
            function () {
                this.mousedown = false;
            }.bind(this),
            false,
        );

        this.container.addEventListener(
            "mouseup",
            function () {
                this.mousedown = false;
            }.bind(this),
            false,
        );

        this.container.addEventListener(
            "mousemove",
            function (event) {
                this.mouse.x = event.clientX / window.innerWidth;
                this.mouse.y = event.clientY / window.innerHeight;

                if (this.mousedown) {
                    event.preventDefault();
                    let dx = this.mouse.x - this.mouse_start.x;
                    let dy = this.mouse.y - this.mouse_start.y;
                    this._set_viewport_center(dx, dy, true);
                    this.mouse_start.x = this.mouse.x;
                    this.mouse_start.y = this.mouse.y;
                }
            }.bind(this),
            false,
        );

        this.container.addEventListener(
            "wheel",
            function (event) {
                event.preventDefault();
                // Update new center
                // let dx =
                //   (event.clientX - 0.5 * window.innerWidth) * event.deltaY * 0.0005;
                // let dy =
                //   (event.clientY - 0.5 * window.innerHeight) * event.deltaY * 0.0005;
                // this._set_viewport_center(dx, dy, true);
                // Update new viewport size
                let ds =
                    (event.deltaY / window.innerHeight) * this.viewport.size;
                let dmy =
                    (0.5 * window.innerHeight - event.clientY) /
                    window.innerHeight;
                let dmx =
                    (-0.5 * window.innerWidth + event.clientX) /
                    window.innerWidth;
                let theta = Math.atan2(dmy, dmx);
                let dx = 0.3 * ds * Math.cos(theta);
                let dy = -0.3 * ds * Math.sin(theta);
                console.log({ ds: ds, dx: dmx, dy: dmy, theta: theta });
                // this._set_viewport_center(dx, dy, true);
                this._set_viewport_size(ds, true);
            }.bind(this),
            false,
        );
    }

    _toggle_stats() {
        if (this.settings.stats.enable) {
            this.container.appendChild(this.stats.dom);
            this.stats.showPanel(0);
        } else {
            this.container.removeChild(this.stats.dom);
        }
    }
}

let shader_cinema = new ShaderCinema(
    vertex_shader,
    fragment_shader_a,
    fragment_shader_b,
    false,
    256,
);
shader_cinema.run();
