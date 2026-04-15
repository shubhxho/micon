// Three.js Volumetric Ray-Marching Brain Renderer
// Manual orbit controls, colormap LUTs, window/level sync

import * as THREE from "three";
import TWEEN from "@tweenjs/tween.js";

// ── Shaders ──────────────────────────────────────────────────────────────────

const vertexShader = /* glsl */ `
varying vec3 vOrigin;
varying vec3 vDirection;

void main() {
  vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
  vOrigin = (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
  vDirection = position - vOrigin;
  gl_Position = projectionMatrix * mvPos;
}
`;

const fragmentShader = /* glsl */ `
precision highp float;
precision highp sampler3D;

uniform sampler3D uVolume;
uniform sampler2D uColormap;
uniform float uThreshold;
uniform float uWindowCenter;
uniform float uWindowWidth;
uniform float uOpacity;
uniform int uSteps;

varying vec3 vOrigin;
varying vec3 vDirection;

void main() {
  vec3 dir = normalize(vDirection);

  // Ray-AABB intersection for unit cube centered at origin
  vec3 tMin = (-0.5 - vOrigin) / dir;
  vec3 tMax = ( 0.5 - vOrigin) / dir;
  vec3 t1 = min(tMin, tMax);
  vec3 t2 = max(tMin, tMax);
  float tNear = max(max(t1.x, t1.y), t1.z);
  float tFar  = min(min(t2.x, t2.y), t2.z);

  if (tNear > tFar) discard;
  tNear = max(tNear, 0.0);

  float stepSize = (tFar - tNear) / float(uSteps);
  vec4 color = vec4(0.0);

  for (int i = 0; i < 512; i++) {
    if (i >= uSteps) break;

    vec3 pos = vOrigin + dir * (tNear + float(i) * stepSize);
    float raw = texture(uVolume, pos + 0.5).r;

    // Apply window/level
    float wlLow = uWindowCenter - uWindowWidth * 0.5;
    float t = clamp((raw - wlLow) / uWindowWidth, 0.0, 1.0);

    if (t > uThreshold) {
      vec4 sampleColor = texture2D(uColormap, vec2(t, 0.5));
      sampleColor.a *= uOpacity;

      // Front-to-back compositing
      color.rgb += (1.0 - color.a) * sampleColor.a * sampleColor.rgb;
      color.a += (1.0 - color.a) * sampleColor.a;
    }

    if (color.a > 0.95) break;
  }

  if (color.a < 0.01) discard;
  gl_FragColor = color;
}
`;

// ── Colormap generation ──────────────────────────────────────────────────────

type ColormapName = "hot" | "plasma" | "viridis" | "gray" | "pet";

function generateColormap(name: ColormapName): THREE.DataTexture {
  const size = 256;
  const data = new Uint8Array(size * 4);

  for (let i = 0; i < size; i++) {
    const t = i / (size - 1);
    let r: number, g: number, b: number;

    switch (name) {
      case "hot":
        r = Math.min(1, t * 3) * 255;
        g = Math.max(0, Math.min(1, (t - 0.333) * 3)) * 255;
        b = Math.max(0, Math.min(1, (t - 0.666) * 3)) * 255;
        break;
      case "plasma":
        r = plasmaR(t) * 255;
        g = plasmaG(t) * 255;
        b = plasmaB(t) * 255;
        break;
      case "viridis":
        r = viridisR(t) * 255;
        g = viridisG(t) * 255;
        b = viridisB(t) * 255;
        break;
      case "gray":
        r = g = b = t * 255;
        break;
      case "pet":
        r = petR(t) * 255;
        g = petG(t) * 255;
        b = petB(t) * 255;
        break;
      default:
        r = g = b = t * 255;
    }

    data[i * 4 + 0] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = 255;
  }

  const tex = new THREE.DataTexture(data, size, 1, THREE.RGBAFormat);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

// Approximated colormap functions
function plasmaR(t: number) {
  return Math.min(1, Math.max(0, 0.05 + 2.8 * t - 3.5 * t * t + 1.7 * t * t * t));
}
function plasmaG(t: number) {
  return Math.min(1, Math.max(0, -0.2 + 1.5 * t * t - 0.4 * t * t * t));
}
function plasmaB(t: number) {
  return Math.min(1, Math.max(0, 0.53 + 1.5 * t - 4.5 * t * t + 3.0 * t * t * t));
}

function viridisR(t: number) {
  return Math.min(1, Math.max(0, 0.27 - 0.25 * t + 1.0 * t * t));
}
function viridisG(t: number) {
  return Math.min(1, Math.max(0, 0.004 + 1.4 * t - 0.55 * t * t));
}
function viridisB(t: number) {
  return Math.min(1, Math.max(0, 0.33 + 0.75 * t - 1.9 * t * t + 0.95 * t * t * t));
}

function petR(t: number) {
  if (t < 0.25) return 0;
  if (t < 0.5) return (t - 0.25) * 4;
  return 1;
}
function petG(t: number) {
  if (t < 0.25) return t * 4;
  if (t < 0.75) return 1;
  return 1 - (t - 0.75) * 4;
}
function petB(t: number) {
  if (t < 0.5) return 1 - t * 2;
  return 0;
}

// ── Brain3D class ────────────────────────────────────────────────────────────

export class Brain3D {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private volMesh: THREE.Mesh | null = null;
  private material: THREE.ShaderMaterial | null = null;
  private colormaps: Map<ColormapName, THREE.DataTexture> = new Map();
  private currentColormap: ColormapName = "hot";
  private container: HTMLElement;

  // Orbit state
  private isDragging = false;
  private prevMouse = { x: 0, y: 0 };
  private spherical = { theta: 0, phi: Math.PI / 4, radius: 2.5 };
  private target = new THREE.Vector3(0, 0, 0);

  // Slice planes
  private slicePlanes: {
    x: THREE.Mesh | null;
    y: THREE.Mesh | null;
    z: THREE.Mesh | null;
  } = { x: null, y: null, z: null };

  private animationId: number = 0;

  constructor(container: HTMLElement) {
    this.container = container;

    // Scene
    this.scene = new THREE.Scene();

    // Camera
    this.camera = new THREE.PerspectiveCamera(
      50,
      container.clientWidth / container.clientHeight,
      0.01,
      100
    );
    this.updateCameraPosition();

    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0a0a0f, 1);
    container.appendChild(this.renderer.domElement);

    // Generate colormaps
    for (const name of ["hot", "plasma", "viridis", "gray", "pet"] as ColormapName[]) {
      this.colormaps.set(name, generateColormap(name));
    }

    // Ambient grid
    this.addGrid();

    // Axis indicator
    this.addAxes();

    // Controls
    this.setupControls();

    // Resize
    const ro = new ResizeObserver(() => this.onResize());
    ro.observe(container);

    // Animate
    this.animate();
  }

  private addGrid() {
    const grid = new THREE.GridHelper(4, 20, 0x1a1a2e, 0x111122);
    grid.position.y = -0.8;
    this.scene.add(grid);
  }

  private addAxes() {
    const len = 0.6;
    const axes = new THREE.Group();

    const makeAxis = (dir: THREE.Vector3, color: number) => {
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        dir.multiplyScalar(len),
      ]);
      const mat = new THREE.LineBasicMaterial({ color, linewidth: 2 });
      return new THREE.Line(geo, mat);
    };

    axes.add(makeAxis(new THREE.Vector3(1, 0, 0), 0xff4444)); // X - red
    axes.add(makeAxis(new THREE.Vector3(0, 1, 0), 0x44ff44)); // Y - green
    axes.add(makeAxis(new THREE.Vector3(0, 0, 1), 0x4488ff)); // Z - blue
    axes.position.set(-0.8, -0.8, -0.8);
    this.scene.add(axes);
  }

  private setupControls() {
    const el = this.renderer.domElement;

    // Mouse orbit
    el.addEventListener("mousedown", (e) => {
      this.isDragging = true;
      this.prevMouse = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener("mousemove", (e) => {
      if (!this.isDragging) return;
      const dx = e.clientX - this.prevMouse.x;
      const dy = e.clientY - this.prevMouse.y;
      this.spherical.theta -= dx * 0.005;
      this.spherical.phi = Math.max(
        0.1,
        Math.min(Math.PI - 0.1, this.spherical.phi - dy * 0.005)
      );
      this.prevMouse = { x: e.clientX, y: e.clientY };
      this.updateCameraPosition();
    });

    window.addEventListener("mouseup", () => {
      this.isDragging = false;
    });

    // Zoom
    el.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.spherical.radius = Math.max(
          0.5,
          Math.min(10, this.spherical.radius + e.deltaY * 0.002)
        );
        this.updateCameraPosition();
      },
      { passive: false }
    );

    // Touch support
    let lastTouchDist = 0;
    let lastTouch = { x: 0, y: 0 };

    el.addEventListener("touchstart", (e) => {
      e.preventDefault();
      if (e.touches.length === 1) {
        this.isDragging = true;
        lastTouch = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      } else if (e.touches.length === 2) {
        const dx = e.touches[1].clientX - e.touches[0].clientX;
        const dy = e.touches[1].clientY - e.touches[0].clientY;
        lastTouchDist = Math.sqrt(dx * dx + dy * dy);
      }
    }, { passive: false });

    el.addEventListener("touchmove", (e) => {
      e.preventDefault();
      if (e.touches.length === 1 && this.isDragging) {
        const dx = e.touches[0].clientX - lastTouch.x;
        const dy = e.touches[0].clientY - lastTouch.y;
        this.spherical.theta -= dx * 0.005;
        this.spherical.phi = Math.max(
          0.1,
          Math.min(Math.PI - 0.1, this.spherical.phi - dy * 0.005)
        );
        lastTouch = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        this.updateCameraPosition();
      } else if (e.touches.length === 2) {
        const dx = e.touches[1].clientX - e.touches[0].clientX;
        const dy = e.touches[1].clientY - e.touches[0].clientY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const delta = lastTouchDist - dist;
        this.spherical.radius = Math.max(
          0.5,
          Math.min(10, this.spherical.radius + delta * 0.005)
        );
        lastTouchDist = dist;
        this.updateCameraPosition();
      }
    }, { passive: false });

    el.addEventListener("touchend", () => {
      this.isDragging = false;
    });
  }

  private updateCameraPosition() {
    const { theta, phi, radius } = this.spherical;
    this.camera.position.set(
      radius * Math.sin(phi) * Math.sin(theta) + this.target.x,
      radius * Math.cos(phi) + this.target.y,
      radius * Math.sin(phi) * Math.cos(theta) + this.target.z
    );
    this.camera.lookAt(this.target);
  }

  private onResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  private animate() {
    this.animationId = requestAnimationFrame(() => this.animate());
    TWEEN.update();
    this.renderer.render(this.scene, this.camera);
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  loadVolume(data: Float32Array, sizeX: number, sizeY: number, sizeZ: number) {
    // Remove existing volume
    if (this.volMesh) {
      this.scene.remove(this.volMesh);
      this.volMesh.geometry.dispose();
      (this.volMesh.material as THREE.Material).dispose();
    }

    // Create 3D texture
    const tex3d = new THREE.Data3DTexture(data, sizeX, sizeY, sizeZ);
    tex3d.format = THREE.RedFormat;
    tex3d.type = THREE.FloatType;
    tex3d.minFilter = THREE.LinearFilter;
    tex3d.magFilter = THREE.LinearFilter;
    tex3d.unpackAlignment = 1;
    tex3d.needsUpdate = true;

    // Shader material
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uVolume: { value: tex3d },
        uColormap: { value: this.colormaps.get(this.currentColormap)! },
        uThreshold: { value: 0.1 },
        uWindowCenter: { value: 0.5 },
        uWindowWidth: { value: 1.0 },
        uOpacity: { value: 0.03 },
        uSteps: { value: 256 },
      },
      vertexShader,
      fragmentShader,
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false,
    });

    // Scale box to match voxel aspect ratio
    const maxDim = Math.max(sizeX, sizeY, sizeZ);
    const scaleX = sizeX / maxDim;
    const scaleY = sizeY / maxDim;
    const scaleZ = sizeZ / maxDim;

    const geometry = new THREE.BoxGeometry(1, 1, 1);
    this.volMesh = new THREE.Mesh(geometry, this.material);
    this.volMesh.scale.set(scaleX, scaleY, scaleZ);
    this.scene.add(this.volMesh);

    // Animate camera to a nice position
    this.animateCamera(0.5, Math.PI / 4, 2.2);

    // Dispatch loaded event
    window.dispatchEvent(new CustomEvent("volume-loaded", { detail: { sizeX, sizeY, sizeZ } }));
  }

  animateCamera(theta: number, phi: number, radius: number) {
    const from = { ...this.spherical };
    new TWEEN.Tween(from)
      .to({ theta, phi, radius }, 1200)
      .easing(TWEEN.Easing.Cubic.InOut)
      .onUpdate(() => {
        this.spherical.theta = from.theta;
        this.spherical.phi = from.phi;
        this.spherical.radius = from.radius;
        this.updateCameraPosition();
      })
      .start();
  }

  setThreshold(value: number) {
    if (this.material) {
      this.material.uniforms.uThreshold.value = value;
    }
  }

  setOpacity(value: number) {
    if (this.material) {
      this.material.uniforms.uOpacity.value = value;
    }
  }

  setSteps(value: number) {
    if (this.material) {
      this.material.uniforms.uSteps.value = value;
    }
  }

  updateWindowLevel(center: number, width: number) {
    if (this.material) {
      this.material.uniforms.uWindowCenter.value = center;
      this.material.uniforms.uWindowWidth.value = width;
    }
  }

  setColormap(name: string) {
    const cm = name as ColormapName;
    if (this.material && this.colormaps.has(cm)) {
      this.currentColormap = cm;
      this.material.uniforms.uColormap.value = this.colormaps.get(cm)!;
    }
  }

  setSliceX(t: number) {
    // Normalized 0-1, can be used for clipping or highlight
    if (this.material) {
      // We can add clipping plane logic here if needed
    }
  }

  setSliceY(t: number) {}
  setSliceZ(t: number) {}

  destroy() {
    cancelAnimationFrame(this.animationId);
    this.renderer.dispose();
  }
}
