// Three.js Enhanced Volumetric Brain Renderer
// Features: gradient lighting, bloom glow, neural particles, auto-rotation

import * as THREE from "three";
import TWEEN from "@tweenjs/tween.js";

// ── Volume Shaders ──────────────────────────────────────────────────────────

const volumeVertexShader = /* glsl */ `
varying vec3 vOrigin;
varying vec3 vDirection;

void main() {
  vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
  vOrigin = (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
  vDirection = position - vOrigin;
  gl_Position = projectionMatrix * mvPos;
}
`;

const volumeFragmentShader = /* glsl */ `
precision highp float;
precision highp sampler3D;

uniform sampler3D uVolume;
uniform sampler2D uColormap;
uniform float uThreshold;
uniform float uWindowCenter;
uniform float uWindowWidth;
uniform float uOpacity;
uniform int uSteps;
uniform vec3 uLightDir;
uniform float uLightIntensity;
uniform float uAmbientIntensity;
uniform float uTime;

varying vec3 vOrigin;
varying vec3 vDirection;

// Estimate surface normal from volume gradient (central differences)
vec3 estimateNormal(vec3 pos, float d) {
  float dx = texture(uVolume, pos + vec3(d, 0, 0) + 0.5).r
           - texture(uVolume, pos - vec3(d, 0, 0) + 0.5).r;
  float dy = texture(uVolume, pos + vec3(0, d, 0) + 0.5).r
           - texture(uVolume, pos - vec3(0, d, 0) + 0.5).r;
  float dz = texture(uVolume, pos + vec3(0, 0, d) + 0.5).r
           - texture(uVolume, pos - vec3(0, 0, d) + 0.5).r;
  vec3 n = vec3(dx, dy, dz);
  float len = length(n);
  return len > 0.0001 ? n / len : vec3(0.0, 1.0, 0.0);
}

// Blinn-Phong lighting
vec3 blinnPhong(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 color) {
  float ambient = uAmbientIntensity;
  float diff = max(dot(normal, lightDir), 0.0);
  vec3 halfDir = normalize(lightDir + viewDir);
  float spec = pow(max(dot(normal, halfDir), 0.0), 40.0);

  // Subtle subsurface scattering approximation
  float sss = max(0.0, dot(viewDir, -lightDir)) * 0.15;

  return color * (ambient + diff * uLightIntensity * 0.65 + sss)
       + vec3(0.7, 0.85, 1.0) * spec * uLightIntensity * 0.5;
}

void main() {
  vec3 dir = normalize(vDirection);
  vec3 viewDir = -dir;
  vec3 lightDir = normalize(uLightDir);

  // Ray-AABB intersection for unit cube
  vec3 tMin = (-0.5 - vOrigin) / dir;
  vec3 tMax = ( 0.5 - vOrigin) / dir;
  vec3 t1 = min(tMin, tMax);
  vec3 t2 = max(tMin, tMax);
  float tNear = max(max(t1.x, t1.y), t1.z);
  float tFar  = min(min(t2.x, t2.y), t2.z);

  if (tNear > tFar) discard;
  tNear = max(tNear, 0.0);

  float stepSize = (tFar - tNear) / float(uSteps);
  float gradDelta = 1.0 / 128.0;
  vec4 color = vec4(0.0);

  // Jitter ray start to reduce banding artifacts
  float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  float t = tNear + jitter * stepSize;

  for (int i = 0; i < 512; i++) {
    if (i >= uSteps) break;
    if (t > tFar) break;

    vec3 pos = vOrigin + dir * t;
    float raw = texture(uVolume, pos + 0.5).r;

    // Window/level transform
    float wlLow = uWindowCenter - uWindowWidth * 0.5;
    float v = clamp((raw - wlLow) / uWindowWidth, 0.0, 1.0);

    if (v > uThreshold) {
      vec4 sampleColor = texture2D(uColormap, vec2(v, 0.5));

      // Gradient-based normal for lighting
      vec3 normal = estimateNormal(pos, gradDelta);
      vec3 litColor = blinnPhong(normal, viewDir, lightDir, sampleColor.rgb);

      // Gradient magnitude for edge enhancement
      float gx = texture(uVolume, pos + vec3(gradDelta, 0, 0) + 0.5).r
               - texture(uVolume, pos - vec3(gradDelta, 0, 0) + 0.5).r;
      float gy = texture(uVolume, pos + vec3(0, gradDelta, 0) + 0.5).r
               - texture(uVolume, pos - vec3(0, gradDelta, 0) + 0.5).r;
      float gz = texture(uVolume, pos + vec3(0, 0, gradDelta) + 0.5).r
               - texture(uVolume, pos - vec3(0, 0, gradDelta) + 0.5).r;
      float gradMag = length(vec3(gx, gy, gz));

      // Surface-aware opacity — emphasize edges/boundaries
      float edgeFactor = 1.0 + gradMag * 4.0;
      float alpha = sampleColor.a * uOpacity * edgeFactor;

      // Depth-based atmospheric attenuation
      float depth = (t - tNear) / (tFar - tNear);
      float attenuation = 1.0 - depth * 0.2;
      litColor *= attenuation;

      // Front-to-back compositing
      color.rgb += (1.0 - color.a) * alpha * litColor;
      color.a += (1.0 - color.a) * alpha;
    }

    if (color.a > 0.95) break;
    t += stepSize;
  }

  if (color.a < 0.01) discard;

  // Rim glow — subtle edge highlight
  float rimDot = abs(dot(viewDir, vec3(0.0, 1.0, 0.0)));
  float rim = pow(1.0 - rimDot, 3.0);
  color.rgb += vec3(0.05, 0.15, 0.3) * rim * color.a * 0.3;

  gl_FragColor = color;
}
`;

// ── Particle Shaders ────────────────────────────────────────────────────────

const particleVertexShader = /* glsl */ `
uniform float uTime;
uniform float uPointSize;

attribute float aPhase;
attribute float aSpeed;
attribute vec3 aVelocity;

varying float vAlpha;
varying vec3 vColor;

void main() {
  // Animate position along velocity with sinusoidal oscillation
  vec3 pos = position + aVelocity * sin(uTime * aSpeed + aPhase) * 0.15;

  // Pulse alpha based on phase
  float pulse = 0.3 + 0.7 * pow(sin(uTime * aSpeed * 0.5 + aPhase) * 0.5 + 0.5, 2.0);
  vAlpha = pulse;

  // Color based on position (map to cyan-blue-purple spectrum)
  float hue = (pos.y + 0.5) * 0.6 + 0.5;
  vColor = vec3(
    0.2 + 0.3 * sin(hue * 6.28),
    0.4 + 0.4 * sin(hue * 6.28 + 2.09),
    0.7 + 0.3 * sin(hue * 6.28 + 4.19)
  );

  vec4 mvPos = modelViewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mvPos;

  // Size attenuation
  gl_PointSize = uPointSize * (200.0 / -mvPos.z);
}
`;

const particleFragmentShader = /* glsl */ `
varying float vAlpha;
varying vec3 vColor;

void main() {
  // Soft circle with glow
  vec2 center = gl_PointCoord - 0.5;
  float dist = length(center);
  if (dist > 0.5) discard;

  float glow = exp(-dist * dist * 8.0);
  float core = smoothstep(0.15, 0.0, dist);

  vec3 color = vColor * glow + vec3(1.0) * core * 0.3;
  float alpha = glow * vAlpha * 0.6;

  gl_FragColor = vec4(color, alpha);
}
`;

// ── Synapse Shaders (neural pathway connections) ────────────────────────────

const synapseVertexShader = /* glsl */ `
uniform float uTime;
attribute float aT;
attribute float aPhase;
varying float vAlpha;
varying float vT;

void main() {
  vT = aT;
  // Traveling pulse along the line
  float pulse = fract(uTime * 0.3 + aPhase);
  float dist = abs(aT - pulse);
  dist = min(dist, 1.0 - dist); // wrap-around distance
  vAlpha = exp(-dist * dist * 80.0) * 0.8;

  vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPos;
}
`;

const synapseFragmentShader = /* glsl */ `
varying float vAlpha;
varying float vT;

void main() {
  // Cyan-white synapse color
  vec3 color = mix(vec3(0.1, 0.5, 0.8), vec3(0.5, 0.9, 1.0), vAlpha);
  gl_FragColor = vec4(color, vAlpha * 0.5 + 0.03);
}
`;

// ── Bloom / Post-Processing Shaders ─────────────────────────────────────────

const fullscreenVertexShader = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position, 1.0);
}
`;

const brightPassFragmentShader = /* glsl */ `
uniform sampler2D uTexture;
uniform float uBrightThreshold;
varying vec2 vUv;

void main() {
  vec4 color = texture2D(uTexture, vUv);
  float brightness = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
  float contribution = max(0.0, brightness - uBrightThreshold);
  gl_FragColor = vec4(color.rgb * contribution, 1.0);
}
`;

const blurFragmentShader = /* glsl */ `
uniform sampler2D uTexture;
uniform vec2 uDirection;
uniform vec2 uResolution;
varying vec2 vUv;

void main() {
  vec2 texelSize = 1.0 / uResolution;
  vec4 result = vec4(0.0);

  // 9-tap Gaussian blur
  float weights[5];
  weights[0] = 0.227027;
  weights[1] = 0.194596;
  weights[2] = 0.121622;
  weights[3] = 0.054054;
  weights[4] = 0.016216;

  result += texture2D(uTexture, vUv) * weights[0];
  for (int i = 1; i < 5; i++) {
    vec2 offset = uDirection * texelSize * float(i) * 2.0;
    result += texture2D(uTexture, vUv + offset) * weights[i];
    result += texture2D(uTexture, vUv - offset) * weights[i];
  }

  gl_FragColor = result;
}
`;

const compositeFragmentShader = /* glsl */ `
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uBloomIntensity;
varying vec2 vUv;

void main() {
  vec4 scene = texture2D(uScene, vUv);
  vec4 bloom = texture2D(uBloom, vUv);

  // Additive blend with tone mapping
  vec3 result = scene.rgb + bloom.rgb * uBloomIntensity;

  // Simple Reinhard tone mapping
  result = result / (result + vec3(1.0));

  // Subtle vignette
  vec2 uv = vUv * 2.0 - 1.0;
  float vignette = 1.0 - dot(uv, uv) * 0.15;
  result *= vignette;

  gl_FragColor = vec4(result, max(scene.a, bloom.a * uBloomIntensity));
}
`;

// ── Colormap generation ──────────────────────────────────────────────────────

type ColormapName = "hot" | "plasma" | "viridis" | "gray" | "pet" | "coolwarm" | "inferno";

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
      case "coolwarm":
        r = coolwarmR(t) * 255;
        g = coolwarmG(t) * 255;
        b = coolwarmB(t) * 255;
        break;
      case "inferno":
        r = infernoR(t) * 255;
        g = infernoG(t) * 255;
        b = infernoB(t) * 255;
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

// Colormap interpolation functions
function plasmaR(t: number) { return Math.min(1, Math.max(0, 0.05 + 2.8 * t - 3.5 * t * t + 1.7 * t * t * t)); }
function plasmaG(t: number) { return Math.min(1, Math.max(0, -0.2 + 1.5 * t * t - 0.4 * t * t * t)); }
function plasmaB(t: number) { return Math.min(1, Math.max(0, 0.53 + 1.5 * t - 4.5 * t * t + 3.0 * t * t * t)); }

function viridisR(t: number) { return Math.min(1, Math.max(0, 0.27 - 0.25 * t + 1.0 * t * t)); }
function viridisG(t: number) { return Math.min(1, Math.max(0, 0.004 + 1.4 * t - 0.55 * t * t)); }
function viridisB(t: number) { return Math.min(1, Math.max(0, 0.33 + 0.75 * t - 1.9 * t * t + 0.95 * t * t * t)); }

function petR(t: number) { return t < 0.25 ? 0 : t < 0.5 ? (t - 0.25) * 4 : 1; }
function petG(t: number) { return t < 0.25 ? t * 4 : t < 0.75 ? 1 : 1 - (t - 0.75) * 4; }
function petB(t: number) { return t < 0.5 ? 1 - t * 2 : 0; }

function coolwarmR(t: number) { return Math.min(1, Math.max(0, 0.23 + 2.2 * t - 1.5 * t * t)); }
function coolwarmG(t: number) { return Math.min(1, Math.max(0, 0.28 + 1.8 * t - 2.8 * t * t + 1.2 * t * t * t)); }
function coolwarmB(t: number) { return Math.min(1, Math.max(0, 0.75 - 0.8 * t + 0.3 * t * t)); }

function infernoR(t: number) { return Math.min(1, Math.max(0, -0.02 + 3.5 * t - 4.0 * t * t + 1.8 * t * t * t)); }
function infernoG(t: number) { return Math.min(1, Math.max(0, -0.1 + 0.6 * t + 1.5 * t * t - 1.5 * t * t * t)); }
function infernoB(t: number) { return Math.min(1, Math.max(0, 0.0 + 3.0 * t - 8.0 * t * t + 7.0 * t * t * t - 1.8 * t * t * t * t)); }

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
  private clock = new THREE.Clock();

  // Orbit state
  private isDragging = false;
  private prevMouse = { x: 0, y: 0 };
  private spherical = { theta: 0, phi: Math.PI / 4, radius: 2.5 };
  private target = new THREE.Vector3(0, 0, 0);

  // Slice planes
  private slicePlanes: { x: THREE.Mesh | null; y: THREE.Mesh | null; z: THREE.Mesh | null } = { x: null, y: null, z: null };

  // Particles
  private particles: THREE.Points | null = null;
  private particleMaterial: THREE.ShaderMaterial | null = null;
  private particleCount = 2000;
  private particlesEnabled = true;

  // Synapses (neural pathway lines)
  private synapseGroup: THREE.Group | null = null;
  private synapseMaterial: THREE.ShaderMaterial | null = null;
  private synapsesEnabled = true;

  // Bloom post-processing
  private sceneTarget: THREE.WebGLRenderTarget;
  private brightTarget: THREE.WebGLRenderTarget;
  private blurTargetH: THREE.WebGLRenderTarget;
  private blurTargetV: THREE.WebGLRenderTarget;
  private fullscreenQuad: THREE.Mesh;
  private brightPassMat: THREE.ShaderMaterial;
  private blurHMat: THREE.ShaderMaterial;
  private blurVMat: THREE.ShaderMaterial;
  private compositeMat: THREE.ShaderMaterial;
  private postScene: THREE.Scene;
  private postCamera: THREE.OrthographicCamera;
  private bloomEnabled = true;
  private bloomIntensity = 0.6;

  // Auto-rotation
  private autoRotate = false;
  private autoRotateSpeed = 0.002;

  // Animation
  private animationId: number = 0;

  // FPS tracking
  private fpsFrames = 0;
  private fpsTime = 0;
  private currentFps = 0;

  constructor(container: HTMLElement) {
    this.container = container;
    const w = container.clientWidth;
    const h = container.clientHeight;

    // Scene
    this.scene = new THREE.Scene();

    // Camera
    this.camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 100);
    this.updateCameraPosition();

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x06060c, 1);
    this.renderer.autoClear = false;
    container.appendChild(this.renderer.domElement);

    // Generate colormaps
    for (const name of ["hot", "plasma", "viridis", "gray", "pet", "coolwarm", "inferno"] as ColormapName[]) {
      this.colormaps.set(name, generateColormap(name));
    }

    // Post-processing render targets
    const rtOpts: THREE.WebGLRenderTargetOptions = {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType,
    };
    this.sceneTarget = new THREE.WebGLRenderTarget(w, h, rtOpts);
    this.brightTarget = new THREE.WebGLRenderTarget(w / 2, h / 2, rtOpts);
    this.blurTargetH = new THREE.WebGLRenderTarget(w / 2, h / 2, rtOpts);
    this.blurTargetV = new THREE.WebGLRenderTarget(w / 2, h / 2, rtOpts);

    // Post-processing materials
    const fsGeo = new THREE.PlaneGeometry(2, 2);

    this.brightPassMat = new THREE.ShaderMaterial({
      uniforms: {
        uTexture: { value: null },
        uBrightThreshold: { value: 0.25 },
      },
      vertexShader: fullscreenVertexShader,
      fragmentShader: brightPassFragmentShader,
      depthTest: false,
    });

    this.blurHMat = new THREE.ShaderMaterial({
      uniforms: {
        uTexture: { value: null },
        uDirection: { value: new THREE.Vector2(1, 0) },
        uResolution: { value: new THREE.Vector2(w / 2, h / 2) },
      },
      vertexShader: fullscreenVertexShader,
      fragmentShader: blurFragmentShader,
      depthTest: false,
    });

    this.blurVMat = new THREE.ShaderMaterial({
      uniforms: {
        uTexture: { value: null },
        uDirection: { value: new THREE.Vector2(0, 1) },
        uResolution: { value: new THREE.Vector2(w / 2, h / 2) },
      },
      vertexShader: fullscreenVertexShader,
      fragmentShader: blurFragmentShader,
      depthTest: false,
    });

    this.compositeMat = new THREE.ShaderMaterial({
      uniforms: {
        uScene: { value: null },
        uBloom: { value: null },
        uBloomIntensity: { value: this.bloomIntensity },
      },
      vertexShader: fullscreenVertexShader,
      fragmentShader: compositeFragmentShader,
      depthTest: false,
    });

    // Post-processing scene
    this.postScene = new THREE.Scene();
    this.postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.fullscreenQuad = new THREE.Mesh(fsGeo, this.compositeMat);
    this.postScene.add(this.fullscreenQuad);

    // Scene elements
    this.addGrid();
    this.addAxes();
    this.addAmbientParticles();

    // Controls
    this.setupControls();

    // Resize
    const ro = new ResizeObserver(() => this.onResize());
    ro.observe(container);

    // Animate
    this.animate();
  }

  // ── Scene environment ───────────────────────────────────────────────────────

  private addGrid() {
    const grid = new THREE.GridHelper(6, 30, 0x151528, 0x0d0d1a);
    grid.position.y = -0.85;
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.5;
    this.scene.add(grid);

    // Subtle center ring
    const ringGeo = new THREE.RingGeometry(0.5, 0.52, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x22d3ee,
      transparent: true,
      opacity: 0.08,
      side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = -0.84;
    this.scene.add(ring);
  }

  private addAxes() {
    const len = 0.6;
    const axes = new THREE.Group();

    const makeAxis = (dir: THREE.Vector3, color: number) => {
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        dir.clone().multiplyScalar(len),
      ]);
      const mat = new THREE.LineBasicMaterial({ color, linewidth: 2, transparent: true, opacity: 0.6 });
      return new THREE.Line(geo, mat);
    };

    axes.add(makeAxis(new THREE.Vector3(1, 0, 0), 0xff4444));
    axes.add(makeAxis(new THREE.Vector3(0, 1, 0), 0x44ff44));
    axes.add(makeAxis(new THREE.Vector3(0, 0, 1), 0x4488ff));
    axes.position.set(-0.9, -0.85, -0.9);
    this.scene.add(axes);
  }

  private addAmbientParticles() {
    // Background dust/stars for atmosphere
    const count = 500;
    const positions = new Float32Array(count * 3);
    const sizes = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 8;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 6;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 8;
      sizes[i] = Math.random() * 2 + 0.5;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("size", new THREE.BufferAttribute(sizes, 1));

    const mat = new THREE.PointsMaterial({
      color: 0x334466,
      size: 0.015,
      transparent: true,
      opacity: 0.3,
      sizeAttenuation: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.scene.add(new THREE.Points(geo, mat));
  }

  // ── Neural particles ──────────────────────────────────────────────────────

  private createNeuralParticles() {
    if (this.particles) {
      this.scene.remove(this.particles);
      this.particles.geometry.dispose();
    }

    const count = this.particleCount;
    const positions = new Float32Array(count * 3);
    const phases = new Float32Array(count);
    const speeds = new Float32Array(count);
    const velocities = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      // Distribute on a spherical shell around the brain
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = 0.35 + Math.random() * 0.3;

      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.cos(phi);
      positions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);

      phases[i] = Math.random() * Math.PI * 2;
      speeds[i] = 0.5 + Math.random() * 2.0;

      // Random velocity direction
      velocities[i * 3] = (Math.random() - 0.5) * 2;
      velocities[i * 3 + 1] = (Math.random() - 0.5) * 2;
      velocities[i * 3 + 2] = (Math.random() - 0.5) * 2;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("aPhase", new THREE.BufferAttribute(phases, 1));
    geo.setAttribute("aSpeed", new THREE.BufferAttribute(speeds, 1));
    geo.setAttribute("aVelocity", new THREE.BufferAttribute(velocities, 3));

    this.particleMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uPointSize: { value: 4.0 },
      },
      vertexShader: particleVertexShader,
      fragmentShader: particleFragmentShader,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.particles = new THREE.Points(geo, this.particleMaterial);
    this.scene.add(this.particles);
  }

  // ── Neural synapse pathways ───────────────────────────────────────────────

  private createSynapses() {
    if (this.synapseGroup) {
      this.scene.remove(this.synapseGroup);
    }

    this.synapseGroup = new THREE.Group();
    const pathCount = 12;

    this.synapseMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
      },
      vertexShader: synapseVertexShader,
      fragmentShader: synapseFragmentShader,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    for (let p = 0; p < pathCount; p++) {
      const points = 64;
      const positions = new Float32Array(points * 3);
      const tValues = new Float32Array(points);
      const phaseValues = new Float32Array(points);

      // Random start and end points on brain surface
      const startTheta = Math.random() * Math.PI * 2;
      const startPhi = Math.acos(2 * Math.random() - 1);
      const endTheta = Math.random() * Math.PI * 2;
      const endPhi = Math.acos(2 * Math.random() - 1);
      const r = 0.35;

      const start = new THREE.Vector3(
        r * Math.sin(startPhi) * Math.cos(startTheta),
        r * Math.cos(startPhi),
        r * Math.sin(startPhi) * Math.sin(startTheta)
      );

      const end = new THREE.Vector3(
        r * Math.sin(endPhi) * Math.cos(endTheta),
        r * Math.cos(endPhi),
        r * Math.sin(endPhi) * Math.sin(endTheta)
      );

      // Create curved path (quadratic bezier through a raised midpoint)
      const mid = start.clone().add(end).multiplyScalar(0.5);
      mid.multiplyScalar(1.3 + Math.random() * 0.4);

      const phase = Math.random() * Math.PI * 2;

      for (let i = 0; i < points; i++) {
        const t = i / (points - 1);
        const it = 1 - t;

        // Quadratic bezier
        const x = it * it * start.x + 2 * it * t * mid.x + t * t * end.x;
        const y = it * it * start.y + 2 * it * t * mid.y + t * t * end.y;
        const z = it * it * start.z + 2 * it * t * mid.z + t * t * end.z;

        positions[i * 3] = x;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = z;
        tValues[i] = t;
        phaseValues[i] = phase;
      }

      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geo.setAttribute("aT", new THREE.BufferAttribute(tValues, 1));
      geo.setAttribute("aPhase", new THREE.BufferAttribute(phaseValues, 1));

      const line = new THREE.Line(geo, this.synapseMaterial);
      this.synapseGroup.add(line);
    }

    this.scene.add(this.synapseGroup);
  }

  // ── Controls ──────────────────────────────────────────────────────────────

  private setupControls() {
    const el = this.renderer.domElement;

    el.addEventListener("mousedown", (e) => {
      this.isDragging = true;
      this.prevMouse = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener("mousemove", (e) => {
      if (!this.isDragging) return;
      const dx = e.clientX - this.prevMouse.x;
      const dy = e.clientY - this.prevMouse.y;
      this.spherical.theta -= dx * 0.005;
      this.spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, this.spherical.phi - dy * 0.005));
      this.prevMouse = { x: e.clientX, y: e.clientY };
      this.updateCameraPosition();
    });

    window.addEventListener("mouseup", () => { this.isDragging = false; });

    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.spherical.radius = Math.max(0.5, Math.min(10, this.spherical.radius + e.deltaY * 0.002));
      this.updateCameraPosition();
    }, { passive: false });

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
        this.spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, this.spherical.phi - dy * 0.005));
        lastTouch = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        this.updateCameraPosition();
      } else if (e.touches.length === 2) {
        const dx = e.touches[1].clientX - e.touches[0].clientX;
        const dy = e.touches[1].clientY - e.touches[0].clientY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const delta = lastTouchDist - dist;
        this.spherical.radius = Math.max(0.5, Math.min(10, this.spherical.radius + delta * 0.005));
        lastTouchDist = dist;
        this.updateCameraPosition();
      }
    }, { passive: false });

    el.addEventListener("touchend", () => { this.isDragging = false; });
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
    if (w === 0 || h === 0) return;

    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);

    // Resize render targets
    this.sceneTarget.setSize(w, h);
    this.brightTarget.setSize(w / 2, h / 2);
    this.blurTargetH.setSize(w / 2, h / 2);
    this.blurTargetV.setSize(w / 2, h / 2);

    this.blurHMat.uniforms.uResolution.value.set(w / 2, h / 2);
    this.blurVMat.uniforms.uResolution.value.set(w / 2, h / 2);
  }

  // ── Bloom rendering pipeline ──────────────────────────────────────────────

  private renderBloom() {
    const renderer = this.renderer;

    // Step 1: Render scene to offscreen target
    renderer.setRenderTarget(this.sceneTarget);
    renderer.clear();
    renderer.render(this.scene, this.camera);

    // Step 2: Bright pass — extract bright areas
    this.fullscreenQuad.material = this.brightPassMat;
    this.brightPassMat.uniforms.uTexture.value = this.sceneTarget.texture;
    renderer.setRenderTarget(this.brightTarget);
    renderer.clear();
    renderer.render(this.postScene, this.postCamera);

    // Step 3: Horizontal blur
    this.fullscreenQuad.material = this.blurHMat;
    this.blurHMat.uniforms.uTexture.value = this.brightTarget.texture;
    renderer.setRenderTarget(this.blurTargetH);
    renderer.clear();
    renderer.render(this.postScene, this.postCamera);

    // Step 4: Vertical blur
    this.fullscreenQuad.material = this.blurVMat;
    this.blurVMat.uniforms.uTexture.value = this.blurTargetH.texture;
    renderer.setRenderTarget(this.blurTargetV);
    renderer.clear();
    renderer.render(this.postScene, this.postCamera);

    // Two more blur passes for wider bloom
    this.fullscreenQuad.material = this.blurHMat;
    this.blurHMat.uniforms.uTexture.value = this.blurTargetV.texture;
    renderer.setRenderTarget(this.blurTargetH);
    renderer.clear();
    renderer.render(this.postScene, this.postCamera);

    this.fullscreenQuad.material = this.blurVMat;
    this.blurVMat.uniforms.uTexture.value = this.blurTargetH.texture;
    renderer.setRenderTarget(this.blurTargetV);
    renderer.clear();
    renderer.render(this.postScene, this.postCamera);

    // Step 5: Composite scene + bloom to screen
    this.fullscreenQuad.material = this.compositeMat;
    this.compositeMat.uniforms.uScene.value = this.sceneTarget.texture;
    this.compositeMat.uniforms.uBloom.value = this.blurTargetV.texture;
    this.compositeMat.uniforms.uBloomIntensity.value = this.bloomIntensity;
    renderer.setRenderTarget(null);
    renderer.clear();
    renderer.render(this.postScene, this.postCamera);
  }

  // ── Animation loop ────────────────────────────────────────────────────────

  private animate() {
    this.animationId = requestAnimationFrame(() => this.animate());

    const dt = this.clock.getDelta();
    const elapsed = this.clock.getElapsedTime();

    // FPS tracking
    this.fpsFrames++;
    this.fpsTime += dt;
    if (this.fpsTime >= 1.0) {
      this.currentFps = Math.round(this.fpsFrames / this.fpsTime);
      this.fpsFrames = 0;
      this.fpsTime = 0;
    }

    // Auto-rotation
    if (this.autoRotate && !this.isDragging) {
      this.spherical.theta += this.autoRotateSpeed;
      this.updateCameraPosition();
    }

    // Update time uniforms
    if (this.material) {
      this.material.uniforms.uTime.value = elapsed;
    }
    if (this.particleMaterial && this.particlesEnabled) {
      this.particleMaterial.uniforms.uTime.value = elapsed;
    }
    if (this.synapseMaterial && this.synapsesEnabled) {
      this.synapseMaterial.uniforms.uTime.value = elapsed;
    }

    // Toggle visibility
    if (this.particles) this.particles.visible = this.particlesEnabled;
    if (this.synapseGroup) this.synapseGroup.visible = this.synapsesEnabled;

    TWEEN.update();

    // Render with or without bloom
    if (this.bloomEnabled) {
      this.renderBloom();
    } else {
      this.renderer.setRenderTarget(null);
      this.renderer.clear();
      this.renderer.render(this.scene, this.camera);
    }
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  loadVolume(data: Float32Array, sizeX: number, sizeY: number, sizeZ: number) {
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

    // Enhanced shader material with lighting
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uVolume: { value: tex3d },
        uColormap: { value: this.colormaps.get(this.currentColormap)! },
        uThreshold: { value: 0.1 },
        uWindowCenter: { value: 0.5 },
        uWindowWidth: { value: 1.0 },
        uOpacity: { value: 0.03 },
        uSteps: { value: 256 },
        uLightDir: { value: new THREE.Vector3(0.5, 0.8, 0.3).normalize() },
        uLightIntensity: { value: 1.2 },
        uAmbientIntensity: { value: 0.2 },
        uTime: { value: 0 },
      },
      vertexShader: volumeVertexShader,
      fragmentShader: volumeFragmentShader,
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

    // Create neural particles and synapses now that we have a volume
    this.createNeuralParticles();
    this.createSynapses();

    // Animate camera
    this.animateCamera(0.5, Math.PI / 4, 2.2);

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
    if (this.material) this.material.uniforms.uThreshold.value = value;
  }

  setOpacity(value: number) {
    if (this.material) this.material.uniforms.uOpacity.value = value;
  }

  setSteps(value: number) {
    if (this.material) this.material.uniforms.uSteps.value = value;
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

  // Lighting controls
  setLightDirection(x: number, y: number, z: number) {
    if (this.material) {
      this.material.uniforms.uLightDir.value.set(x, y, z).normalize();
    }
  }

  setLightIntensity(value: number) {
    if (this.material) this.material.uniforms.uLightIntensity.value = value;
  }

  setAmbientIntensity(value: number) {
    if (this.material) this.material.uniforms.uAmbientIntensity.value = value;
  }

  // Bloom controls
  setBloomEnabled(enabled: boolean) {
    this.bloomEnabled = enabled;
  }

  setBloomIntensity(value: number) {
    this.bloomIntensity = value;
  }

  // Particle controls
  setParticlesEnabled(enabled: boolean) {
    this.particlesEnabled = enabled;
  }

  setParticleCount(count: number) {
    this.particleCount = count;
    if (this.volMesh) this.createNeuralParticles();
  }

  // Synapse controls
  setSynapsesEnabled(enabled: boolean) {
    this.synapsesEnabled = enabled;
  }

  // Auto-rotation
  setAutoRotate(enabled: boolean) {
    this.autoRotate = enabled;
  }

  setAutoRotateSpeed(speed: number) {
    this.autoRotateSpeed = speed;
  }

  // FPS
  getFps(): number {
    return this.currentFps;
  }

  // Slice controls (placeholder for future clipping)
  setSliceX(t: number) {}
  setSliceY(t: number) {}
  setSliceZ(t: number) {}

  destroy() {
    cancelAnimationFrame(this.animationId);
    this.renderer.dispose();
    this.sceneTarget.dispose();
    this.brightTarget.dispose();
    this.blurTargetH.dispose();
    this.blurTargetV.dispose();
  }
}
