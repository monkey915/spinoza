// main.js – Three.js 3D Visualization for spinoza
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { simulate, Table, BALL_RADIUS } from "./physics.js";

// ===== Coordinate mapping: sim(X,Y,Z) → Three(X,Z,Y) with Z-up → Y-up =====
function s2t(sx, sy, sz) {
  return new THREE.Vector3(sx, sz, sy);
}

// ===== Scene Setup =====

const canvas = document.getElementById("viewport");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
scene.fog = new THREE.Fog(0x1a1a2e, 8, 20);

const camera = new THREE.PerspectiveCamera(50, 2, 0.01, 50);
camera.position.copy(s2t(0.76, -1.8, 1.6));
camera.up.set(0, 1, 0);

const controls = new OrbitControls(camera, canvas);
controls.target.copy(s2t(0.76, 1.37, 0.76));
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.5;
controls.maxDistance = 10;
controls.update();

// ===== Lighting =====

scene.add(new THREE.AmbientLight(0x404060, 0.6));

const hemiLight = new THREE.HemisphereLight(0x87ceeb, 0x444444, 0.4);
scene.add(hemiLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(2, 4, 1);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
dirLight.shadow.camera.left = -3;
dirLight.shadow.camera.right = 3;
dirLight.shadow.camera.top = 3;
dirLight.shadow.camera.bottom = -3;
dirLight.shadow.camera.near = 0.5;
dirLight.shadow.camera.far = 10;
dirLight.shadow.bias = -0.001;
scene.add(dirLight);

const fillLight = new THREE.DirectionalLight(0x8888cc, 0.3);
fillLight.position.set(-2, 2, -1);
scene.add(fillLight);

// ===== Ground =====

const groundGeo = new THREE.PlaneGeometry(20, 20);
const groundMat = new THREE.MeshStandardMaterial({
  color: 0x2a2a3a,
  roughness: 0.8,
});
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.position.y = 0;
ground.receiveShadow = true;
scene.add(ground);

// ===== Table Construction =====

const TABLE = new Table();

function buildTable() {
  const tableGroup = new THREE.Group();
  const tw = TABLE.width;
  const tl = TABLE.length;
  const th = TABLE.height;
  const surfaceThickness = 0.025;
  const lineWidth = 0.02;
  const centerLineWidth = 0.003;

  // Surface
  const surfGeo = new THREE.BoxGeometry(tw, surfaceThickness, tl);
  const surfMat = new THREE.MeshStandardMaterial({
    color: 0x0d3b66,
    roughness: 0.3,
    metalness: 0.05,
  });
  const surface = new THREE.Mesh(surfGeo, surfMat);
  surface.position.copy(
    s2t(tw / 2, tl / 2, th - surfaceThickness / 2),
  );
  surface.castShadow = true;
  surface.receiveShadow = true;
  tableGroup.add(surface);

  // White lines
  const lineMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.5,
  });
  const lineY = th + 0.001;

  function addLine(cx, cy, w, l) {
    const g = new THREE.BoxGeometry(w, 0.002, l);
    const m = new THREE.Mesh(g, lineMat);
    m.position.copy(s2t(cx, cy, lineY));
    tableGroup.add(m);
  }

  // End lines (width of table, at y=0 and y=tl)
  addLine(tw / 2, lineWidth / 2, tw, lineWidth);
  addLine(tw / 2, tl - lineWidth / 2, tw, lineWidth);
  // Side lines (length of table, at x=0 and x=tw)
  addLine(lineWidth / 2, tl / 2, lineWidth, tl);
  addLine(tw - lineWidth / 2, tl / 2, lineWidth, tl);
  // Center line (lengthwise, for doubles)
  addLine(tw / 2, tl / 2, centerLineWidth, tl);

  // Table legs
  const legRadius = 0.03;
  const legHeight = th - surfaceThickness;
  const legGeo = new THREE.CylinderGeometry(
    legRadius,
    legRadius,
    legHeight,
    12,
  );
  const legMat = new THREE.MeshStandardMaterial({
    color: 0x333333,
    roughness: 0.6,
  });
  const legInset = 0.08;

  for (const [lx, ly] of [
    [legInset, legInset],
    [tw - legInset, legInset],
    [legInset, tl - legInset],
    [tw - legInset, tl - legInset],
  ]) {
    const leg = new THREE.Mesh(legGeo, legMat);
    leg.position.copy(s2t(lx, ly, legHeight / 2));
    leg.castShadow = true;
    tableGroup.add(leg);
  }

  // Net
  const netH = TABLE.netHeight;
  const netGeo = new THREE.PlaneGeometry(tw + 0.03, netH);
  const netMat = new THREE.MeshStandardMaterial({
    color: 0xcccccc,
    transparent: true,
    opacity: 0.4,
    side: THREE.DoubleSide,
    roughness: 0.8,
  });
  const net = new THREE.Mesh(netGeo, netMat);
  // Net is at y=tl/2, from z=th to z=th+netH
  net.position.copy(s2t(tw / 2, tl / 2, th + netH / 2));
  net.rotation.y = Math.PI / 2;
  tableGroup.add(net);

  // Net top cord (thin white line)
  const cordGeo = new THREE.CylinderGeometry(0.002, 0.002, tw + 0.03, 8);
  const cordMat = new THREE.MeshStandardMaterial({ color: 0xffffff });
  const cord = new THREE.Mesh(cordGeo, cordMat);
  cord.position.copy(s2t(tw / 2, tl / 2, th + netH));
  cord.rotation.z = Math.PI / 2;
  tableGroup.add(cord);

  // Net posts
  const postGeo = new THREE.CylinderGeometry(0.008, 0.008, netH + 0.02, 8);
  const postMat = new THREE.MeshStandardMaterial({ color: 0x666666 });
  for (const px of [-0.015, tw + 0.015]) {
    const post = new THREE.Mesh(postGeo, postMat);
    post.position.copy(s2t(px, tl / 2, th + netH / 2));
    post.castShadow = true;
    tableGroup.add(post);
  }

  return tableGroup;
}

const tableModel = buildTable();
scene.add(tableModel);

// ===== Ball =====

const VISUAL_BALL_SCALE = 1.0; // true physical size
const ballGeo = new THREE.SphereGeometry(
  BALL_RADIUS * VISUAL_BALL_SCALE,
  32,
  32,
);
const ballMat = new THREE.MeshStandardMaterial({
  color: 0xffffff,
  emissive: 0xffffff,
  emissiveIntensity: 0.15,
  roughness: 0.3,
  metalness: 0.05,
});
const ballMesh = new THREE.Mesh(ballGeo, ballMat);
ballMesh.castShadow = true;
scene.add(ballMesh);

// Spin indicator arrow on the ball
const arrowHelper = new THREE.ArrowHelper(
  new THREE.Vector3(0, 1, 0),
  new THREE.Vector3(0, 0, 0),
  BALL_RADIUS * 3,
  0xff4444,
  BALL_RADIUS * 1.2,
  BALL_RADIUS * 0.8,
);
arrowHelper.visible = false;
scene.add(arrowHelper);

// ===== Trajectory & Bounce Markers =====

let trajectoryLine = null;
let bounceMarkers = [];
let trajectoryData = null;
let netMarker = null;
let predLines = [];  // prediction overlay lines

function clearVisualization() {
  stopPredAnim();
  if (trajectoryLine) {
    scene.remove(trajectoryLine);
    trajectoryLine.geometry.dispose();
    trajectoryLine = null;
  }
  for (const m of bounceMarkers) {
    scene.remove(m);
    m.geometry.dispose();
  }
  bounceMarkers = [];
  if (netMarker) {
    scene.remove(netMarker);
    netMarker.geometry.dispose();
    netMarker = null;
  }
  for (const l of predLines) {
    scene.remove(l);
    l.geometry.dispose();
  }
  predLines = [];
}

function buildTrajectory(result) {
  clearVisualization();
  trajectoryData = result;

  const points = result.trajectory.map((p) =>
    s2t(p.state.pos.x, p.state.pos.y, p.state.pos.z),
  );

  if (points.length < 2) return;

  // Color gradient: yellow → orange (pre-bounce) → cyan (post-bounce)
  const colors = [];
  const firstBounceTime =
    result.bounces.length > 0 ? result.bounces[0].time : Infinity;
  const totalTime = result.trajectory[result.trajectory.length - 1].t;

  for (const p of result.trajectory) {
    if (p.t <= firstBounceTime) {
      // Pre-bounce: yellow → orange
      const f = firstBounceTime > 0 ? p.t / firstBounceTime : 0;
      colors.push(1.0, 0.9 - f * 0.4, 0.2 - f * 0.2);
    } else {
      // Post-bounce: cyan → blue
      const f =
        totalTime > firstBounceTime
          ? (p.t - firstBounceTime) / (totalTime - firstBounceTime)
          : 0;
      colors.push(0.2 - f * 0.1, 0.7 - f * 0.3, 1.0);
    }
  }

  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  geometry.setAttribute(
    "color",
    new THREE.Float32BufferAttribute(colors, 3),
  );

  const material = new THREE.LineBasicMaterial({
    vertexColors: true,
    linewidth: 2,
  });
  trajectoryLine = new THREE.Line(geometry, material);
  scene.add(trajectoryLine);

  // Bounce markers (pulsing rings)
  for (const bounce of result.bounces) {
    const ringGeo = new THREE.RingGeometry(0.02, 0.035, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xff6644,
      transparent: true,
      opacity: 0.8,
      side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.position.copy(
      s2t(bounce.landing.x, bounce.landing.y, bounce.landing.z + 0.001),
    );
    ring.rotation.x = -Math.PI / 2;
    scene.add(ring);
    bounceMarkers.push(ring);
  }

  // Net hit marker
  if (result.hitNet) {
    const lastPt = result.trajectory[result.trajectory.length - 1];
    const crossGeo = new THREE.SphereGeometry(0.015, 16, 16);
    const crossMat = new THREE.MeshBasicMaterial({
      color: 0xff0000,
      transparent: true,
      opacity: 0.8,
    });
    netMarker = new THREE.Mesh(crossGeo, crossMat);
    netMarker.position.copy(
      s2t(lastPt.state.pos.x, lastPt.state.pos.y, lastPt.state.pos.z),
    );
    scene.add(netMarker);
  }

  // Place ball at start
  const startPos = result.trajectory[0].state.pos;
  ballMesh.position.copy(s2t(startPos.x, startPos.y, startPos.z));
  ballMesh.visible = true;
}

// ===== Animation State =====

let animTime = 0;
let animPlaying = false;
let animSpeed = 0.3; // default: slow motion
let lastFrameTime = 0;

function getStateAtTime(t) {
  if (!trajectoryData || trajectoryData.trajectory.length === 0) return null;
  const traj = trajectoryData.trajectory;

  if (t <= traj[0].t) return traj[0].state;
  if (t >= traj[traj.length - 1].t) return traj[traj.length - 1].state;

  // Binary search for the interval
  let lo = 0,
    hi = traj.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (traj[mid].t <= t) lo = mid;
    else hi = mid;
  }

  // Linear interpolation
  const a = traj[lo],
    b = traj[hi];
  const f = (t - a.t) / (b.t - a.t);
  return {
    pos: {
      x: a.state.pos.x + (b.state.pos.x - a.state.pos.x) * f,
      y: a.state.pos.y + (b.state.pos.y - a.state.pos.y) * f,
      z: a.state.pos.z + (b.state.pos.z - a.state.pos.z) * f,
    },
    vel: {
      x: a.state.vel.x + (b.state.vel.x - a.state.vel.x) * f,
      y: a.state.vel.y + (b.state.vel.y - a.state.vel.y) * f,
      z: a.state.vel.z + (b.state.vel.z - a.state.vel.z) * f,
    },
    omega: {
      x: a.state.omega.x + (b.state.omega.x - a.state.omega.x) * f,
      y: a.state.omega.y + (b.state.omega.y - a.state.omega.y) * f,
      z: a.state.omega.z + (b.state.omega.z - a.state.omega.z) * f,
    },
  };
}

function updateBallPosition(t) {
  const state = getStateAtTime(t);
  if (!state) return;

  ballMesh.position.copy(s2t(state.pos.x, state.pos.y, state.pos.z));

  // Spin indicator
  const omegaNorm = Math.sqrt(
    state.omega.x ** 2 + state.omega.y ** 2 + state.omega.z ** 2,
  );
  if (omegaNorm > 5) {
    arrowHelper.visible = true;
    arrowHelper.position.copy(ballMesh.position);
    // Arrow shows rotation axis per right-hand rule (omega vector)
    // IMPORTANT: omega is a pseudo-vector — the Y/Z swap in s2t (det=-1)
    // requires negation to preserve correct rotation direction in Three.js
    const dir = s2t(
      -state.omega.x / omegaNorm,
      -state.omega.y / omegaNorm,
      -state.omega.z / omegaNorm,
    ).normalize();
    arrowHelper.setDirection(dir);
    arrowHelper.setLength(
      BALL_RADIUS * 2 + (omegaNorm / 300) * BALL_RADIUS * 3,
      BALL_RADIUS * 1.0,
      BALL_RADIUS * 0.6,
    );
  } else {
    arrowHelper.visible = false;
  }

  // Live spin overlay
  const spinOverlay = document.getElementById("spin-overlay");
  if (spinOverlay) {
    if (omegaNorm > 1) {
      const label = state.omega.x > 10 ? "Topspin" : state.omega.x < -10 ? "Backspin" : "Sidespin";
      const color = state.omega.x > 10 ? "#44ff66" : state.omega.x < -10 ? "#ff6644" : "#ffcc44";
      spinOverlay.innerHTML = `<span style="color:${color}">⟳ ${Math.round(omegaNorm)} rad/s ${label}</span>`;
      spinOverlay.style.display = "block";
    } else {
      spinOverlay.style.display = "none";
    }
  }

  // Rotate ball based on spin (pseudo-vector: negate + Y/Z swap)
  const dt = 0.016;
  ballMesh.rotation.x += -state.omega.x * dt * animSpeed;
  ballMesh.rotation.y += -state.omega.z * dt * animSpeed;
  ballMesh.rotation.z += -state.omega.y * dt * animSpeed;
}

// ===== UI Controls =====

function getParam(id) {
  return parseFloat(document.getElementById(id).value);
}

function getParams() {
  return {
    speed: getParam("speed"),
    elevation: getParam("elevation"),
    azimuth: getParam("azimuth"),
    topspin: getParam("topspin"),
    backspin: getParam("backspin"),
    sidespin: getParam("sidespin"),
    x0: getParam("x0"),
    y0: getParam("y0"),
    z0: getParam("z0"),
  };
}

function updateValueDisplays() {
  document.querySelectorAll(".param-slider").forEach((slider) => {
    const display = document.getElementById(slider.id + "-val");
    if (display) {
      display.textContent = parseFloat(slider.value).toFixed(
        slider.step && slider.step.includes(".") ? slider.step.split(".")[1].length : 1,
      );
    }
  });
}

function runSimulation() {
  const params = getParams();
  const result = simulate(params);
  buildTrajectory(result);
  updateInfo(result, params);
  animTime = 0;
  updateTimeline();
  if (!animPlaying) {
    updateBallPosition(0);
  }
}

function updateInfo(result, params) {
  const info = document.getElementById("info");
  if (result.bounces.length > 0) {
    const b = result.bounces[0];
    const speed = Math.sqrt(
      b.preBounce.vel.x ** 2 +
        b.preBounce.vel.y ** 2 +
        b.preBounce.vel.z ** 2,
    );
    const onOwnHalf = b.landing.y < TABLE.length / 2;
    info.innerHTML = `
      <div class="info-row"><span>Impact after</span><span>${b.time.toFixed(4)} s</span></div>
      <div class="info-row"><span>Impact point</span><span>x=${b.landing.x.toFixed(3)} y=${b.landing.y.toFixed(3)} m</span></div>
      <div class="info-row"><span>|v| before impact</span><span>${speed.toFixed(2)} m/s</span></div>
      <div class="info-row"><span>Table half</span><span>${onOwnHalf ? "Own" : "Opponent's"}</span></div>
      ${result.hitNet ? '<div class="info-row warn"><span>⚠ Hit the net!</span></div>' : ""}
    `;
  } else {
    info.innerHTML = `
      <div class="info-row warn"><span>${result.hitNet ? "⚠ Ball hits the net!" : "⚠ Ball missed the table!"}</span></div>
    `;
  }
}

function updateTimeline() {
  if (!trajectoryData) return;
  const tMax = trajectoryData.trajectory[trajectoryData.trajectory.length - 1].t;
  const slider = document.getElementById("timeline");
  slider.max = tMax.toFixed(5);
  slider.value = animTime.toFixed(5);
  document.getElementById("time-display").textContent =
    `t = ${animTime.toFixed(3)} s`;
}

// ===== Event Listeners =====

document.querySelectorAll(".param-slider").forEach((slider) => {
  slider.addEventListener("input", () => {
    updateValueDisplays();
    runSimulation();
  });
});

document.getElementById("btn-play").addEventListener("click", () => {
  animPlaying = !animPlaying;
  document.getElementById("btn-play").textContent = animPlaying ? "⏸" : "▶";
  if (animPlaying) lastFrameTime = performance.now();
});

document.getElementById("btn-reset").addEventListener("click", () => {
  animTime = 0;
  animPlaying = false;
  document.getElementById("btn-play").textContent = "▶";
  updateTimeline();
  updateBallPosition(0);
});

document.getElementById("anim-speed").addEventListener("input", (e) => {
  animSpeed = parseFloat(e.target.value);
  document.getElementById("anim-speed-val").textContent =
    animSpeed.toFixed(2) + "×";
});

document.getElementById("timeline").addEventListener("input", (e) => {
  animTime = parseFloat(e.target.value);
  animPlaying = false;
  document.getElementById("btn-play").textContent = "▶";
  updateBallPosition(animTime);
  updateRobotAnimation(animTime);
  document.getElementById("time-display").textContent =
    `t = ${animTime.toFixed(3)} s`;
});

// Camera presets
document.getElementById("cam-side").addEventListener("click", () => {
  camera.position.copy(s2t(-1.5, 1.37, 1.2));
  controls.target.copy(s2t(0.76, 1.37, 0.76));
  controls.update();
});
document.getElementById("cam-top").addEventListener("click", () => {
  camera.position.copy(s2t(0.76, 1.37, 3.5));
  controls.target.copy(s2t(0.76, 1.37, 0.76));
  controls.update();
});
document.getElementById("cam-server").addEventListener("click", () => {
  camera.position.copy(s2t(0.76, -1.8, 1.6));
  controls.target.copy(s2t(0.76, 1.37, 0.76));
  controls.update();
});
document.getElementById("cam-receiver").addEventListener("click", () => {
  camera.position.copy(s2t(0.76, 4.5, 1.6));
  controls.target.copy(s2t(0.76, 1.37, 0.76));
  controls.update();
});
document.getElementById("cam-robot").addEventListener("click", () => {
  // Behind and slightly above robot, looking at the table
  camera.position.copy(s2t(-0.5, 3.8, 1.5));
  controls.target.copy(s2t(0.76, 2.0, 0.85));
  controls.update();
});

// Paddle close-up: positions camera near the current paddle position
let lastPaddlePos = null; // stored from showReplay
document.getElementById("cam-paddle").addEventListener("click", () => {
  const p = lastPaddlePos || { x: 0.76, y: 2.7, z: 0.9 };
  // Camera 40cm to the side and 20cm above, looking at paddle
  camera.position.copy(s2t(p.x - 0.4, p.y + 0.15, p.z + 0.2));
  controls.target.copy(s2t(p.x, p.y, p.z));
  controls.update();
});

// Presets
document.getElementById("preset-topspin").addEventListener("click", () => {
  setParams({ speed: 9, elevation: 15, topspin: 150, backspin: 0, sidespin: 0, azimuth: 0 });
});
document.getElementById("preset-backspin").addEventListener("click", () => {
  setParams({ speed: 5, elevation: 18, topspin: 0, backspin: 100, sidespin: 0, azimuth: 0 });
});
document.getElementById("preset-side").addEventListener("click", () => {
  setParams({ speed: 8, elevation: 15, topspin: 50, backspin: 0, sidespin: 100, azimuth: 5 });
});
document.getElementById("preset-fast").addEventListener("click", () => {
  setParams({ speed: 12, elevation: 8, topspin: 100, backspin: 0, sidespin: 0, azimuth: 0 });
});

function setParams(p) {
  for (const [key, val] of Object.entries(p)) {
    const el = document.getElementById(key);
    if (el) el.value = val;
  }
  updateValueDisplays();
  runSimulation();
}

// ===== Resize =====

function onResize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", onResize);

// ===== Animation Loop =====

function animate(time) {
  requestAnimationFrame(animate);

  if (animPlaying && trajectoryData) {
    const dt = (time - lastFrameTime) / 1000;
    lastFrameTime = time;
    animTime += dt * animSpeed;

    const tMax =
      trajectoryData.trajectory[trajectoryData.trajectory.length - 1].t;
    if (animTime >= tMax) {
      animTime = tMax;
      if (replayAutoplay && replayData) {
        // Brief pause, then auto-advance
        animPlaying = false;
        replayPauseTimer = replayPauseBetween;
      } else {
        animPlaying = false;
        document.getElementById("btn-play").textContent = "▶";
      }
    }

    updateBallPosition(animTime);
    updateTimeline();
    updateRobotAnimation(animTime);
  }

  // Pulse bounce markers
  if (bounceMarkers.length > 0) {
    const pulse = 0.6 + 0.4 * Math.sin(time * 0.005);
    for (const m of bounceMarkers) {
      m.material.opacity = pulse;
      m.scale.setScalar(0.8 + 0.4 * Math.sin(time * 0.003));
    }
  }

  // Autoplay: pause between replays then advance
  if (replayAutoplay && !animPlaying && replayPauseTimer > 0) {
    replayPauseTimer -= 0.016; // ~60fps frame time
    if (replayPauseTimer <= 0) {
      replayPauseTimer = 0;
      showReplay(replayIndex + 1);
      animTime = 0;
      animPlaying = true;
      lastFrameTime = time;
      document.getElementById("btn-play").textContent = "⏸";
    }
  }

  // Prediction animation
  updatePredAnimation(time);

  controls.update();
  renderer.render(scene, camera);
}

// ===== AI Replay Mode =====

let replayData = null;
let replayIndex = 0;
let replayMode = false;
let replayAutoplay = false;
let replayPauseBetween = 1.2; // seconds pause between replays
let replayPauseTimer = 0;
let returnTrajectoryLine = null;
let paddleMesh = null;
let swingArrow = null;

// Paddle mesh (flat disc + handle to the side like a human holds it)
function createPaddleMesh() {
  const group = new THREE.Group();
  const discGeo = new THREE.CylinderGeometry(0.085, 0.085, 0.015, 32);
  const discMat = new THREE.MeshStandardMaterial({
    color: 0xcc2222,
    roughness: 0.6,
    metalness: 0.1,
  });
  const disc = new THREE.Mesh(discGeo, discMat);
  disc.castShadow = true;
  group.add(disc);

  // Handle — positioned to the side, direction set dynamically in showReplay
  const handleGeo = new THREE.CylinderGeometry(0.012, 0.012, 0.12, 8);
  const handleMat = new THREE.MeshStandardMaterial({
    color: 0x664422,
    roughness: 0.8,
  });
  const handle = new THREE.Mesh(handleGeo, handleMat);
  handle.name = "paddle-handle";
  group.add(handle);

  group.visible = false;
  scene.add(group);
  return group;
}

// Swing direction arrow (cyan arrow showing swing vector)
function createSwingArrow() {
  const dir = new THREE.Vector3(0, 0, 1);
  const arrow = new THREE.ArrowHelper(dir, new THREE.Vector3(), 0.3, 0x00ddff, 0.06, 0.04);
  arrow.visible = false;
  scene.add(arrow);
  return arrow;
}

paddleMesh = createPaddleMesh();
swingArrow = createSwingArrow();

// ===== Robot Arm =====

const ROBOT = {
  // Base position in sim coords: centered behind table end
  baseX: TABLE.width / 2,    // 0.7625m — center of table
  baseY: TABLE.length + 0.15, // just behind table edge (receiver side)
  baseZ: TABLE.height,        // shoulder at table height
  upperArmLen: 0.40,          // shoulder → elbow
  forearmLen: 0.35,           // elbow → wrist
  handleLen: 0.10,            // paddle handle length (in disc plane, ~100mm real)
  paddleR: 0.085,             // paddle disc radius
};

let robotGroup = null;
let robotShoulderYaw = null;   // φ1: rotation around vertical (0-360°)
let robotShoulderPitch = null; // φ2: elevation from vertical (0-180°)
let robotUpperArm = null;      // mesh
let robotElbow = null;         // φ3: elbow bend (0-180°)
let robotForearm = null;       // mesh
let robotWrist = null;         // φ4: wrist tilt (paddle orientation)
let robotPaddleGroup = null;   // disc group — for FK debug (world position readout)
let robotFKDiscWorld = null;   // last FK-verified disc world pos (Three.js coords)
let robotVisible = true;

function createRobotArm() {
  const group = new THREE.Group();
  const jointMat = new THREE.MeshStandardMaterial({ color: 0x888888, metalness: 0.6, roughness: 0.4 });
  const jointGeo = new THREE.SphereGeometry(0.035, 16, 16);

  // Support pillar from floor to shoulder height
  const pillarHeight = ROBOT.baseZ;
  const pillarGeo = new THREE.CylinderGeometry(0.04, 0.05, pillarHeight, 12);
  const pillarMat = new THREE.MeshStandardMaterial({ color: 0x444444, metalness: 0.7, roughness: 0.3 });
  const pillar = new THREE.Mesh(pillarGeo, pillarMat);
  pillar.position.y = -pillarHeight / 2;
  pillar.castShadow = true;
  group.add(pillar);

  // Floor base plate
  const plateGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.02, 16);
  const plateMat = new THREE.MeshStandardMaterial({ color: 0x333333, metalness: 0.8, roughness: 0.2 });
  const plate = new THREE.Mesh(plateGeo, plateMat);
  plate.position.y = -pillarHeight;
  plate.receiveShadow = true;
  group.add(plate);

  // Shoulder mount
  const baseGeo = new THREE.CylinderGeometry(0.06, 0.06, 0.04, 16);
  const baseMat = new THREE.MeshStandardMaterial({ color: 0x555555, metalness: 0.8, roughness: 0.3 });
  const baseMesh = new THREE.Mesh(baseGeo, baseMat);
  baseMesh.castShadow = true;
  group.add(baseMesh);

  // Shoulder joint sphere
  const shoulderJoint = new THREE.Mesh(jointGeo.clone(), jointMat.clone());
  shoulderJoint.castShadow = true;
  group.add(shoulderJoint);

  // φ1: Shoulder yaw pivot (rotation around vertical axis, 0-360°)
  robotShoulderYaw = new THREE.Group();
  group.add(robotShoulderYaw);

  // φ2: Shoulder pitch pivot (elevation from vertical, 0-180°)
  robotShoulderPitch = new THREE.Group();
  robotShoulderYaw.add(robotShoulderPitch);

  // Upper arm segment
  const armMat = new THREE.MeshStandardMaterial({ color: 0x2266aa, metalness: 0.5, roughness: 0.4 });
  const upperArmGeo = new THREE.CylinderGeometry(0.025, 0.022, ROBOT.upperArmLen, 12);
  robotUpperArm = new THREE.Mesh(upperArmGeo, armMat);
  robotUpperArm.position.y = ROBOT.upperArmLen / 2;
  robotUpperArm.castShadow = true;
  robotShoulderPitch.add(robotUpperArm);

  // φ3: Elbow pivot (bend 0-180°)
  robotElbow = new THREE.Group();
  robotElbow.position.y = ROBOT.upperArmLen;
  robotShoulderPitch.add(robotElbow);

  const elbowJoint = new THREE.Mesh(jointGeo.clone(), jointMat.clone());
  elbowJoint.castShadow = true;
  robotElbow.add(elbowJoint);

  // Forearm segment
  const forearmMat = new THREE.MeshStandardMaterial({ color: 0x2288cc, metalness: 0.5, roughness: 0.4 });
  const forearmGeo = new THREE.CylinderGeometry(0.022, 0.018, ROBOT.forearmLen, 12);
  robotForearm = new THREE.Mesh(forearmGeo, forearmMat);
  robotForearm.position.y = ROBOT.forearmLen / 2;
  robotForearm.castShadow = true;
  robotElbow.add(robotForearm);

  // φ4: Wrist pivot (paddle tilt)
  robotWrist = new THREE.Group();
  robotWrist.position.y = ROBOT.forearmLen;
  robotElbow.add(robotWrist);

  const wristJoint = new THREE.Mesh(
    new THREE.SphereGeometry(0.025, 12, 12),
    new THREE.MeshStandardMaterial({ color: 0xaaaaaa, metalness: 0.6, roughness: 0.3 })
  );
  wristJoint.castShadow = true;
  robotWrist.add(wristJoint);

  // === Paddle: handle + blade, both COPLANAR in wrist local XZ plane ===
  // Y = paddle normal direction. Handle and disc lie in the XZ plane (Y≈0).
  // Handle extends from wrist (0,0,0) in +X direction for handleLen.
  // Disc edge meets handle at (handleLen, 0, 0).
  // Disc center at (handleLen + paddleR, 0, 0).
  const paddleR = ROBOT.paddleR;
  const handleLen = ROBOT.handleLen;
  const woodMat = new THREE.MeshStandardMaterial({ color: 0x8B6914, roughness: 0.65 });
  const darkWoodMat = new THREE.MeshStandardMaterial({ color: 0x6B4F12, roughness: 0.6 });

  // Handle: flat wooden piece lying along +X in the disc plane
  // Real TT handle: ~100mm long, ~25mm wide, ~23mm thick
  const handleGeo = new THREE.BoxGeometry(handleLen, 0.023, 0.025);
  const handleMesh = new THREE.Mesh(handleGeo, woodMat);
  handleMesh.position.set(handleLen / 2, 0, 0);
  handleMesh.castShadow = true;
  robotWrist.add(handleMesh);

  // Handle butt cap (rounded end at grip bottom, near wrist)
  const buttCap = new THREE.Mesh(
    new THREE.SphereGeometry(0.014, 8, 8),
    darkWoodMat
  );
  buttCap.scale.set(0.5, 0.8, 0.9);
  buttCap.position.set(0, 0, 0);
  robotWrist.add(buttCap);

  // Throat: flared transition from handle to blade
  const throatGeo = new THREE.BoxGeometry(0.025, 0.020, 0.032);
  const throatMesh = new THREE.Mesh(throatGeo, darkWoodMat);
  throatMesh.position.set(handleLen + 0.005, 0, 0);
  throatMesh.castShadow = true;
  robotWrist.add(throatMesh);

  // Paddle disc — center at (handleLen + paddleR, 0, 0), in the disc plane
  const paddleGroup = new THREE.Group();
  paddleGroup.position.set(handleLen + paddleR, 0, 0);

  // Red rubber side (hitting side, faces +Y)
  const discGeo = new THREE.CylinderGeometry(paddleR, paddleR, 0.008, 32);
  const discMatRed = new THREE.MeshStandardMaterial({ color: 0xcc2222, roughness: 0.6, metalness: 0.1 });
  const disc = new THREE.Mesh(discGeo, discMatRed);
  disc.castShadow = true;
  paddleGroup.add(disc);

  // Black rubber on back side
  const discMatBlack = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.7, metalness: 0.05 });
  const discBack = new THREE.Mesh(discGeo.clone(), discMatBlack);
  discBack.position.y = -0.003;
  paddleGroup.add(discBack);

  // Paddle edge ring
  const edgeGeo = new THREE.TorusGeometry(paddleR, 0.004, 8, 32);
  const edgeMat = new THREE.MeshStandardMaterial({ color: 0x444444, roughness: 0.5 });
  const edge = new THREE.Mesh(edgeGeo, edgeMat);
  edge.rotation.x = Math.PI / 2;
  paddleGroup.add(edge);

  robotWrist.add(paddleGroup);
  robotPaddleGroup = paddleGroup;

  // Position the whole group at the robot base in Three.js coords
  group.position.copy(s2t(ROBOT.baseX, ROBOT.baseY, ROBOT.baseZ));

  scene.add(group);
  return group;
}

/**
 * Solve 4-DOF IK for the robot arm.
 *
 * φ1 (shoulderYaw): rotation around vertical axis to face the target
 * φ2 (shoulderPitch): elevation in the arm plane (from vertical down)
 * φ3 (elbowAngle): bend at the elbow (0=folded, π=straight)
 * φ4 (wristTilt): paddle orientation (applied separately from paddle action)
 *
 * The IK reduces to a 2D problem in the arm plane after yaw rotation.
 * Returns { phi1, phi2, phi3, reachable }
 */
function solveIK(targetX, targetY, targetZ) {
  const L1 = ROBOT.upperArmLen;
  const L2 = ROBOT.forearmLen; // IK targets the wrist, not the paddle tip

  // Target relative to shoulder base (sim coords)
  const dx = targetX - ROBOT.baseX;
  const dy = targetY - ROBOT.baseY;
  const dz = targetZ - ROBOT.baseZ;

  // φ1: Yaw — rotation around vertical to face target
  const phi1 = Math.atan2(dx, -dy); // -dy because robot faces -Y (toward net)

  // Project into arm plane: horizontal distance + vertical offset
  const horizontalDist = Math.sqrt(dx * dx + dy * dy);
  const d = Math.sqrt(horizontalDist * horizontalDist + dz * dz);

  // Check reachability
  const maxReach = L1 + L2 - 0.01;
  const minReach = Math.abs(L1 - L2) + 0.01;
  let reachable = true;
  let clampedD = d;
  if (d > maxReach) { clampedD = maxReach; reachable = false; }
  if (d < minReach) { clampedD = minReach; reachable = false; }

  // φ3: Elbow angle (cosine rule on the triangle shoulder-elbow-wrist)
  const cosElbow = (clampedD * clampedD - L1 * L1 - L2 * L2) / (2 * L1 * L2);
  const phi3 = Math.acos(Math.max(-1, Math.min(1, cosElbow)));

  // φ2: Shoulder pitch — angle in arm plane
  // alpha: angle from horizontal to the target direction
  const alpha = Math.atan2(dz, horizontalDist);
  // beta: angle offset due to elbow bend (from the shoulder-target line to the upper arm)
  const beta = Math.acos(Math.max(-1, Math.min(1,
    (L1 * L1 + clampedD * clampedD - L2 * L2) / (2 * L1 * clampedD)
  )));
  // Elbow-up configuration (α - β): arm reaches forward, elbow above arm plane
  // This is the standard 2-link IK q1 for elbow-up (q2 positive)
  const phi2 = alpha - beta;

  return { phi1, phi2, phi3, reachable };
}

/**
 * Compute paddle normal N and radial direction R in sim coords.
 * R = normalize(N × sim_up), pointing sideways in the disc plane.
 */
function paddleFrameSim(tiltX, tiltZ) {
  const nx = Math.sin(tiltZ || 0);
  const nz = Math.sin(tiltX || 0);
  const ny = -Math.sqrt(Math.max(0, 1 - nx * nx - nz * nz));
  // R = N × (0,0,1) in sim coords
  let rx = ny, ry = -nx, rz = 0;
  const rLen = Math.sqrt(rx * rx + ry * ry);
  if (rLen < 1e-6) {
    // N nearly vertical — use N × (1,0,0) instead
    rx = 0; ry = nz; rz = -ny;
    const rl = Math.sqrt(rx * rx + ry * ry + rz * rz);
    rx /= rl; ry /= rl; rz /= rl;
  } else {
    rx /= rLen; ry /= rLen; rz /= rLen;
  }
  return { nx, ny, nz, rx, ry, rz };
}

/**
 * Compute wrist quaternion so paddle face matches the physics normal.
 * Uses a full rotation frame: local +Y → N (paddle normal),
 * local +X → R (radial direction in disc plane).
 */
function computeWristQuat(phi1, phi2, phi3, tiltX, tiltZ) {
  const { nx, ny, nz, rx, ry, rz } = paddleFrameSim(tiltX, tiltZ);

  // N and R in Three.js coords: sim(x,y,z) → three(x,z,y)
  let N = new THREE.Vector3(nx, nz, ny);
  let R = new THREE.Vector3(rx, rz, ry);

  // Exact parent rotation: R_y(-phi1) * R_x(phi2 + phi3 - π/2)
  const qYaw = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(0, 1, 0), -phi1
  );
  const qPE = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(1, 0, 0), phi2 + phi3 - Math.PI / 2
  );
  const parentQuat = qYaw.clone().multiply(qPE);

  // Build full world frame: X=R, Y=N, Z=R×N (right-handed)
  const F = new THREE.Vector3().crossVectors(R, N).normalize();
  // Re-orthogonalize R from N and F
  R.crossVectors(N, F).normalize();

  const mat = new THREE.Matrix4().makeBasis(R, N, F);
  const worldQuat = new THREE.Quaternion().setFromRotationMatrix(mat);

  const parentInv = parentQuat.clone().invert();
  return parentInv.multiply(worldQuat);
}

/**
 * Compute wrist position from paddle position and orientation.
 * Handle and disc are coplanar (in the disc plane). The wrist is at the
 * handle butt, offset from disc center by (handleLen + paddleR) in the
 * radial direction R. No offset along the normal N.
 * wrist = paddle_pos - (handleLen + paddleR) * R (in sim coords)
 */
function paddleToWrist(paddleX, paddleY, paddleZ, tiltX, tiltZ) {
  const { rx, ry, rz } = paddleFrameSim(tiltX, tiltZ);
  const totalR = ROBOT.handleLen + ROBOT.paddleR;
  return {
    x: paddleX - totalR * rx,
    y: paddleY - totalR * ry,
    z: paddleZ - totalR * rz,
  };
}

// Debug markers for IK verification
let debugMarkers = [];
function clearDebugMarkers() {
  for (const m of debugMarkers) scene.remove(m);
  debugMarkers = [];
}
function addDebugSphere(simX, simY, simZ, color, radius) {
  const geo = new THREE.SphereGeometry(radius || 0.015, 12, 12);
  const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8 });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.copy(s2t(simX, simY, simZ));
  scene.add(mesh);
  debugMarkers.push(mesh);
  return mesh;
}

// Robot arm animation state
let robotTargetAngles = null;   // { yaw, pitch, elbow, wristQuat }
let robotCurrentAngles = null;  // same shape, for interpolation
let robotContactTime = 0;       // time when arm should reach target
const ROBOT_REST = { yaw: 0, pitch: -0.3, elbow: -0.8, wristQuat: new THREE.Quaternion() };

/** Apply raw joint angles to the robot arm meshes */
function applyRobotAngles(angles) {
  if (!robotGroup) return;
  robotShoulderYaw.rotation.set(0, angles.yaw, 0);
  robotShoulderPitch.rotation.set(angles.pitch, 0, 0);
  robotElbow.rotation.set(angles.elbow, 0, 0);
  robotWrist.quaternion.copy(angles.wristQuat);
  robotGroup.visible = robotVisible;
}

/**
 * Set up robot arm animation target for a paddle action.
 * The arm will animate from rest to hit position during the serve.
 */
function setRobotTarget(paddleX, paddleY, paddleZ, tiltX, tiltZ, contactTime) {
  if (!robotGroup || !robotVisible) return;

  const wrist = paddleToWrist(paddleX, paddleY, paddleZ, tiltX, tiltZ);
  const ik = solveIK(wrist.x, wrist.y, wrist.z);
  const yaw = -ik.phi1;
  const pitch = -(Math.PI / 2 - ik.phi2);
  const elbow = ik.phi3;
  const wristQuat = computeWristQuat(ik.phi1, ik.phi2, ik.phi3, tiltX, tiltZ);

  robotTargetAngles = { yaw, pitch, elbow, wristQuat };
  robotCurrentAngles = { ...ROBOT_REST, wristQuat: ROBOT_REST.wristQuat.clone() };
  robotContactTime = contactTime || 0.3;

  // FK verification: check disc world position at target pose
  applyRobotAngles(robotTargetAngles);
  robotGroup.updateMatrixWorld(true);
  robotFKDiscWorld = null;
  if (robotPaddleGroup) {
    const discWorld = new THREE.Vector3();
    robotPaddleGroup.getWorldPosition(discWorld);
    const expected = s2t(paddleX, paddleY, paddleZ);
    const err = discWorld.distanceTo(expected) * 1000;
    console.log(`🎯 FK check: disc at (${discWorld.x.toFixed(3)}, ${discWorld.y.toFixed(3)}, ${discWorld.z.toFixed(3)})` +
      ` expected (${expected.x.toFixed(3)}, ${expected.y.toFixed(3)}, ${expected.z.toFixed(3)}) error=${err.toFixed(1)}mm`);
    if (err > 5) console.warn(`⚠️ FK mismatch: ${err.toFixed(1)}mm!`);
    // Store for green debug sphere (added AFTER clearDebugMarkers in loadReplay)
    robotFKDiscWorld = discWorld.clone();
  }

  // Start from rest pose
  applyRobotAngles(robotCurrentAngles);
}

/** Smoothstep easing: 0→0, 0.5→0.5, 1→1, smooth at both ends */
function smoothstep(t) {
  t = Math.max(0, Math.min(1, t));
  return t * t * (3 - 2 * t);
}

/** Update robot arm animation (called each frame) */
function updateRobotAnimation(currentTime) {
  if (!robotTargetAngles || !robotGroup || !robotVisible) return;

  // Arm starts moving from t=0, arrives at contactTime
  // Use 80% of serve time so arm arrives slightly early (more natural)
  const moveTime = robotContactTime * 0.8;
  const startDelay = robotContactTime * 0.1; // small delay before moving
  const t = (currentTime - startDelay) / moveTime;
  const progress = smoothstep(Math.max(0, Math.min(1, t)));

  const lerp = (a, b, f) => a + (b - a) * f;
  const angles = {
    yaw: lerp(ROBOT_REST.yaw, robotTargetAngles.yaw, progress),
    pitch: lerp(ROBOT_REST.pitch, robotTargetAngles.pitch, progress),
    elbow: lerp(ROBOT_REST.elbow, robotTargetAngles.elbow, progress),
    wristQuat: ROBOT_REST.wristQuat.clone().slerp(robotTargetAngles.wristQuat, progress),
  };

  applyRobotAngles(angles);
}

/**
 * Position the robot arm instantly (for scrubbing / non-animated use).
 */
function positionRobotArm(paddleX, paddleY, paddleZ, tiltX, tiltZ) {
  if (!robotGroup || !robotVisible) return;

  const wrist = paddleToWrist(paddleX, paddleY, paddleZ, tiltX, tiltZ);
  const ik = solveIK(wrist.x, wrist.y, wrist.z);
  const angles = {
    yaw: -ik.phi1,
    pitch: -(Math.PI / 2 - ik.phi2),
    elbow: ik.phi3,
    wristQuat: computeWristQuat(ik.phi1, ik.phi2, ik.phi3, tiltX, tiltZ),
  };

  applyRobotAngles(angles);
  robotTargetAngles = null; // cancel any animation
}

/** Set robot to rest pose (arm relaxed, slightly forward) */
function robotRestPose() {
  if (!robotGroup) return;
  applyRobotAngles(ROBOT_REST);
  robotTargetAngles = null;
}

robotGroup = createRobotArm();
robotRestPose();

function clearReturnVisualization() {
  if (returnTrajectoryLine) {
    scene.remove(returnTrajectoryLine);
    returnTrajectoryLine.geometry.dispose();
    returnTrajectoryLine = null;
  }
  paddleMesh.visible = false;
  if (swingArrow) swingArrow.visible = false;
  clearDebugMarkers();
  robotRestPose();
}

function loadReplay(replay) {
  clearVisualization();
  clearReturnVisualization();
  replayMode = true;

  // Convert replay JSON arrays to internal trajectory format
  // Serve trajectory: [t, x, y, z, vx, vy, vz, ox, oy, oz]
  const serveTraj = replay.serve_trajectory.map((p) => ({
    t: p[0],
    state: {
      pos: { x: p[1], y: p[2], z: p[3] },
      vel: { x: p[4], y: p[5], z: p[6] },
      omega: { x: p[7], y: p[8], z: p[9] },
    },
  }));

  // Return trajectory (time continues from serve end)
  const returnTraj = replay.return_trajectory.map((p) => ({
    t: p[0],
    state: {
      pos: { x: p[1], y: p[2], z: p[3] },
      vel: { x: p[4], y: p[5], z: p[6] },
      omega: { x: p[7], y: p[8], z: p[9] },
    },
  }));

  // Merge into unified timeline for animation
  // Trim serve trajectory at paddle contact point so ball doesn't fly through paddle
  // Only trim on actual contact (not paddle_miss — ball flies past)
  let trimmedServeTraj = serveTraj;
  if (replay.outcome !== "paddle_miss" && replay.contact_pos && replay.contact_pos.length === 3 && serveTraj.length > 1) {
    const contactY = replay.contact_pos[1]; // paddle Y position
    let cutIdx = serveTraj.length;
    for (let i = 1; i < serveTraj.length; i++) {
      if (serveTraj[i].state.pos.y >= contactY) {
        cutIdx = i;
        break;
      }
    }
    trimmedServeTraj = serveTraj.slice(0, cutIdx + 1);
  }
  // Offset return trajectory timestamps so they continue after serve
  const serveEndTime = trimmedServeTraj.length > 0 ? trimmedServeTraj[trimmedServeTraj.length - 1].t : 0;
  const offsetReturnTraj = returnTraj.map((p) => ({
    ...p,
    t: p.t + serveEndTime,
  }));
  const mergedTraj = [...trimmedServeTraj, ...offsetReturnTraj];
  const serveBounces = (replay.serve_bounces || []).map((b) => ({
    landing: { x: b[1], y: b[2], z: b[3] },
    time: b[0],
  }));
  const returnBounces = (replay.return_bounces || []).map((b) => ({
    landing: { x: b[1], y: b[2], z: b[3] },
    time: b[0],
  }));

  // Build serve trajectory line (yellow → orange) — trimmed at paddle contact
  const servePoints = trimmedServeTraj.map((p) =>
    s2t(p.state.pos.x, p.state.pos.y, p.state.pos.z)
  );

  if (servePoints.length >= 2) {
    const serveColors = [];
    for (let i = 0; i < trimmedServeTraj.length; i++) {
      const f = i / (trimmedServeTraj.length - 1);
      serveColors.push(1.0, 0.9 - f * 0.4, 0.2 - f * 0.2); // yellow → orange
    }
    const serveGeo = new THREE.BufferGeometry().setFromPoints(servePoints);
    serveGeo.setAttribute("color", new THREE.Float32BufferAttribute(serveColors, 3));
    trajectoryLine = new THREE.Line(
      serveGeo,
      new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 })
    );
    scene.add(trajectoryLine);
  }

  // Build return trajectory line (green for success, red-ish otherwise)
  if (returnTraj.length >= 2) {
    const returnPoints = returnTraj.map((p) =>
      s2t(p.state.pos.x, p.state.pos.y, p.state.pos.z)
    );
    const isSuccess = replay.outcome === "success";
    const returnColors = [];
    for (let i = 0; i < returnTraj.length; i++) {
      const f = i / (returnTraj.length - 1);
      if (isSuccess) {
        returnColors.push(0.2, 0.9 - f * 0.3, 0.3 - f * 0.1); // green
      } else {
        returnColors.push(0.9, 0.3 - f * 0.2, 0.2); // red
      }
    }
    const returnGeo = new THREE.BufferGeometry().setFromPoints(returnPoints);
    returnGeo.setAttribute("color", new THREE.Float32BufferAttribute(returnColors, 3));
    returnTrajectoryLine = new THREE.Line(
      returnGeo,
      new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 })
    );
    scene.add(returnTrajectoryLine);
  }

  // Bounce markers
  for (const bounce of [...serveBounces, ...returnBounces]) {
    const isReturn = returnBounces.includes(bounce);
    const ringGeo = new THREE.RingGeometry(0.02, 0.035, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: isReturn ? (replay.outcome === "success" ? 0x44ff66 : 0xff4444) : 0xff6644,
      transparent: true,
      opacity: 0.8,
      side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.position.copy(s2t(bounce.landing.x, bounce.landing.y, bounce.landing.z + 0.001));
    ring.rotation.x = -Math.PI / 2;
    scene.add(ring);
    bounceMarkers.push(ring);
  }

  // Paddle position — use actual paddle coordinates, not ball contact point
  const pa = replay.paddle;
  if (pa) {
    lastPaddlePos = { x: pa.paddle_x, y: pa.paddle_y, z: pa.paddle_z };

    paddleMesh.position.copy(s2t(pa.paddle_x, pa.paddle_y, pa.paddle_z));
    paddleMesh.rotation.set(0, 0, 0);
    paddleMesh.rotateX(-Math.PI / 2); // face forward (along Y)
    paddleMesh.rotateX(pa.tilt_x || 0);
    paddleMesh.rotateZ(pa.tilt_z || 0);

    // Handle direction: point toward nearest side (like a human holding it)
    const tableCenterX = 1.525 / 2; // 0.7625m
    const handleDir = pa.paddle_x > tableCenterX ? -1 : 1; // left side → handle right, right side → handle left
    const handle = paddleMesh.getObjectByName("paddle-handle");
    if (handle) {
      handle.rotation.set(0, 0, Math.PI / 2); // rotate cylinder to horizontal
      handle.position.set(handleDir * 0.075, 0, -0.02); // offset to the side + slightly down
    }

    paddleMesh.visible = !robotVisible; // hide floating paddle when robot arm has its own

    // Set up robot arm animation — arm will move from rest to hit position during serve
    setRobotTarget(pa.paddle_x, pa.paddle_y, pa.paddle_z, pa.tilt_x, pa.tilt_z, serveEndTime);

    // Debug markers: paddle target (cyan), contact pos (magenta), wrist target (yellow)
    clearDebugMarkers();
    addDebugSphere(pa.paddle_x, pa.paddle_y, pa.paddle_z, 0x00ffff, 0.012); // cyan = paddle action pos
    if (replay.contact_pos && replay.contact_pos.length === 3) {
      addDebugSphere(replay.contact_pos[0], replay.contact_pos[1], replay.contact_pos[2], 0xff00ff, 0.012); // magenta = actual contact
    }
    const wrist = paddleToWrist(pa.paddle_x, pa.paddle_y, pa.paddle_z, pa.tilt_x, pa.tilt_z);
    addDebugSphere(wrist.x, wrist.y, wrist.z, 0xffff00, 0.010); // yellow = IK wrist target
    // Green = ACTUAL disc center from scene graph FK (should overlap cyan if IK is correct)
    if (robotFKDiscWorld) {
      addDebugSphere(robotFKDiscWorld.x, robotFKDiscWorld.z, robotFKDiscWorld.y, 0x00ff00, 0.018);
    }

    // Swing direction arrow: shows the swing vector (direction + speed)
    // Sim coords: swing is in -Y direction (toward opponent), elevated by swing_elevation
    // swing_elevation > 0 = upward component, < 0 = downward
    if (swingArrow && pa.swing_speed) {
      const elev = pa.swing_elevation || 0;
      const spd = pa.swing_speed || 5;
      // Swing direction in sim coords: forward (-Y) + upward (Z) component
      const sy = -Math.cos(elev); // forward component (toward opponent)
      const sz = Math.sin(elev);   // upward component
      // Convert sim (x,y,z) → Three.js (x,z,y) via s2t convention
      const dir = new THREE.Vector3(0, sz, sy).normalize();
      const arrowLen = 0.15 + (spd / 12.0) * 0.25; // length scales with speed
      swingArrow.position.copy(s2t(pa.paddle_x, pa.paddle_y, pa.paddle_z));
      swingArrow.setDirection(dir);
      swingArrow.setLength(arrowLen, 0.06, 0.04);
      swingArrow.visible = true;
    }
  }

  // Use merged trajectory for ball animation
  trajectoryData = {
    trajectory: mergedTraj,
    bounces: [...serveBounces, ...returnBounces],
    hitNet: replay.outcome === "return_hit_net" || replay.outcome === "hit_net",
  };

  // Set ball at start
  if (mergedTraj.length > 0) {
    const p0 = mergedTraj[0].state.pos;
    ballMesh.position.copy(s2t(p0.x, p0.y, p0.z));
    ballMesh.visible = true;
  }

  animTime = 0;
  updateTimeline();
  updateReplayInfo(replay);
}

function updateReplayInfo(replay) {
  const info = document.getElementById("replay-info");
  const outcomeColors = {
    success: "#44ff66",
    paddle_miss: "#ff6644",
    return_hit_net: "#ffaa44",
    hit_net: "#ffaa44",
    return_missed_table: "#ff8844",
    missed_table: "#ff8844",
    bad_serve: "#888888",
  };
  const color = outcomeColors[replay.outcome] || "#ffffff";
  const outcomeLabel = replay.outcome.replace(/_/g, " ");

  let html = `
    <div class="info-row">
      <span>Outcome</span>
      <span style="color:${color}; font-weight:bold;">${outcomeLabel}</span>
    </div>
    <div class="info-row"><span>Reward</span><span style="color:${replay.reward > 2 ? '#44ff66' : replay.reward > 1 ? '#88cc44' : replay.reward > 0 ? '#ffaa44' : '#ff4444'}; font-weight:bold;">${replay.reward.toFixed(3)}</span></div>
  `;

  const pa = replay.paddle;
  const tiltDeg = ((pa.tilt_x || 0) * 180 / Math.PI).toFixed(1);
  const tiltLabel = pa.tilt_x > 0.05 ? "geschlossen" : pa.tilt_x < -0.05 ? "offen" : "vertikal";
  const tiltColor = pa.tilt_x > 0.05 ? "#44aaff" : pa.tilt_x < -0.05 ? "#ffaa44" : "#aaaaaa";
  const elevDeg = ((pa.swing_elevation || 0) * 180 / Math.PI).toFixed(1);
  html += `
    <div class="info-row"><span>Paddle X</span><span>${pa.paddle_x.toFixed(3)} m</span></div>
    <div class="info-row"><span>Paddle Y</span><span>${(pa.paddle_y || 0).toFixed(3)} m</span></div>
    <div class="info-row"><span>Paddle Z</span><span>${pa.paddle_z.toFixed(3)} m</span></div>
    <div class="info-row"><span>Swing</span><span>${pa.swing_speed.toFixed(1)} m/s ↗${elevDeg}°</span></div>
    <div class="info-row">
      <span>Schläger</span>
      <span style="color:${tiltColor}; font-weight:bold;">${tiltDeg}° ${tiltLabel}</span>
    </div>
  `;

  if (replay.landing && replay.landing.length === 2) {
    html += `<div class="info-row"><span>Landing</span><span>x=${replay.landing[0].toFixed(3)} y=${replay.landing[1].toFixed(3)}</span></div>`;
  }

  if (replay.hit_omega) {
    const ox = replay.hit_omega[0];
    const spinLabel = ox > 10 ? "🔄 Topspin" : ox < -10 ? "🔃 Backspin" : "— Flat";
    const spinColor = ox > 10 ? "#44ff66" : ox < -10 ? "#ff6644" : "#aaaaaa";
    html += `
      <div class="info-row">
        <span>Spin type</span>
        <span style="color:${spinColor}; font-weight:bold;">${spinLabel}</span>
      </div>
      <div class="info-row">
        <span>ω.x (topspin)</span>
        <span>${ox.toFixed(1)} rad/s</span>
      </div>
    `;
  }

  info.innerHTML = html;
}

document.getElementById("replay-load").addEventListener("click", async () => {
  try {
    const source = document.getElementById("replay-source").value;
    const resp = await fetch(`${source}?t=${Date.now()}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${source} not found`);
    replayData = await resp.json();
    document.getElementById("replay-controls").style.display = "block";
    document.getElementById("replay-load").textContent = "✓ Loaded";
    document.getElementById("replay-load").style.opacity = "0.5";
    showReplay(0);
    // Switch to receiver camera for best view of returns
    camera.position.copy(s2t(-1.5, 1.37, 1.2));
    controls.target.copy(s2t(0.76, 1.37, 0.76));
    controls.update();
  } catch (e) {
    document.getElementById("replay-info").innerHTML =
      `<div class="info-row warn"><span>⚠ ${e.message}</span></div>
       <div class="info-row"><span style="color:var(--text-dim)">Run export_replays.py first</span></div>`;
    document.getElementById("replay-controls").style.display = "block";
  }
});

document.getElementById("replay-prev").addEventListener("click", () => {
  showReplay(replayIndex - 1);
});

document.getElementById("replay-next").addEventListener("click", () => {
  showReplay(replayIndex + 1);
});

document.getElementById("replay-autoplay").addEventListener("click", () => {
  replayAutoplay = !replayAutoplay;
  const btn = document.getElementById("replay-autoplay");
  if (replayAutoplay) {
    btn.textContent = "⏹ Stop";
    // Restart current replay from beginning
    showReplay(replayIndex);
    animTime = 0;
    animPlaying = true;
    animSpeed = 0.4;
    document.getElementById("anim-speed").value = "0.40";
    document.getElementById("anim-speed-val").textContent = "0.40×";
    lastFrameTime = performance.now();
    document.getElementById("btn-play").textContent = "⏸";
  } else {
    btn.textContent = "▶ Auto-play";
    replayPauseTimer = 0;
  }
});

document.getElementById("replay-filter").addEventListener("click", () => {
  const btn = document.getElementById("replay-filter");
  const filters = ["all", "success", "paddle_miss", "hit_net", "missed_table", "return_missed_table", "return_hit_net"];
  const current = btn.dataset.filter;
  const next = filters[(filters.indexOf(current) + 1) % filters.length];
  btn.dataset.filter = next;
  const labels = { all: "All", success: "✓ Success", paddle_miss: "✗ Miss", hit_net: "🥅 Net", missed_table: "↗ Off Table", return_missed_table: "↗ Off Table", return_hit_net: "🥅 Net" };
  btn.textContent = labels[next] || next;
  // Re-show current filtered replay
  if (replayData) showReplay(0);
});

document.getElementById("robot-toggle").addEventListener("click", () => {
  robotVisible = !robotVisible;
  const btn = document.getElementById("robot-toggle");
  btn.textContent = robotVisible ? "🦾 Robot" : "🦾 ✗";
  btn.style.opacity = robotVisible ? "1" : "0.5";
  if (robotGroup) robotGroup.visible = robotVisible;
});

function getFilteredReplays() {
  if (!replayData || !replayData.replays) return [];
  const filter = document.getElementById("replay-filter").dataset.filter;
  if (filter === "all") return replayData.replays;
  return replayData.replays.filter((r) => r.outcome === filter);
}

// Override showReplay to use filtered list
function showReplay(index) {
  const replays = getFilteredReplays();
  if (replays.length === 0) return;
  replayIndex = ((index % replays.length) + replays.length) % replays.length;
  document.getElementById("replay-index").textContent =
    `${replayIndex + 1} / ${replays.length}`;
  loadReplay(replays[replayIndex]);
}

// ===== Live Training Monitor =====

let liveAutoInterval = null;

function drawLiveChart(history) {
  const canvas = document.getElementById("live-chart");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!history || history.length < 2) return;

  ctx.fillStyle = "#0d0d1a";
  ctx.fillRect(0, 0, W, H);

  // Grid lines at 25/50/75%
  ctx.strokeStyle = "#2a2a40";
  ctx.lineWidth = 1;
  [25, 50, 75].forEach(pct => {
    const cy = H - (pct / 100) * H;
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();
    ctx.fillStyle = "#444466"; ctx.font = "9px monospace";
    ctx.fillText(`${pct}%`, 2, cy - 2);
  });

  // Success rate line
  ctx.strokeStyle = "#ff8844";
  ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach((pt, i) => {
    const x = (i / (history.length - 1)) * W;
    const y = H - (pt.success / 100) * H;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Dot at last point
  const last = history[history.length - 1];
  const lx = W - 3;
  const ly = H - (last.success / 100) * H;
  ctx.fillStyle = "#ff8844";
  ctx.beginPath(); ctx.arc(lx, ly, 4, 0, Math.PI * 2); ctx.fill();
}

async function fetchLiveStats() {
  const statusEl = document.getElementById("live-status");
  try {
    const resp = await fetch(`training_live.json?t=${Date.now()}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} – training_live.json nicht gefunden`);
    const data = await resp.json();

    document.getElementById("live-stats").style.display = "block";
    document.getElementById("live-chart").style.display = "block";
    document.getElementById("live-stage").textContent   = `Stage ${data.stage}`;
    document.getElementById("live-steps").textContent   = `${(data.step / 1e6).toFixed(2)}M`;
    document.getElementById("live-success").textContent = `${data.success.toFixed(1)}%`;
    document.getElementById("live-eps").textContent     = `${data.eps_s.toFixed(0)}`;

    const mins = Math.floor(data.elapsed / 60);
    const hrs  = Math.floor(mins / 60);
    document.getElementById("live-elapsed").textContent =
      hrs > 0 ? `${hrs}h ${mins % 60}min` : `${mins}min`;

    statusEl.textContent = `Letzte Aktualisierung: ${data.last_update}`;
    drawLiveChart(data.history);
  } catch (e) {
    statusEl.textContent = `⚠ ${e.message}`;
  }
}

document.getElementById("live-refresh").addEventListener("click", fetchLiveStats);

document.getElementById("live-auto").addEventListener("click", (e) => {
  const btn = e.currentTarget;
  if (liveAutoInterval) {
    clearInterval(liveAutoInterval);
    liveAutoInterval = null;
    btn.textContent = "Auto 3min";
    btn.classList.remove("btn-accent");
  } else {
    fetchLiveStats();
    liveAutoInterval = setInterval(fetchLiveStats, 180_000);
    btn.textContent = "⏸ Stop Auto";
    btn.classList.add("btn-accent");
  }
});

document.getElementById("live-replay-btn").addEventListener("click", async () => {
  const btn = document.getElementById("live-replay-btn");
  try {
    btn.textContent = "⏳ Lade…";
    const resp = await fetch(`replays_live.json?t=${Date.now()}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    replayData = await resp.json();
    document.getElementById("replay-controls").style.display = "block";
    showReplay(0);
    camera.position.copy(s2t(-1.5, 1.37, 1.2));
    controls.target.copy(s2t(0.76, 1.37, 0.76));
    controls.update();
    btn.textContent = "✓ Live geladen";
    btn.classList.add("btn-accent");
  } catch (e) {
    btn.textContent = "Live Replays";
    document.getElementById("live-status").textContent = `⚠ Replay-Fehler: ${e.message}`;
  }
});

// ===== Trajectory Predictor Visualization =====

let predData = null;
let predCategory = null;
let predIndex = 0;
let predNInputIdx = 2; // index into n_input_options

// Prediction animation state
let predAnimActive = false;
let predAnimPlaying = false;
let predAnimFrame = 0;
let predAnimLastTime = 0;
let predAnimGT = null;     // ground truth positions [[x,y,z],...]
let predAnimPR = null;     // predicted positions [[x,y,z],...]
let predAnimStates = null; // full_states for GT spin per frame
let predAnimSpin = null;   // predicted spin [wx,wy,wz] (constant)
let predAnimNInput = 0;
let predFrameDt = 1.0 / 60.0;

// Two extra ball meshes for prediction mode
const predGTBall = new THREE.Mesh(
  new THREE.SphereGeometry(BALL_RADIUS, 24, 24),
  new THREE.MeshStandardMaterial({ color: 0x44ddff, emissive: 0x44ddff, emissiveIntensity: 0.3 })
);
predGTBall.visible = false;
scene.add(predGTBall);

const predPRBall = new THREE.Mesh(
  new THREE.SphereGeometry(BALL_RADIUS, 24, 24),
  new THREE.MeshStandardMaterial({ color: 0xff8844, emissive: 0xff8844, emissiveIntensity: 0.3 })
);
predPRBall.visible = false;
scene.add(predPRBall);

// Spin arrows for prediction balls
const predGTArrow = new THREE.ArrowHelper(
  new THREE.Vector3(0,1,0), new THREE.Vector3(), BALL_RADIUS*3, 0x44ddff, BALL_RADIUS*1.0, BALL_RADIUS*0.6
);
predGTArrow.visible = false;
scene.add(predGTArrow);

const predPRArrow = new THREE.ArrowHelper(
  new THREE.Vector3(0,1,0), new THREE.Vector3(), BALL_RADIUS*3, 0xff8844, BALL_RADIUS*1.0, BALL_RADIUS*0.6
);
predPRArrow.visible = false;
scene.add(predPRArrow);

function projectToScreen(pos3d) {
  const v = pos3d.clone().project(camera);
  const rect = canvas.getBoundingClientRect();
  return {
    x: (v.x * 0.5 + 0.5) * rect.width + rect.left,
    y: (-v.y * 0.5 + 0.5) * rect.height + rect.top,
  };
}

function updateSpinArrow(arrow, pos3d, omega) {
  const norm = Math.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2);
  if (norm > 5) {
    arrow.visible = true;
    arrow.position.copy(pos3d);
    const dir = s2t(-omega[0]/norm, -omega[1]/norm, -omega[2]/norm).normalize();
    arrow.setDirection(dir);
    arrow.setLength(BALL_RADIUS*2 + (norm/300)*BALL_RADIUS*3, BALL_RADIUS*1.0, BALL_RADIUS*0.6);
  } else {
    arrow.visible = false;
  }
}

function spinLabel(omega) {
  const norm = Math.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2);
  if (norm < 5) return '';
  const label = omega[0] > 10 ? 'TS' : omega[0] < -10 ? 'BS' : 'SS';
  return `${Math.round(norm)} rad/s ${label}`;
}

function stopPredAnim() {
  predAnimActive = false;
  predAnimPlaying = false;
  predGTBall.visible = false;
  predPRBall.visible = false;
  predGTArrow.visible = false;
  predPRArrow.visible = false;
  document.getElementById('pred-gt-label').style.display = 'none';
  document.getElementById('pred-pr-label').style.display = 'none';
  const btn = document.getElementById('pred-play');
  if (btn) btn.textContent = '▶ Play';
}

function showPrediction() {
  if (!predData || !predCategory) return;
  const cat = predData.categories[predCategory];
  if (!cat || cat.trajectories.length === 0) return;

  const traj = cat.trajectories[predIndex];
  const nInput = predData.n_input_options[predNInputIdx];
  const pred = traj.predictions[String(nInput)];
  if (!pred) return;

  // Stop any running replay/anim
  animPlaying = false;
  stopPredAnim();
  clearVisualization();
  ballMesh.visible = false;
  paddleMesh.visible = false;
  swingArrow.visible = false;
  arrowHelper.visible = false;

  predFrameDt = predData.frame_dt || (1.0 / 60.0);

  const gt = traj.ground_truth;
  const pr = pred.predicted;

  // Draw trail lines (dimmed, visible from start)
  // Observed: white dashed
  const obsPoints = [];
  for (let i = 0; i < nInput; i++) obsPoints.push(s2t(gt[i][0], gt[i][1], gt[i][2]));
  if (obsPoints.length >= 2) {
    const geo = new THREE.BufferGeometry().setFromPoints(obsPoints);
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0x666666, linewidth: 1 }));
    scene.add(line);
    predLines.push(line);
  }

  // GT future: cyan dim trail
  const gtPts = [];
  for (let i = Math.max(0, nInput-1); i < gt.length; i++) gtPts.push(s2t(gt[i][0], gt[i][1], gt[i][2]));
  if (gtPts.length >= 2) {
    const geo = new THREE.BufferGeometry().setFromPoints(gtPts);
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0x225577, linewidth: 1 }));
    scene.add(line);
    predLines.push(line);
  }

  // Predicted future: orange dim trail
  const prPts = [];
  for (let i = Math.max(0, nInput-1); i < pr.length; i++) prPts.push(s2t(pr[i][0], pr[i][1], pr[i][2]));
  if (prPts.length >= 2) {
    const geo = new THREE.BufferGeometry().setFromPoints(prPts);
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0x553311, linewidth: 1 }));
    scene.add(line);
    predLines.push(line);
  }

  // Yellow split marker
  const sp = gt[nInput - 1];
  const splitMesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.008, 12, 12),
    new THREE.MeshBasicMaterial({ color: 0xffff00 })
  );
  splitMesh.position.copy(s2t(sp[0], sp[1], sp[2]));
  scene.add(splitMesh);
  predLines.push(splitMesh);

  // Store animation data
  predAnimGT = gt;
  predAnimPR = pr;
  predAnimStates = traj.full_states;
  predAnimSpin = pred.predicted_spin || null;
  predAnimNInput = nInput;
  predAnimFrame = 0;
  predAnimActive = true;

  // Position balls at frame 0
  predGTBall.visible = true;
  predPRBall.visible = true;
  predGTBall.position.copy(s2t(gt[0][0], gt[0][1], gt[0][2]));
  predPRBall.position.copy(s2t(pr[0][0], pr[0][1], pr[0][2]));

  // Update info panel
  const info = document.getElementById('pred-info');
  const finalErr = pred.errors_mm[pred.errors_mm.length - 1];

  // Spin info
  let gtSpinText = '';
  if (predAnimStates && predAnimStates[0]) {
    const s = predAnimStates[0];
    const gts = [s[6], s[7], s[8]];
    gtSpinText = `<b>GT Spin:</b> ${spinLabel(gts)}`;
  }
  let prSpinText = '';
  if (predAnimSpin) {
    prSpinText = `<b>Pred Spin:</b> ${spinLabel(predAnimSpin)}`;
  }

  info.innerHTML = `
    <b>Serve:</b> ${traj.serve_speed} m/s<br>
    <b>Topspin:</b> ${traj.topspin} rad/s &nbsp;
    <b>Backspin:</b> ${traj.backspin} rad/s<br>
    <b>Sidespin:</b> ${traj.sidespin} rad/s<br>
    ${gtSpinText ? gtSpinText + '<br>' : ''}
    ${prSpinText ? prSpinText + '<br>' : ''}
    <hr style="border-color:var(--panel-border); margin:6px 0;">
    <span style="color:#44ddff">●</span> Ground Truth &nbsp;
    <span style="color:#ff8844">●</span> Vorhersage &nbsp;
    <span style="color:#ffff00">●</span> Trennpunkt<br>
    <hr style="border-color:var(--panel-border); margin:6px 0;">
    <b>Ø Fehler:</b> ${pred.avg_error_mm} mm &nbsp;
    <b>Max:</b> ${pred.max_error_mm} mm<br>
    <b>Endpunkt-Fehler:</b> ${finalErr} mm
  `;

  document.getElementById('pred-index').textContent = `${predIndex + 1} / ${cat.trajectories.length}`;
}

function updatePredAnimation(time) {
  if (!predAnimActive) return;

  if (predAnimPlaying) {
    const dt = (time - predAnimLastTime) / 1000;
    predAnimLastTime = time;
    // Advance frame (slow-mo: 0.3× speed for visibility)
    predAnimFrame += dt / predFrameDt * 0.3;
  }

  const f = Math.min(predAnimFrame, predAnimGT.length - 1);
  const fi = Math.floor(f);
  const frac = f - fi;
  const fi2 = Math.min(fi + 1, predAnimGT.length - 1);

  // Interpolate GT ball position
  const gtPos = s2t(
    predAnimGT[fi][0] + (predAnimGT[fi2][0] - predAnimGT[fi][0]) * frac,
    predAnimGT[fi][1] + (predAnimGT[fi2][1] - predAnimGT[fi][1]) * frac,
    predAnimGT[fi][2] + (predAnimGT[fi2][2] - predAnimGT[fi][2]) * frac,
  );
  predGTBall.position.copy(gtPos);

  // Interpolate predicted ball position
  const prPos = s2t(
    predAnimPR[fi][0] + (predAnimPR[fi2][0] - predAnimPR[fi][0]) * frac,
    predAnimPR[fi][1] + (predAnimPR[fi2][1] - predAnimPR[fi][1]) * frac,
    predAnimPR[fi][2] + (predAnimPR[fi2][2] - predAnimPR[fi][2]) * frac,
  );
  predPRBall.position.copy(prPos);

  // GT spin arrow (from full_states)
  if (predAnimStates && predAnimStates[fi]) {
    const s = predAnimStates[fi];
    updateSpinArrow(predGTArrow, gtPos, [s[6], s[7], s[8]]);
  }

  // Predicted spin arrow (constant across frames)
  if (predAnimSpin) {
    updateSpinArrow(predPRArrow, prPos, predAnimSpin);
  }

  // Project to screen and update HTML labels
  const gtLabel = document.getElementById('pred-gt-label');
  const prLabel = document.getElementById('pred-pr-label');

  const gtScreen = projectToScreen(gtPos);
  gtLabel.style.left = gtScreen.x + 'px';
  gtLabel.style.top = (gtScreen.y - 20) + 'px';
  gtLabel.style.display = 'block';

  const prScreen = projectToScreen(prPos);
  prLabel.style.left = prScreen.x + 'px';
  prLabel.style.top = (prScreen.y - 20) + 'px';
  prLabel.style.display = 'block';

  // Show spin text on labels
  if (predAnimStates && predAnimStates[fi]) {
    const s = predAnimStates[fi];
    const gtSpin = spinLabel([s[6], s[7], s[8]]);
    gtLabel.textContent = gtSpin ? `GT: ${gtSpin}` : 'GT';
  }
  if (predAnimSpin) {
    const prSpin = spinLabel(predAnimSpin);
    prLabel.textContent = prSpin ? `Pred: ${prSpin}` : 'Pred';
  } else {
    prLabel.textContent = 'Pred';
  }

  // Highlight when past the split point
  if (fi >= predAnimNInput) {
    predGTBall.material.emissiveIntensity = 0.5;
    predPRBall.material.emissiveIntensity = 0.5;
  } else {
    predGTBall.material.emissiveIntensity = 0.15;
    predPRBall.material.emissiveIntensity = 0.15;
  }

  // Loop when done
  if (predAnimFrame >= predAnimGT.length - 1 && predAnimPlaying) {
    predAnimPlaying = false;
    predAnimFrame = predAnimGT.length - 1;
    document.getElementById('pred-play').textContent = '▶ Play';
  }
}

document.getElementById('pred-load').addEventListener('click', async () => {
  const btn = document.getElementById('pred-load');
  btn.textContent = 'Loading...';
  try {
    const resp = await fetch(`predictions.json?t=${Date.now()}`);
    predData = await resp.json();

    const catContainer = document.getElementById('pred-categories');
    catContainer.innerHTML = '';
    const cats = Object.keys(predData.categories);
    for (const catName of cats) {
      const b = document.createElement('button');
      b.className = 'btn';
      b.textContent = predData.categories[catName].label;
      b.dataset.cat = catName;
      b.addEventListener('click', () => {
        predCategory = catName;
        predIndex = 0;
        // Highlight active category
        catContainer.querySelectorAll('.btn').forEach(x => x.classList.remove('btn-accent'));
        b.classList.add('btn-accent');
        showPrediction();
      });
      catContainer.appendChild(b);
    }

    // Setup N-input slider
    const slider = document.getElementById('pred-n-slider');
    slider.max = predData.n_input_options.length - 1;
    slider.value = predNInputIdx;
    document.getElementById('pred-n-label').textContent = predData.n_input_options[predNInputIdx];
    slider.addEventListener('input', () => {
      predNInputIdx = parseInt(slider.value);
      document.getElementById('pred-n-label').textContent = predData.n_input_options[predNInputIdx];
      showPrediction();
    });

    document.getElementById('pred-controls').style.display = 'block';
    btn.textContent = '✓ Loaded';

    // Auto-select first category
    if (cats.length > 0) {
      predCategory = cats[0];
      catContainer.querySelector('.btn').classList.add('btn-accent');
      showPrediction();
    }
  } catch (e) {
    btn.textContent = 'Load Predictions';
    console.error('Failed to load predictions:', e);
  }
});

document.getElementById('pred-prev').addEventListener('click', () => {
  if (!predData || !predCategory) return;
  const cat = predData.categories[predCategory];
  predIndex = (predIndex - 1 + cat.trajectories.length) % cat.trajectories.length;
  showPrediction();
});

document.getElementById('pred-next').addEventListener('click', () => {
  if (!predData || !predCategory) return;
  const cat = predData.categories[predCategory];
  predIndex = (predIndex + 1) % cat.trajectories.length;
  showPrediction();
});

document.getElementById('pred-play').addEventListener('click', () => {
  if (!predAnimActive) return;
  if (predAnimPlaying) {
    predAnimPlaying = false;
    document.getElementById('pred-play').textContent = '▶ Play';
  } else {
    // Restart if at end
    if (predAnimFrame >= predAnimGT.length - 1) predAnimFrame = 0;
    predAnimPlaying = true;
    predAnimLastTime = performance.now();
    document.getElementById('pred-play').textContent = '⏸ Pause';
  }
});

// ===== Init =====

onResize();
updateValueDisplays();
runSimulation();
requestAnimationFrame(animate);
