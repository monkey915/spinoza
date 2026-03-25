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

const VISUAL_BALL_SCALE = 1.5; // slightly larger for visibility
const ballGeo = new THREE.SphereGeometry(
  BALL_RADIUS * VISUAL_BALL_SCALE,
  32,
  32,
);
const ballMat = new THREE.MeshStandardMaterial({
  color: 0xffeeee,
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

function clearVisualization() {
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
    const dir = s2t(
      state.omega.x / omegaNorm,
      state.omega.y / omegaNorm,
      state.omega.z / omegaNorm,
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

  // Rotate ball based on spin (visual only)
  const dt = 0.016; // approximate frame time
  ballMesh.rotation.x += state.omega.z * dt * animSpeed;
  ballMesh.rotation.y += state.omega.y * dt * animSpeed;
  ballMesh.rotation.z += state.omega.x * dt * animSpeed;
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
    replayPauseTimer -= (time - lastFrameTime) / 1000;
    lastFrameTime = time;
    if (replayPauseTimer <= 0) {
      replayPauseTimer = 0;
      showReplay(replayIndex + 1);
      animPlaying = true;
      lastFrameTime = performance.now();
      document.getElementById("btn-play").textContent = "⏸";
    }
  }

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

// Paddle mesh (flat disc)
function createPaddleMesh() {
  const group = new THREE.Group();
  const discGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.005, 32);
  const discMat = new THREE.MeshStandardMaterial({
    color: 0xcc2222,
    roughness: 0.6,
    metalness: 0.1,
  });
  const disc = new THREE.Mesh(discGeo, discMat);
  disc.castShadow = true;
  group.add(disc);

  // Handle
  const handleGeo = new THREE.CylinderGeometry(0.012, 0.012, 0.12, 8);
  const handleMat = new THREE.MeshStandardMaterial({
    color: 0x664422,
    roughness: 0.8,
  });
  const handle = new THREE.Mesh(handleGeo, handleMat);
  handle.position.y = -0.065;
  group.add(handle);

  group.visible = false;
  scene.add(group);
  return group;
}

paddleMesh = createPaddleMesh();

function clearReturnVisualization() {
  if (returnTrajectoryLine) {
    scene.remove(returnTrajectoryLine);
    returnTrajectoryLine.geometry.dispose();
    returnTrajectoryLine = null;
  }
  paddleMesh.visible = false;
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
  const mergedTraj = [...serveTraj, ...returnTraj];
  const serveBounces = (replay.serve_bounces || []).map((b) => ({
    landing: { x: b[1], y: b[2], z: b[3] },
    time: b[0],
  }));
  const returnBounces = (replay.return_bounces || []).map((b) => ({
    landing: { x: b[1], y: b[2], z: b[3] },
    time: b[0],
  }));

  // Build serve trajectory line (yellow → orange)
  const servePoints = serveTraj.map((p) =>
    s2t(p.state.pos.x, p.state.pos.y, p.state.pos.z)
  );

  if (servePoints.length >= 2) {
    const serveColors = [];
    for (let i = 0; i < serveTraj.length; i++) {
      const f = i / (serveTraj.length - 1);
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

  // Paddle position
  if (replay.contact_pos && replay.contact_pos.length === 3) {
    const cp = replay.contact_pos;
    paddleMesh.position.copy(s2t(cp[0], cp[1], cp[2]));
    // Orient paddle based on tilt
    const pa = replay.paddle;
    paddleMesh.rotation.set(0, 0, 0);
    paddleMesh.rotateX(-Math.PI / 2); // face forward (along Y)
    paddleMesh.rotateX(pa.tilt_x || 0);
    paddleMesh.rotateZ(pa.tilt_z || 0);
    paddleMesh.visible = true;
  }

  // Use merged trajectory for ball animation
  trajectoryData = {
    trajectory: mergedTraj,
    bounces: [...serveBounces, ...returnBounces],
    hitNet: replay.outcome === "return_hit_net",
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
    return_missed_table: "#ff8844",
    bad_serve: "#888888",
  };
  const color = outcomeColors[replay.outcome] || "#ffffff";
  const outcomeLabel = replay.outcome.replace(/_/g, " ");

  let html = `
    <div class="info-row">
      <span>Outcome</span>
      <span style="color:${color}; font-weight:bold;">${outcomeLabel}</span>
    </div>
    <div class="info-row"><span>Reward</span><span>${replay.reward.toFixed(3)}</span></div>
  `;

  const pa = replay.paddle;
  html += `
    <div class="info-row"><span>Paddle X</span><span>${pa.paddle_x.toFixed(3)} m</span></div>
    <div class="info-row"><span>Paddle Z</span><span>${pa.paddle_z.toFixed(3)} m</span></div>
    <div class="info-row"><span>Swing</span><span>${pa.swing_speed.toFixed(1)} m/s</span></div>
  `;

  if (replay.landing && replay.landing.length === 2) {
    html += `<div class="info-row"><span>Landing</span><span>x=${replay.landing[0].toFixed(3)} y=${replay.landing[1].toFixed(3)}</span></div>`;
  }

  info.innerHTML = html;
}

document.getElementById("replay-load").addEventListener("click", async () => {
  try {
    const resp = await fetch("replays.json");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: replays.json not found`);
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
    // Start playing current replay
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
  const filters = ["all", "success", "paddle_miss"];
  const current = btn.dataset.filter;
  const next = filters[(filters.indexOf(current) + 1) % filters.length];
  btn.dataset.filter = next;
  btn.textContent = next === "all" ? "All" : next === "success" ? "✓ Success" : "✗ Miss";
  // Re-show current filtered replay
  if (replayData) showReplay(0);
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

// ===== Init =====

onResize();
updateValueDisplays();
runSimulation();
requestAnimationFrame(animate);
