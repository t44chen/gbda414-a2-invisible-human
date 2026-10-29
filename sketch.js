// Invisible Human: GBDA 414 Group 1
// COCO-SSD scores how sure the AI is that you're a person:
// red = HUMAN DETECTED, amber = UNSURE, green = UNDETECTED.
// ml5 bodyPose (MoveNet) shows which body parts the AI can still see.

// ---------- Tunable settings ----------
const HIGH = 0.7; // score at or above this -> DETECTED (red)
const LOW = 0.3; // score below this -> UNDETECTED (green)
const MARGIN = 0.05; // hysteresis: score must move back this far to leave a state
const SMOOTHING = 0.5; // 0..1, higher = faster score changes
const KP_MIN = 0.3; // keypoint confidence that counts as "seen"
const POSE_EVERY = 2; // run bodyPose once every N COCO detections
const DETECT_GAP = 30; // ms pause between detections so drawing stays smooth
const SHOW_SKELETON = true; // draw the bodyPose skeleton over the video

const PARTS = [
  {
    label: "Face",
    names: ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
  },
  { label: "Hands", names: ["left_wrist", "right_wrist"] },
  {
    label: "Body",
    names: ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
  },
  {
    label: "Legs",
    names: ["left_knee", "right_knee", "left_ankle", "right_ankle"],
  },
];

const STATES = {
  LOADING: { label: "LOADING MODELS", accent: [150, 150, 160] },
  DETECTED: { label: "HUMAN DETECTED", accent: [255, 59, 59] }, // #ff3b3b
  UNSURE: { label: "UNSURE", accent: [255, 176, 32] }, // #ffb020
  UNDETECTED: { label: "UNDETECTED", accent: [47, 227, 140] }, // #2fe38c
};

// ---------- Visual style ----------
const BG = [11, 11, 13]; // #0b0b0d
const HEAD = '"Space Grotesk", sans-serif';
const MONO = '"JetBrains Mono", monospace';
const FADE_MS = 100; // colour fade time constant (~300 ms to settle)
const SCAN_MS = 5000; // time for the scan line to cross the video

const SKELETON = [
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
];

// ---------- State ----------
let video, cocoModel, bodyPose;
let people = []; // COCO "person" boxes {x, y, w, h, score} in video pixels
let keypoints = []; // MoveNet keypoints {x, y, confidence, name} in video pixels
let topScore = 0; // smoothed best person score
let state = "LOADING";
let tick = 0;
let accent = STATES.LOADING.accent.slice(); // current (fading) state colour
let shownScore = 0; // eased score for the confidence bar
let view; // where the video sits on screen: {x, y, w, h, s}

function setup() {
  createCanvas(windowWidth, windowHeight);
  video = createCapture({ video: { width: 640, height: 480 }, audio: false });
  video.hide();
  // Match the element size to the real camera size so both models use the same pixel space
  video.elt.addEventListener("loadedmetadata", () => {
    video.size(video.elt.videoWidth, video.elt.videoHeight);
  });

  // Canvas text doesn't trigger web font downloads on its own
  for (const f of [
    `300 20px ${HEAD}`,
    `500 20px ${HEAD}`,
    `300 20px ${MONO}`,
    `400 20px ${MONO}`,
    `500 20px ${MONO}`,
  ]) {
    document.fonts.load(f);
  }

  // WebGL is the steadiest backend for COCO-SSD (on WebGPU its box filtering blocks the page)
  tf.setBackend("webgl")
    .then(() =>
      Promise.all([
        cocoSsd.load(),
        new Promise((resolve) => {
          bodyPose = ml5.bodyPose(
            "MoveNet",
            { modelType: "SINGLEPOSE_LIGHTNING" },
            resolve,
          );
        }),
      ]),
    )
    .then(([m]) => {
      cocoModel = m;
      state = "UNDETECTED";
      detectLoop();
    });
}

// One loop runs both models in turn so they never compete for the GPU
async function detectLoop() {
  if (video.elt.readyState >= 2) {
    try {
      const preds = await cocoModel.detect(video.elt, 20, 0.05);
      people = preds
        .filter((p) => p.class === "person")
        .map((p) => ({
          x: p.bbox[0],
          y: p.bbox[1],
          w: p.bbox[2],
          h: p.bbox[3],
          score: p.score,
        }));
      const best = people.reduce((m, p) => max(m, p.score), 0);
      topScore = lerp(topScore, best, SMOOTHING);
      state = nextState(state, topScore);

      if (tick++ % POSE_EVERY === 0) {
        const poses = await bodyPose.detect(video);
        keypoints = poses.length ? poses[0].keypoints : [];
      }
    } catch (err) {
      console.warn(err);
    }
  }
  setTimeout(detectLoop, DETECT_GAP);
}

// Thresholds with hysteresis: entering a state uses HIGH/LOW, leaving it needs an extra MARGIN
function nextState(current, s) {
  if (current === "DETECTED" && s >= HIGH - MARGIN) return "DETECTED";
  if (current === "UNDETECTED" && s < LOW + MARGIN) return "UNDETECTED";
  if (s >= HIGH) return "DETECTED";
  if (s < LOW) return "UNDETECTED";
  return "UNSURE";
}

// ---------- Drawing ----------
function draw() {
  const u = min(width, (height * 16) / 9) / 100; // 1% of the largest 16:9 area that fits
  const k = 1 - exp(-deltaTime / FADE_MS);
  const target = STATES[state].accent;
  accent = accent.map((c, i) => c + (target[i] - c) * k);
  shownScore += (topScore - shownScore) * k;

  drawBackdrop();

  // Video area: centred, keeps the camera's aspect ratio
  const camW = video.elt.videoWidth || 640;
  const camH = video.elt.videoHeight || 480;
  let vh = height * 0.56;
  let vw = (vh * camW) / camH;
  if (vw > width * 0.58) {
    vw = width * 0.58;
    vh = (vw * camH) / camW;
  }
  view = {
    x: (width - vw) / 2,
    y: height * 0.215 + (height * 0.56 - vh) / 2,
    w: vw,
    h: vh,
    s: vw / camW,
  };

  drawVideo();
  if (state !== "LOADING") {
    if (SHOW_SKELETON) drawSkeleton(u);
    drawPeople(u);
  }
  drawScanLine(u);
  drawFrame(u);

  // Heading
  const q = "Can you make the AI stop seeing you as human?";
  type(q, width / 2, height * 0.07, {
    font: HEAD,
    weight: 300,
    size: fitType(q, 2.6 * u, width * 0.9, { font: HEAD, weight: 300 }),
    align: "center",
    color: "rgba(255,255,255,0.8)",
  });
  const label = STATES[state].label;
  const labelOpts = { font: HEAD, weight: 500, spacing: 0.6 * u };
  type(label, width / 2, height * 0.145, {
    ...labelOpts,
    size: fitType(label, 4.4 * u, width * 0.9, labelOpts),
    align: "center",
    color: css(accent),
    glow: 2 * u,
  });

  drawStats(u);
  drawFigurePanel(u);
}

// Near-black with a faint radial wash of the state colour behind the video
function drawBackdrop() {
  background(...BG);
  const ctx = drawingContext;
  const g = ctx.createRadialGradient(
    width / 2,
    height * 0.5,
    0,
    width / 2,
    height * 0.5,
    max(width, height) * 0.7,
  );
  g.addColorStop(0, css(accent, 0.14));
  g.addColorStop(1, css(accent, 0.05));
  ctx.save(); // keeps p5's cached fill colour in sync with the canvas
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, width, height);
  ctx.restore();
}

// Canvas glow, set directly because p5 1.x has no shadow API. Not wrapped in
// save/restore: that would undo p5's stroke/fill without p5 noticing.
function glow(col, blur) {
  drawingContext.shadowColor = col;
  drawingContext.shadowBlur = blur;
}

function drawVideo() {
  const { x, y, w, h } = view;
  push();
  translate(x + w, y);
  scale(-1, 1);
  image(video, 0, 0, w, h);
  pop();
  noStroke();
  fill(...BG, 40);
  rect(x, y, w, h);
}

function drawFrame(u) {
  const { x, y, w, h } = view;
  const f = 0.7 * u;
  glow(css(accent, 0.9), 3 * u);
  noFill();
  stroke(...accent);
  strokeWeight(0.35 * u);
  rect(x - f, y - f, w + 2 * f, h + 2 * f);
  glow("transparent", 0);
}

// Faint horizontal line sweeping down the video with a short fading trail
function drawScanLine(u) {
  const { x, y, w, h } = view;
  const sy = y + ((millis() % SCAN_MS) / SCAN_MS) * h;
  strokeWeight(1);
  for (let i = 0; i < 30; i++) {
    const ly = sy - i * 0.25 * u;
    if (ly < y) break;
    stroke(...accent, 28 * (1 - i / 30));
    line(x, ly, x + w, ly);
  }
  stroke(...accent, 70);
  line(x, sy, x + w, sy);
}

// Video pixels -> mirrored screen position
function toScreen(px, py) {
  return [view.x + view.w - px * view.s, view.y + py * view.s];
}

// Corner brackets around every person; the most confident one gets the state colour
function drawPeople(u) {
  const shown = people.filter((p) => p.score >= LOW);
  const top = shown.reduce((b, p) => (!b || p.score > b.score ? p : b), null);
  for (const p of shown) {
    const [x1, y1] = toScreen(p.x + p.w, p.y);
    const bw = p.w * view.s;
    const bh = p.h * view.s;
    const col = p === top ? css(accent) : "rgba(255,255,255,0.5)";
    const len = min(bw, bh) * 0.18;

    stroke(col);
    strokeWeight(max(1.5, 0.16 * u));
    noFill();
    for (const [cx, cy, dx, dy] of [
      [x1, y1, 1, 1],
      [x1 + bw, y1, -1, 1],
      [x1, y1 + bh, 1, -1],
      [x1 + bw, y1 + bh, -1, -1],
    ]) {
      line(cx, cy, cx + dx * len, cy);
      line(cx, cy, cx, cy + dy * len);
    }

    const tag = "HUMAN " + round(p.score * 100) + "%";
    const opts = { size: 1.25 * u, weight: 500, spacing: 0.15 * u };
    const th = 2.4 * u;
    const tw = typeWidth(tag, opts) + 1.6 * u;
    // Above the box; if the box touches the top of the video, inside at the bottom so it doesn't cover the face
    const ty =
      y1 - th - 0.6 * u >= view.y ? y1 - th - 0.6 * u : y1 + bh - th - 0.6 * u;
    noStroke();
    fill(...BG, 180);
    rect(x1, ty, tw, th, th / 2);
    type(tag, x1 + 0.8 * u, ty + th / 2, { ...opts, color: col });
  }
}

// Thin skeleton with small rings on joints and crosshairs on the face
function drawSkeleton(u) {
  const kp = {};
  for (const k of keypoints)
    if (k.confidence >= KP_MIN) kp[k.name] = toScreen(k.x, k.y);
  const col = [...accent, 180];

  stroke(...col);
  strokeWeight(1.5);
  noFill();
  for (const [a, b] of SKELETON) {
    if (kp[a] && kp[b]) line(...kp[a], ...kp[b]);
  }
  const r = 0.75 * u;
  for (const name of [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
  ]) {
    if (kp[name]) circle(...kp[name], r);
  }

  // Face: crosshairs on eyes and nose, rings on ears, dashed outline when both eyes are seen
  strokeWeight(1);
  for (const name of ["left_eye", "right_eye", "nose"]) {
    if (!kp[name]) continue;
    const [x, y] = kp[name];
    const a = 0.55 * u;
    const g = 0.18 * u;
    line(x - a, y, x - g, y);
    line(x + g, y, x + a, y);
    line(x, y - a, x, y - g);
    line(x, y + g, x, y + a);
  }
  for (const name of ["left_ear", "right_ear"])
    if (kp[name]) circle(...kp[name], r * 0.8);
  if (kp.left_eye && kp.right_eye) {
    const eyeGap = dist(...kp.left_eye, ...kp.right_eye);
    const [cx, cy] = kp.nose || [
      (kp.left_eye[0] + kp.right_eye[0]) / 2,
      (kp.left_eye[1] + kp.right_eye[1]) / 2,
    ];
    drawingContext.setLineDash([3, 4]);
    stroke(...accent, 110);
    ellipse(cx, cy - eyeGap * 0.2, eyeGap * 2.6, eyeGap * 3.3);
    drawingContext.setLineDash([]);
  }
}

// Left: confidence number and bar. Right: humans count.
function drawStats(u) {
  const { x, y, w, h } = view;
  const gap = 5 * u;
  const colW = min(x - gap - 3 * u, 22 * u);
  if (colW < 10 * u) return; // window too narrow for side panels
  const cy = y + h / 2;
  const ready = state !== "LOADING";
  const labelOpts = {
    size: 1.2 * u,
    weight: 400,
    spacing: 0.35 * u,
    color: "rgba(255,255,255,0.55)",
  };
  const numOpts = { size: 8.5 * u, weight: 300, color: "#fff" };

  // Confidence
  const lx = x - gap - colW;
  type("HUMAN CONFIDENCE", lx, cy - 6.5 * u, labelOpts);
  const num = ready ? str(round(topScore * 100)) : "--";
  const nw = type(num, lx - 0.4 * u, cy, numOpts);
  if (ready)
    type("%", lx + nw, cy + 1.5 * u, {
      ...numOpts,
      size: 3.4 * u,
      color: "rgba(255,255,255,0.6)",
    });

  const by = cy + 6 * u;
  noStroke();
  fill(255, 30);
  rect(lx, by, colW, 0.3 * u);
  fill(...accent);
  rect(lx, by, colW * constrain(ready ? shownScore : 0, 0, 1), 0.3 * u);
  fill(255, 90);
  for (const t of [LOW, HIGH])
    rect(lx + colW * t - 0.5, by - 0.6 * u, 1, 1.5 * u);
  type("0", lx, by + 1.8 * u, {
    ...labelOpts,
    size: 1 * u,
    spacing: 0,
    color: "rgba(255,255,255,0.35)",
  });
  type("100", lx + colW, by + 1.8 * u, {
    ...labelOpts,
    size: 1 * u,
    spacing: 0,
    align: "right",
    color: "rgba(255,255,255,0.35)",
  });

  // Humans count
  const rx = x + w + gap;
  const count = people.filter((p) => p.score >= LOW).length;
  type("HUMANS DETECTED", rx, cy - 6.5 * u, labelOpts);
  type(ready ? str(count) : "--", rx - 0.4 * u, cy, numOpts);

  // One segment per detected human, matching the confidence bar
  const segs = max(6, count);
  const segGap = 0.5 * u;
  const segW = (colW - segGap * (segs - 1)) / segs;
  noStroke();
  for (let i = 0; i < segs; i++) {
    if (ready && i < count) fill(...accent);
    else fill(255, 30);
    rect(rx + i * (segW + segGap), by, segW, 0.3 * u);
  }
}

// Small human outline: each region lights up in the state colour when bodyPose sees it
function drawFigurePanel(u) {
  const seen = {};
  for (const k of keypoints) if (k.confidence >= KP_MIN) seen[k.name] = true;
  const lit = {};
  for (const part of PARTS)
    lit[part.label] = state !== "LOADING" && part.names.some((n) => seen[n]);

  const H = min(height * 0.135, 9 * u);
  const textOpts = { size: 1.3 * u, weight: 400, spacing: 0.3 * u };
  const col1 = typeWidth("HANDS", textOpts) + 2.5 * u;
  const listW = col1 + 1.6 * u + typeWidth("HIDDEN", textOpts);
  const figW = H * 0.55;
  const gap = 5 * u;
  const x0 = (width - (figW + gap + listW)) / 2;
  const fx = x0 + figW / 2;
  const fy = height * 0.965 - H;

  drawFigure(fx, fy, H, lit, u);

  // Status list
  const rowH = H / 5;
  const ly = fy + H / 2 - rowH * 1.5;
  const lx = x0 + figW + gap;
  PARTS.forEach((part, i) => {
    const y = ly + i * rowH;
    const on = lit[part.label];
    const c = on ? css(accent) : "rgba(255,255,255,0.35)";
    type(part.label.toUpperCase(), lx, y, { ...textOpts, color: c });
    const mx = lx + col1 + 0.45 * u;
    if (on) {
      noStroke();
      fill(...accent);
    } else {
      noFill();
      stroke(255, 90);
      strokeWeight(1);
    }
    circle(mx, y, 0.9 * u);
    type(on ? "SEEN" : "HIDDEN", mx + 1.15 * u, y, { ...textOpts, color: c });
  });
}

function drawFigure(cx, top, H, lit, u) {
  const sw = max(1.5, 0.12 * u);
  const region = (isLit, drawFn) => {
    if (isLit) {
      glow(css(accent, 0.8), 1.2 * u);
      stroke(...accent);
      fill(...accent, 45);
      strokeWeight(sw * 1.3);
    } else {
      stroke(255, 70);
      noFill();
      strokeWeight(sw);
    }
    drawFn();
    glow("transparent", 0);
  };
  const X = (f) => cx + f * H;
  const Y = (f) => top + f * H;

  // Arms are not a region: always a neutral outline
  stroke(255, 70);
  strokeWeight(sw);
  noFill();
  for (const d of [-1, 1]) {
    line(X(d * 0.14), Y(0.25), X(d * 0.21), Y(0.4));
    line(X(d * 0.21), Y(0.4), X(d * 0.24), Y(0.52));
  }

  region(lit.Face, () => circle(X(0), Y(0.09), 0.17 * H));
  region(lit.Body, () => {
    line(X(0), Y(0.175), X(0), Y(0.22));
    beginShape();
    vertex(X(-0.14), Y(0.22));
    vertex(X(0.14), Y(0.22));
    vertex(X(0.1), Y(0.55));
    vertex(X(-0.1), Y(0.55));
    endShape(CLOSE);
  });
  region(lit.Hands, () => {
    circle(X(-0.245), Y(0.56), 0.075 * H);
    circle(X(0.245), Y(0.56), 0.075 * H);
  });
  region(lit.Legs, () => {
    noFill();
    for (const d of [-1, 1]) {
      line(X(d * 0.06), Y(0.55), X(d * 0.09), Y(0.77));
      line(X(d * 0.09), Y(0.77), X(d * 0.1), Y(0.98));
    }
  });
}

// ---------- Text helpers (p5 1.x can't set font weight or letter spacing) ----------
function css(c, alpha = 1) {
  return `rgba(${round(c[0])},${round(c[1])},${round(c[2])},${alpha})`;
}

function typeWidth(str, { font = MONO, size, weight = 400, spacing = 0 }) {
  drawingContext.font = `${weight} ${size}px ${font}`;
  if (!spacing) return drawingContext.measureText(str).width;
  let w = 0;
  for (const ch of str) w += drawingContext.measureText(ch).width;
  return w + spacing * (str.length - 1);
}

// Shrinks a size so the string fits in maxW pixels
function fitType(str, size, maxW, opts) {
  const w = typeWidth(str, { ...opts, size });
  return w > maxW ? (size * maxW) / w : size;
}

// Draws text and returns its width
function type(str, x, y, opts) {
  const {
    font = MONO,
    size,
    weight = 400,
    spacing = 0,
    align = "left",
    color = "#fff",
    glow = 0,
  } = opts;
  const ctx = drawingContext;
  const w = typeWidth(str, opts);
  ctx.save();
  ctx.font = `${weight} ${size}px ${font}`;
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  ctx.fillStyle = color;
  if (glow) {
    ctx.shadowColor = color;
    ctx.shadowBlur = glow;
  }
  let cx = align === "center" ? x - w / 2 : align === "right" ? x - w : x;
  if (!spacing) {
    ctx.fillText(str, cx, y);
  } else {
    for (const ch of str) {
      ctx.fillText(ch, cx, y);
      cx += ctx.measureText(ch).width + spacing;
    }
  }
  ctx.restore();
  return w;
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}
