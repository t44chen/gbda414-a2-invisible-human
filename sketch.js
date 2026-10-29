// Invisible Human: GBDA 414 Group 1
// COCO-SSD scores how sure the AI is that you're a person:
// red = HUMAN DETECTED, amber = UNSURE, green = UNDETECTED.
// ml5 bodyPose (MoveNet) shows which body parts the AI can still see.

// ---------- Tunable settings ----------
const HIGH = 0.70;            // score at or above this -> DETECTED (red)
const LOW = 0.30;             // score below this -> UNDETECTED (green)
const MARGIN = 0.05;          // hysteresis: score must move back this far to leave a state
const SMOOTHING = 0.5;        // 0..1, higher = faster score changes
const KP_MIN = 0.3;           // keypoint confidence that counts as "seen"
const POSE_EVERY = 2;         // run bodyPose once every N COCO detections
const DETECT_GAP = 30;        // ms pause between detections so drawing stays smooth
const SHOW_DOTS = true;       // draw keypoint dots over the video

const PARTS = [
  { label: "Face", names: ["nose", "left_eye", "right_eye", "left_ear", "right_ear"] },
  { label: "Hands", names: ["left_wrist", "right_wrist"] },
  { label: "Body", names: ["left_shoulder", "right_shoulder", "left_hip", "right_hip"] },
  { label: "Legs", names: ["left_knee", "right_knee", "left_ankle", "right_ankle"] },
];

const STATES = {
  LOADING:    { label: "Loading AI...",  bg: [40, 40, 40],   fg: 255 },
  DETECTED:   { label: "HUMAN DETECTED", bg: [214, 40, 48],  fg: 255 },
  UNSURE:     { label: "UNSURE",         bg: [242, 169, 34], fg: 20 },
  UNDETECTED: { label: "UNDETECTED",     bg: [28, 166, 92],  fg: 255 },
};

// ---------- State ----------
let video, cocoModel, bodyPose;
let people = [];              // COCO "person" boxes {x, y, w, h, score} in video pixels
let keypoints = [];           // MoveNet keypoints {x, y, confidence, name} in video pixels
let topScore = 0;             // smoothed best person score
let state = "LOADING";
let tick = 0;

function setup() {
  createCanvas(windowWidth, windowHeight);
  video = createCapture({ video: { width: 640, height: 480 }, audio: false });
  video.hide();
  // Match the element size to the real camera size so both models use the same pixel space
  video.elt.addEventListener("loadedmetadata", () => {
    video.size(video.elt.videoWidth, video.elt.videoHeight);
  });

  textFont("Helvetica");
  // WebGL is the steadiest backend for COCO-SSD (on WebGPU its box filtering blocks the page)
  tf.setBackend("webgl").then(() => Promise.all([
    cocoSsd.load(),
    new Promise((resolve) => {
      bodyPose = ml5.bodyPose("MoveNet", { modelType: "SINGLEPOSE_LIGHTNING" }, resolve);
    }),
  ])).then(([m]) => {
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
        .map((p) => ({ x: p.bbox[0], y: p.bbox[1], w: p.bbox[2], h: p.bbox[3], score: p.score }));
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

function draw() {
  const st = STATES[state];
  background(...st.bg);
  const fg = st.fg;
  const u = min(width, height * 16 / 9) / 100;   // 1% of the largest 16:9 area that fits

  // Video area: centred, keeps the camera's aspect ratio
  const camW = video.elt.videoWidth || 640;
  const camH = video.elt.videoHeight || 480;
  let vh = height * 0.52;
  let vw = vh * camW / camH;
  if (vw > width * 0.6) {
    vw = width * 0.6;
    vh = vw * camH / camW;
  }
  const vx = (width - vw) / 2;
  const vy = height * 0.24 + (height * 0.52 - vh) / 2;
  const s = vw / camW;                           // video pixels -> screen pixels

  // Mirrored webcam
  push();
  translate(vx + vw, vy);
  scale(-1, 1);
  image(video, 0, 0, vw, vh);
  pop();
  noFill();
  stroke(fg);
  strokeWeight(0.4 * u);
  rect(vx, vy, vw, vh);

  if (state !== "LOADING") {
    if (SHOW_DOTS) drawKeypoints(vx, vy, vw, s, u);
    drawPeople(vx, vy, vw, s, u);
  }

  // Question and state label
  noStroke();
  fill(fg);
  textAlign(CENTER, CENTER);
  textStyle(NORMAL);
  textSize(fitSize("Can you make the AI stop seeing you as human?", 3.4 * u, width * 0.92));
  text("Can you make the AI stop seeing you as human?", width / 2, height * 0.07);
  drawStateLabel(st.label, height * 0.16, fg, u);

  // Side panels: confidence on the left, number of humans on the right
  const count = people.filter((p) => p.score >= LOW).length;
  const ready = state !== "LOADING";
  drawStat(ready ? round(topScore * 100) + "%" : "--", "Human confidence", vx / 2, vy + vh / 2, vx * 0.85, fg, u);
  drawStat(ready ? str(count) : "--", "Humans detected", vx + vw + vx / 2, vy + vh / 2, vx * 0.85, fg, u);

  drawParts(height * 0.875, fg, st.bg, u);
}

// Box around every person, mirrored to line up with the video
function drawPeople(vx, vy, vw, s, u) {
  textAlign(LEFT, CENTER);
  textStyle(BOLD);
  textSize(1.8 * u);
  for (const p of people) {
    if (p.score < LOW) continue;
    const x = vx + vw - (p.x + p.w) * s;
    const y = vy + p.y * s;
    const c = p.score >= HIGH ? STATES.DETECTED : STATES.UNSURE;
    noFill();
    stroke(...c.bg);
    strokeWeight(0.5 * u);
    rect(x, y, p.w * s, p.h * s);

    const label = "Human " + round(p.score * 100) + "%";
    const tw = textWidth(label) + 1.6 * u;
    const th = 3 * u;
    const ty = y - th >= vy ? y - th : y;   // tag above the box, or inside it at the top edge
    noStroke();
    fill(...c.bg);
    rect(x, ty, tw, th);
    fill(c.fg);
    text(label, x + 0.8 * u, ty + th / 2);
  }
}

function drawKeypoints(vx, vy, vw, s, u) {
  stroke(20);
  strokeWeight(0.25 * u);
  fill(255);
  for (const k of keypoints) {
    if (k.confidence < KP_MIN) continue;
    circle(vx + vw - k.x * s, vy + k.y * s, 1.1 * u);
  }
}

function drawStateLabel(label, y, fg, u) {
  textStyle(BOLD);
  const size = fitSize(label, 7 * u, width * 0.8);
  textSize(size);
  const iconD = size * 0.9;
  const gap = size * 0.35;
  const hasIcon = state !== "LOADING";
  const total = textWidth(label) + (hasIcon ? iconD + gap : 0);
  const x0 = (width - total) / 2;
  if (hasIcon) drawStateIcon(x0 + iconD / 2, y, iconD, fg);
  noStroke();
  fill(fg);
  textAlign(LEFT, CENTER);
  text(label, x0 + (hasIcon ? iconD + gap : 0), y);
}

// Open eye = seen, question mark = unsure, crossed-out eye = not seen
function drawStateIcon(cx, cy, d, fg) {
  push();
  translate(cx, cy);
  stroke(fg);
  strokeWeight(d * 0.09);
  noFill();
  if (state === "UNSURE") {
    circle(0, 0, d);
    noStroke();
    fill(fg);
    textAlign(CENTER, CENTER);
    textStyle(BOLD);
    textSize(d * 0.7);
    text("?", 0, d * 0.03);
  } else {
    const r = d / 2;
    beginShape();
    vertex(-r, 0);
    bezierVertex(-r * 0.5, -r * 0.8, r * 0.5, -r * 0.8, r, 0);
    bezierVertex(r * 0.5, r * 0.8, -r * 0.5, r * 0.8, -r, 0);
    endShape();
    fill(fg);
    circle(0, 0, d * 0.3);
    if (state === "UNDETECTED") line(-r * 0.8, r * 0.8, r * 0.8, -r * 0.8);
  }
  pop();
}

function drawStat(value, caption, cx, cy, maxW, fg, u) {
  if (maxW < 6 * u) return;   // window too narrow for side panels
  noStroke();
  fill(fg);
  textAlign(CENTER, CENTER);
  textStyle(BOLD);
  const big = fitSize(value, 10 * u, maxW);
  textSize(big);
  text(value, cx, cy - 1.5 * u);
  textStyle(NORMAL);
  textSize(fitSize(caption, 2.2 * u, maxW));
  text(caption, cx, cy + big / 2 + 1 * u);
}

// Row of 4 body-part icons: lit when MoveNet sees that part, dimmed when it doesn't
function drawParts(cy, fg, bg, u) {
  const seen = {};
  for (const k of keypoints) seen[k.name] = k.confidence >= KP_MIN;

  const d = 6.5 * u;
  const spacing = 13 * u;
  const x0 = width / 2 - spacing * (PARTS.length - 1) / 2;
  for (let i = 0; i < PARTS.length; i++) {
    const part = PARTS[i];
    const lit = state !== "LOADING" && part.names.some((n) => seen[n]);
    const x = x0 + i * spacing;
    const dimFg = color(fg);
    dimFg.setAlpha(90);

    strokeWeight(0.3 * u);
    if (lit) {
      noStroke();
      fill(fg);
    } else {
      stroke(dimFg);
      noFill();
    }
    circle(x, cy - u, d);
    drawPartGlyph(part.label, x, cy - u, d, lit ? color(...bg) : dimFg);

    noStroke();
    fill(lit ? fg : dimFg);
    textAlign(CENTER, CENTER);
    textStyle(lit ? BOLD : NORMAL);
    textSize(2 * u);
    text(part.label, x, cy + d / 2 + 1.2 * u);
  }
}

function drawPartGlyph(label, cx, cy, d, c) {
  push();
  translate(cx, cy);
  stroke(c);
  strokeWeight(d * 0.07);
  strokeCap(ROUND);
  noFill();
  if (label === "Face") {
    circle(0, 0, d * 0.5);
    fill(c);
    noStroke();
    circle(-d * 0.09, -d * 0.04, d * 0.07);
    circle(d * 0.09, -d * 0.04, d * 0.07);
  } else if (label === "Hands") {
    rect(-d * 0.13, -d * 0.02, d * 0.26, d * 0.2, d * 0.05);
    for (let i = 0; i < 4; i++) {
      const fx = -d * 0.1 + i * d * 0.067;
      line(fx, -d * 0.02, fx, -d * 0.2);
    }
    line(-d * 0.13, d * 0.06, -d * 0.2, -d * 0.04);
  } else if (label === "Body") {
    // Head and shoulders
    circle(0, -d * 0.12, d * 0.17);
    arc(0, d * 0.22, d * 0.44, d * 0.36, PI, TWO_PI, CHORD);
  } else if (label === "Legs") {
    line(-d * 0.06, -d * 0.2, d * 0.06, -d * 0.2);
    line(-d * 0.06, -d * 0.2, -d * 0.11, d * 0.18);
    line(d * 0.06, -d * 0.2, d * 0.11, d * 0.18);
    line(-d * 0.11, d * 0.18, -d * 0.19, d * 0.18);
    line(d * 0.11, d * 0.18, d * 0.19, d * 0.18);
  }
  pop();
}

// Shrinks a text size so the string fits in maxW pixels
function fitSize(str, size, maxW) {
  textSize(size);
  const w = textWidth(str);
  return w > maxW ? size * maxW / w : size;
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}
