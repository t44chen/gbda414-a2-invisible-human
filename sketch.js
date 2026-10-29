// Invisible Human: GBDA 414 Group 1
// Red = AI sees a person. Green = in frame but AI can't tell you're human.
// Grey = nobody in frame (walking away no longer counts as winning).

// ---------- Tunable settings ----------
const HIGH = 0.70;            // person score above this -> RED
const LOW = 0.30;             // person score below this (while present) -> GREEN
const PRESENCE_MIN = 0.06;    // fraction of changed pixels that means "someone is in frame"
const PIXEL_DIFF = 40;        // how different a pixel must be from the empty background
const EMPTY_DELAY = 600;      // ms the frame must stay empty before showing grey
const SMOOTHING = 0.3;        // 0..1, higher = faster score changes

// ---------- State ----------
let video, small, model;
let personScore = 0;          // smoothed "person" confidence
let presence = 0;             // fraction of frame that differs from background
let bgPixels = null;          // snapshot of the empty scene
let state = "LOADING";        // LOADING, CALIBRATE, EMPTY, RED, GREEN
let calibrateEnd = 0;
let lastSeen = 0;             // last time someone was present
let greenStart = 0;
let bestTime = 0;

function setup() {
  createCanvas(windowWidth, windowHeight);
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  // Small copy of the video for cheap presence detection
  small = createGraphics(64, 48);
  small.pixelDensity(1);

  textFont("Helvetica");
  cocoSsd.load().then((m) => {
    model = m;
    startCalibration();
    detectLoop();
  });
}

// Asks visitors to clear the frame, then saves the empty scene
function startCalibration() {
  state = "CALIBRATE";
  bgPixels = null;
  calibrateEnd = millis() + 3000;
}

async function detectLoop() {
  if (model && video.elt.readyState >= 2) {
    const preds = await model.detect(video.elt, 10, 0.05);
    let best = 0;
    for (const p of preds) {
      if (p.class === "person" && p.score > best) best = p.score;
    }
    personScore = lerp(personScore, best, SMOOTHING);
  }
  setTimeout(detectLoop, 100);
}

function measurePresence() {
  small.image(video, 0, 0, small.width, small.height);
  small.loadPixels();
  const px = small.pixels;

  if (state === "CALIBRATE") {
    // Don't snapshot until the camera is actually streaming (permission prompt may still be open)
    if (video.elt.readyState < 2) {
      calibrateEnd = millis() + 3000;
      return;
    }
    if (millis() >= calibrateEnd) {
      bgPixels = Uint8ClampedArray.from(px);
      state = "EMPTY";
    }
    return;
  }
  if (!bgPixels) return;

  let changed = 0;
  const total = px.length / 4;
  for (let i = 0; i < px.length; i += 4) {
    const d = abs(px[i] - bgPixels[i]) + abs(px[i + 1] - bgPixels[i + 1]) + abs(px[i + 2] - bgPixels[i + 2]);
    if (d / 3 > PIXEL_DIFF) changed++;
  }
  presence = changed / total;
}

function updateState() {
  if (state === "LOADING" || state === "CALIBRATE") return;

  const now = millis();
  const present = presence > PRESENCE_MIN || personScore > LOW;
  if (present) lastSeen = now;

  let next = state;
  if (!present && now - lastSeen > EMPTY_DELAY) {
    next = "EMPTY";
  } else if (personScore >= HIGH) {
    next = "RED";
  } else if (present && personScore < LOW) {
    next = "GREEN";
  } else if (state === "EMPTY" && present) {
    next = "RED"; // someone just stepped in but the AI hasn't decided yet
  }
  // Between LOW and HIGH we keep the previous colour so it doesn't flicker

  if (next === "GREEN" && state !== "GREEN") greenStart = now;
  if (state === "GREEN" && next !== "GREEN") {
    bestTime = max(bestTime, (now - greenStart) / 1000);
  }
  state = next;
}

function draw() {
  if (state !== "LOADING") measurePresence();
  updateState();

  // Background colour by state
  if (state === "RED") background(220, 20, 30);
  else if (state === "GREEN") background(20, 180, 80);
  else background(40);

  // Mirrored webcam in the middle
  const vw = min(width * 0.6, (height * 0.6) * (4 / 3));
  const vh = vw * 0.75;
  const vx = (width - vw) / 2;
  const vy = (height - vh) / 2;
  push();
  translate(vx + vw, vy);
  scale(-1, 1);
  image(video, 0, 0, vw, vh);
  pop();
  noFill();
  stroke(255);
  strokeWeight(4);
  rect(vx, vy, vw, vh);

  // Text
  noStroke();
  fill(255);
  textAlign(CENTER, CENTER);
  const big = min(width, height) * 0.07;
  const mid = big * 0.45;

  let headline = "";
  let sub = "";
  if (state === "LOADING") {
    headline = "Loading AI...";
  } else if (state === "CALIBRATE") {
    headline = "Please clear the frame";
    sub = "Scanning empty room in " + ceil((calibrateEnd - millis()) / 1000);
  } else if (state === "EMPTY") {
    headline = "STAY IN THE FRAME";
    sub = "Step into the Play Zone. Leaving doesn't count.";
  } else if (state === "RED") {
    headline = "HUMAN DETECTED";
    sub = "Can you make the AI stop seeing you?";
  } else if (state === "GREEN") {
    headline = "UNDETECTED";
    sub = nf((millis() - greenStart) / 1000, 1, 1) + "s  Hold it!";
  }

  textSize(big);
  text(headline, width / 2, vy / 2);
  textSize(mid);
  text(sub, width / 2, vy + vh + (height - vy - vh) / 2 - mid * 0.6);

  // Small readout: score, presence, best time
  textSize(mid * 0.6);
  text(
    "Person confidence: " + nf(personScore, 1, 2) +
    "   |   Best invisible time: " + nf(bestTime, 1, 1) + "s",
    width / 2,
    vy + vh + (height - vy - vh) / 2 + mid * 0.7
  );
}

// Press B to re-scan the empty room (do this whenever lighting changes)
// Press D to toggle a debug readout of presence in the browser console
function keyPressed() {
  if (key === "b" || key === "B") startCalibration();
  if (key === "d" || key === "D") console.log("presence:", presence.toFixed(3), "score:", personScore.toFixed(2));
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}
