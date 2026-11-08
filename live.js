const NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
const LOW = 36, HIGH = 84;      // C2 to C6, the cello range
const WINDOW_SEC = 8;           // seconds shown on the roll

const micBtn = document.getElementById("micBtn");
const liveNote = document.getElementById("liveNote");
const liveInfo = document.getElementById("liveInfo");
const lv = document.getElementById("liveRoll");
const lg = lv.getContext("2d");

let actx, analyser, buf, stream, timer;
let running = false;
let liveNotes = [];
let current = null;
let candidate = null, candCount = 0, lastVoiced = 0;


function detectPitch(data, sr) {
    let rms = 0;
  for (const v of data) rms += v * v;
    if (Math.sqrt(rms / data.length) < 0.01) return -1;

    const W = data.length >> 1;
    const minTau = Math.floor(sr / 1000);
    const maxTau = Math.min(Math.floor(sr / 60), W);
    const d = new Float32Array(maxTau);

for (let tau = 1; tau < maxTau; tau++) {
    let s = 0;
    for (let i = 0; i < W; i++) { const x = data[i] - data[i + tau]; s += x * x; }
    d[tau] = s;
    }
    let run = 0; d[0] = 1;
  for (let tau = 1; tau < maxTau; tau++) { run += d[tau]; d[tau] = d[tau] * tau / run; }

for (let tau = minTau; tau < maxTau - 1; tau++) {
    if (d[tau] < 0.15) {
    while (tau + 1 < maxTau && d[tau + 1] < d[tau]) tau++;
    if (tau + 1 >= maxTau) return -1;
    const a = d[tau - 1], b = d[tau], c = d[tau + 1];
      const denom = 2 * (a - 2 * b + c);
        return sr / (tau + (denom ? (a - c) / denom : 0));
    }
    }
    return -1;
}

function step() {
    const now = performance.now() / 1000;
    analyser.getFloatTimeDomainData(buf);
    const f = detectPitch(buf, actx.sampleRate);

    if (f > 0) {
    const exact = 69 + 12 * Math.log2(f / 440);
    const midi = Math.round(exact);
    const name = NOTE_NAMES[midi % 12] + (Math.floor(midi / 12) - 1);
    const cents = Math.round((exact - midi) * 100);

    liveNote.textContent = name;
    liveInfo.textContent = f.toFixed(1) + " Hz  (" + (cents >= 0 ? "+" : "") + cents + " cents)";
    lastVoiced = now;


    if (midi === candidate) candCount++; else { candidate = midi; candCount = 1; }
    if (candCount >= 3) {
    if (!current || current.midi !== midi) {
        current = { midi, name, start: now, end: now };
        liveNotes.push(current);
    } else {
        current.end = now;
    }
    }
    } else if (now - lastVoiced > 0.25) {
    current = null;
    candidate = null;
    candCount = 0;
    liveNote.textContent = "-";
    liveInfo.textContent = "Play or hum a note near the mic.";
    }
    drawLive(now);
}

function drawLive(now) {
    lg.clearRect(0, 0, lv.width, lv.height);
    const rowH = lv.height / (HIGH - LOW + 1);
    const t0 = now - WINDOW_SEC;
    lg.font = "11px sans-serif";

    liveNotes = liveNotes.filter(n => n.end > t0);
    liveNotes.forEach(n => {
    const x = (n.start - t0) / WINDOW_SEC * lv.width;
    const w = Math.max(3, (n.end - n.start) / WINDOW_SEC * lv.width);
    const y = lv.height - (n.midi - LOW + 1) * rowH;
    lg.fillStyle = n === current ? "#ffb84d" : "#4da3ff";
    lg.fillRect(Math.max(0, x), y, w, rowH);
    if (w > 22) { lg.fillStyle = "#000"; lg.fillText(n.name, x + 3, y + rowH - 2); }
    });
}

micBtn.onclick = async () => {
    if (running) {
    running = false;
    clearInterval(timer);
    stream.getTracks().forEach(t => t.stop());
    actx.close();
    micBtn.textContent = "Start microphone";
    micBtn.classList.remove("live");
    return;
    }
    try {
    stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
    });
    } catch (e) {
    liveInfo.textContent = "Microphone blocked. Click the lock icon in the address bar and allow it.";
    return;
    }
    actx = new AudioContext();
    analyser = actx.createAnalyser();
    analyser.fftSize = 4096;
    buf = new Float32Array(analyser.fftSize);
    actx.createMediaStreamSource(stream).connect(analyser);

    liveNotes = []; current = null; candidate = null; candCount = 0;
    running = true;
    micBtn.textContent = "Stop microphone";
    micBtn.classList.add("live");
    timer = setInterval(step, 50);
};