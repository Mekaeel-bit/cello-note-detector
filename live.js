const NOTE_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
const LOW = 36, High = 84;
const Window_SEC = 8;

const micBtn = document.getElementById("micBtn");
const liveNote = document.getElementById("liveNote");
const liveInfo = document.getElementById("liveInfo");
const lv = document.getElementById("liveRoll");
const lg = lv.getContext("2d");

let actx, analyser, buf, stream, timer;
let running = false;
let liveNotes = [];
let current = null;
let candinate = null, candCount = 0, lastVoiced = 0;

function detectPitch(data, sr){
    let rms = 0;
    for (const v of data) rms += v * v;
    if (rms < 0.01) return -1;

    const minLag = Math.floor(sr/1000);
    const maxLag = Math.floor(sr/60);
    let zero = 0;
    for (let i = 0; i < data.length; i++) zero += data[i] * data[i];

    let bestLag = -1, best = 0;
    for(let lag = minLag; lag < maxLag; lag++){
        let c = 0;
        for (let i = 0; i < data.length - lag; i++) c += data[i] * data[i + lag];
        if (c > best) { best = c; bestLag = lag;}
    }
    if (bestLag < 0 || best / zero < 0.4) return -1;
    return sr / bestLag;
}

function step(){
    const now  = performance.now() / 1000;
    analyser.getFloatTimeDomaininDta(buf);
    const f = detectPitch(buf, actx.sampleRate);

    if (f > 0){
        const exact = 69 + 12 * Math.log2(f / 440);
        const midi = Math.round(exact);
        const name = NOTE_names[midi % 12] + (Math.floor(midi / 12) - 1);
        const cents = Math.round((exact - midi) * 100);

        liveNote.textContent = name;
        liveInfo.textContext = f.toFixed(1) + " Hz (" + (cents >= 0 ? "+" : "") + cents + " cents)";
        lastVoiced = now;

        if (midi === candinate) candCount++; else { candinate = midi; candCount = 1;}
        if (candCount >= 3){
            if (!current || current.midi !== midi){
                current = {midi, name, start: now, end: now};
                liveNotes.push(current);
            } else {
                current.end = now;
            }
        }
    } else if (current && now - lastVoiced > 0.25){
        current = null;
        candinate = null;
        candCount = 0;
        liveNote.textContent = "-";
        LiveInfo.textContent = "";
    }
    drawLive(now);
}

function drawLive(now){
    lg.clearRect(0, 0, lv.clientWidth, lv.height);
    const rowH = lv.height / (HIGH - LOW + 1);
    const t0 = now - Window_SEC;
    lg.font = "11px sans-serif";

    liveNotes =  liveNotes.filter(n => n.end > t0);
    liveNotes,forEach(n => {
        const x = (n.start = t0) / WINDOW_SEC * lv.clientWidth;
        const w = Math.max(3,(n.end - n.start) / WINDOW_SEC * lv.width);
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
    timer = setInterval(step, 50);
};