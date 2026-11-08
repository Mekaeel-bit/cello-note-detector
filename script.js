let notes = [], duration = 1;
let pollTimer = null;
let osmd = null;

const cv = document.getElementById("roll");
const g = cv.getContext("2d");
const player = document.getElementById("player");
const statusEl = document.getElementById("status");

const videoStatus = document.getElementById("videoStatus");
const videoPreview = document.getElementById("videoPreview");
const videoActions = document.getElementById("videoActions");
const scoreMeta = document.getElementById("scoreMeta");
const overlayRow = document.getElementById("overlayRow");
const osmdContainer = document.getElementById("osmdContainer");
const musescoreHint = document.getElementById("musescoreHint");

document.querySelectorAll(".tab").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b === btn));
    document.getElementById("video").hidden = btn.dataset.tab !== "video";
    document.getElementById("live").hidden = btn.dataset.tab !== "live";
    document.getElementById("upload").hidden = btn.dataset.tab !== "upload";
  };
});

document.getElementById("videoFile").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  videoPreview.src = URL.createObjectURL(file);
  videoPreview.hidden = false;
  videoStatus.textContent = `Ready: ${file.name} (${(file.size / 1e6).toFixed(1)} MB)`;
});

document.getElementById("videoGo").onclick = async () => {
  const file = document.getElementById("videoFile").files[0];
  if (!file) {
    videoStatus.textContent = "Pick a video first (mp4 / webm / mov).";
    return;
  }
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  videoActions.hidden = true;
  scoreMeta.hidden = true;
  overlayRow.hidden = true;
  overlayRow.innerHTML = "";
  osmdContainer.hidden = true;
  osmdContainer.innerHTML = "";
  musescoreHint.hidden = true;

  const fd = new FormData();
  fd.append("file", file);
  if (document.getElementById("fastDemo").checked) {
    fd.append("max_frames", "180");
  }
  fd.append("overlay", document.getElementById("wantOverlay").checked ? "1" : "0");
  fd.append("quantize", "coarse");

  videoStatus.textContent = "Uploading & starting AV fuse…";
  document.getElementById("videoGo").disabled = true;

  try {
    const r = await fetch("/api/video-score", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok || !d.ok) {
      videoStatus.textContent = "Error: " + (d.error || r.statusText);
      document.getElementById("videoGo").disabled = false;
      return;
    }
    videoStatus.textContent = "Job " + d.job_id + " — processing (first MediaPipe download can take a bit)…";
    pollTimer = setInterval(() => pollJob(d.job_id), 1500);
  } catch (e) {
    videoStatus.textContent = "Could not reach the server. Start it with: python -m cello_note_detector.web";
    document.getElementById("videoGo").disabled = false;
  }
};

async function pollJob(jobId) {
  try {
    const r = await fetch("/api/jobs/" + jobId + "/status");
    const d = await r.json();
    if (!d.ok) {
      videoStatus.textContent = "Error: " + (d.error || "unknown");
      stopPoll();
      return;
    }
    videoStatus.textContent = d.message || d.status;
    if (d.status === "done" && d.result) {
      stopPoll();
      showResult(d.result);
    } else if (d.status === "error") {
      stopPoll();
      videoStatus.textContent = "Error: " + (d.error || d.message || "pipeline failed");
    }
  } catch (e) {
    videoStatus.textContent = "Lost connection while polling.";
    stopPoll();
  }
}

function stopPoll() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
  document.getElementById("videoGo").disabled = false;
}

function showResult(result) {
  videoActions.hidden = false;
  const dl = document.getElementById("dlMusicxml");
  dl.href = result.musicxml_url;
  dl.download = (result.stem || "score") + "_fused.musicxml";
  const dj = document.getElementById("dlScoreJson");
  if (result.score_json_url) {
    dj.href = result.score_json_url;
    dj.hidden = false;
  } else {
    dj.hidden = true;
  }

  scoreMeta.hidden = false;
  scoreMeta.textContent =
    `${result.num_notes || 0} notes · ${result.num_rests || 0} rests · ` +
    `key ${result.key || "?"} · ${result.tempo_bpm != null ? Math.round(result.tempo_bpm) + " BPM" : "tempo ?"} · ` +
    `source: fused transcription`;

  if (result.overlay_urls && result.overlay_urls.length) {
    overlayRow.hidden = false;
    overlayRow.innerHTML = "";
    result.overlay_urls.forEach((url) => {
      const img = document.createElement("img");
      img.src = url;
      img.alt = "fused overlay";
      overlayRow.appendChild(img);
    });
  }

  musescoreHint.hidden = false;
  renderOsmd(result.musicxml_url);
}

async function renderOsmd(url) {
  if (typeof opensheetmusicdisplay === "undefined") {
    videoStatus.textContent += " (OSMD CDN unavailable — use Download MusicXML)";
    return;
  }
  osmdContainer.hidden = false;
  osmdContainer.innerHTML = "";
  try {
    if (!osmd) {
      osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(osmdContainer, {
        autoResize: true,
        backend: "svg",
        drawTitle: true,
      });
    }
    const xml = await fetch(url).then((r) => r.text());
    await osmd.load(xml);
    osmd.render();
  } catch (e) {
    osmdContainer.hidden = true;
    videoStatus.textContent += " — score viewer failed; download MusicXML instead.";
  }
}

/* ---- existing audio-upload tab (expects optional /transcribe) ---- */

document.getElementById("go").onclick = async () => {
  const file = document.getElementById("file").files[0];
  if (!file) { statusEl.textContent = "Pick a file first."; return; }

  player.src = URL.createObjectURL(file);
  statusEl.textContent = "Analyzing… this can take 10–30 seconds.";

  const fd = new FormData();
  fd.append("file", file);

  try {
    const r = await fetch("/transcribe", { method: "POST", body: fd });
    const d = await r.json();
    if (d.error) { statusEl.textContent = "Error: " + d.error; return; }
    notes = d.notes;
    duration = d.duration;
    const cut = d.total > d.duration + 1 ? ` (analyzed the first ${Math.round(d.duration)}s)` : "";
    statusEl.textContent = notes.length
      ? `Found ${notes.length} notes${cut}. Press play, or click the roll to jump.`
      : "No notes found. Try a cleaner, louder solo recording.";
    buildlist();
    draw(0);
  } catch (e) {
    statusEl.textContent = "Could not reach /transcribe (audio path not wired on this server). Use Video → Score.";
  }
};

cv.onclick = e => {
  if (!notes.length) return;
  const x = (e.clientX - cv.getBoundingClientRect().left) / cv.clientWidth;
  player.currentTime = x * duration;
};

function buildlist() {
  const list = document.getElementById("list");
  list.innerHTML = "";
  notes.forEach((n, i) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.id = "n" + i;
    c.textContent = n.name;
    list.appendChild(c);
  });
}

function draw(t) {
  g.clearRect(0, 0, cv.width, cv.height);
  if (!notes.length) return;
  t = Math.min(t, duration);

  const lo = Math.min(...notes.map(n => n.midi)) - 1;
  const hi = Math.max(...notes.map(n => n.midi)) + 1;
  const rowH = cv.height / (hi - lo + 1);

  g.strokeStyle = "#1f2733";
  for (let m = lo; m <= hi; m++) {
    const y = cv.height - (m - lo + 1) * rowH;
    g.beginPath(); g.moveTo(0, y); g.lineTo(cv.width, y); g.stroke();
  }

  g.font = "11px sans-serif";
  notes.forEach((n, i) => {
    const x = n.start / duration * cv.width;
    const w = Math.max(3, (n.end - n.start) / duration * cv.width);
    const y = cv.height - (n.midi - lo + 1) * rowH;
    const active = t >= n.start && t < n.end;

    g.fillStyle = active ? "#ffb84d" : "#4da3ff";
    g.fillRect(x, y + 1, w, rowH - 2);
    if (w > 24) {
      g.fillStyle = "#000";
      g.fillText(n.name, x + 3, y + rowH - 5);
    }

    const chip = document.getElementById("n" + i);
    if (chip) chip.classList.toggle("on", active);
  });

  g.fillStyle = "#fff";
  g.fillRect(t / duration * cv.width, 0, 2, cv.height);
}

function tick() {
  draw(player.currentTime);
  requestAnimationFrame(tick);
}
tick();
