let note = [], duration = 1;

const cv = document.getElementById("roll")
const g = cv.getContext("2d")
const player = document.getElementById("player")
const statusEl = document.getElementById("status")

document.getElementById("go").onclick = async () => {
    const file = document.getElementById("file").files[0]
    if (!file) { statusEl.textContent = "pick a file first."; return;}

    player.src = URL.createObjectURL(file)
    statusEl.textContent = "Analyzing..."

    const fd = new FormData();
    fd.append("file", file)

    try{
        const r = await fetch("/transcribe", { method: "POST", body: fd});
        const d = await r.json();
        if (d.error) { statusEl.textContent = "Error: " + d.error; return;}
        notes = d.notes
        duration = d.duration
        statusEl.textContent = "Found " + notes.length + "notes. Press play !";
        buildlist();
        draw(0);

    } catch (e){
        statusEl.textContent = "Could not reach the server";
    }
};

function buildlist() {
    const list = document.getElementById("list");
    list.innerHTML = "";
    notes.forEach((n,i) => {
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
    const lo = Math.min(...notes.map(n => n.midi)) - 1;
    const hi = Math.max(...notes.map(n => n.midi)) + 1;
    const rowH = cv.height / (hi - lo + 1);
    g.font = "11px sans-serif";
    
    notes.forEach((n, i) => {
        const x = n.start / duration * cv.width;
        const w = Math.max(3, (n.end - n.start) / duration * cv.width)
        const y = cv.height - (n.midi - lo + 1) * rowH;
        const active = t >= n.start && t < n.end;
        g.fillStyle = active ? "#ffb84d" : "#4da3ff";
        g.fillRect(x, y, w, rowH - 2);
        if (w > 22){
            g.fillStyle = "#000";
            g.fillText(n.name,  x + 3, y + rowH - 5)
        }

        const chip = document.getElementById("n" + i);
        if (chip) chip.classList.toggle("on", active);
    });

    g.fillStyle = "#fff";
    g.fillRect(t / duration * cv.width, 0.2, cv.height);
}

function tick() {
    draw(player.currentTime);
    requestAnimationFrame(tick);
}
tick();