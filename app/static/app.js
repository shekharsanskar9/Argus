// Argus front-end: uploads a file, reads the NDJSON event stream from
// /api/analyze and renders each pipeline stage as it happens.

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const PACE_MS = 900;

let file = null;
let ready = false;
let busy = false;
let plan = [];

// ---------------------------------------------------------------- health

async function checkHealth() {
  const el = $("status");
  try {
    const h = await (await fetch("/api/health")).json();
    ready = h.ready;
    el.className = "status " + (h.ready ? "ready" : h.loading ? "loading" : "error");
    $("status-text").textContent = h.ready
      ? "Models ready" + (h.seatbelt_model ? " · helmet + seat belt" : " · helmet")
      : h.loading ? "Loading models..." : "Model error";
    if (h.error) showNotice(h.error, true);
    if (h.limits) {
      $("limits").textContent = `Videos: the first ${h.limits.max_seconds}s are analysed at about ` +
        `${h.limits.process_fps} frames per second. Images run in a few seconds.`;
    }
    if (h.loading) setTimeout(checkHealth, 3000);
  } catch {
    el.className = "status error";
    $("status-text").textContent = "Server unreachable";
    setTimeout(checkHealth, 5000);
  }
  updateRunButton();
}

function updateRunButton() {
  $("run").disabled = !file || !ready || busy;
  $("run").textContent = busy ? "Analysing..." : "Analyse";
}

function showNotice(text, isError = false) {
  const n = $("notice");
  n.hidden = !text;
  n.textContent = text || "";
  n.classList.toggle("error", isError);
}

// ---------------------------------------------------------------- file picking

function pickFile(f) {
  if (!f || busy) return;
  const isVideo = f.type.startsWith("video/") || /\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(f.name);
  const isImage = f.type.startsWith("image/");
  if (!isVideo && !isImage) {
    showNotice("Please choose an image or a video file.", true);
    return;
  }
  file = f;
  showNotice(null);
  const url = URL.createObjectURL(f);
  const prev = $("drop-preview");
  prev.innerHTML = isVideo
    ? `<video src="${url}" muted playsinline controls></video>`
    : `<img src="${url}" alt="Selected image">`;
  prev.hidden = false;
  $("drop-empty").hidden = true;
  $("file-meta").textContent = `${f.name} · ${(f.size / 1048576).toFixed(1)} MB · ${isVideo ? "video" : "image"}`;
  updateRunButton();
}

$("file").addEventListener("change", (e) => pickFile(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => {
  e.preventDefault();
  drop.classList.add("over");
}));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => {
  e.preventDefault();
  drop.classList.remove("over");
}));
drop.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));

// ---------------------------------------------------------------- pipeline view

function buildPlan(ev) {
  plan = ev.steps;
  $("rail").innerHTML = plan.map((s, i) => `
    <li id="rail-${s.id}" class="pending" data-step="${s.id}">
      <span class="badge">${i + 1}</span><span>${esc(s.title)}</span><span class="ms"></span>
    </li>`).join("");
  $("cards").innerHTML = plan.map((s, i) => `
    <article class="card pending" id="card-${s.id}">
      <div class="card-head">
        <span class="badge">${i + 1}</span>
        <div><h3>${esc(s.title)}</h3><p>${esc(s.desc)}</p></div>
        <span class="ms"></span>
      </div>
      <div class="card-body"></div>
    </article>`).join("");
  $("rail").querySelectorAll("li").forEach((li) => li.addEventListener("click", () =>
    $("card-" + li.dataset.step).scrollIntoView({ behavior: "smooth", block: "start" })));
  $("pipeline").hidden = false;
  updateOverall();
}

function setStatus(id, status) {
  for (const el of [$("rail-" + id), $("card-" + id)]) {
    if (!el) continue;
    el.classList.remove("pending", "running", "done", "skipped", "error");
    el.classList.add(status);
    const badge = el.querySelector(".badge");
    const n = plan.findIndex((s) => s.id === id) + 1;
    badge.textContent = status === "done" ? "✓" : status === "skipped" ? "–" : status === "error" ? "!" : n;
  }
  updateOverall();
}

function updateOverall() {
  const finished = plan.filter((s) => {
    const c = $("card-" + s.id)?.classList;
    return c && (c.contains("done") || c.contains("skipped"));
  }).length;
  $("overall-bar").style.width = plan.length ? `${(100 * finished) / plan.length}%` : "0";
  $("overall-text").textContent = `${finished} of ${plan.length} stages complete`;
}

function metricsHtml(m) {
  if (!m || !Object.keys(m).length) return "";
  return `<div class="metrics">${Object.entries(m).map(([k, v]) =>
    `<div class="metric"><b>${esc(v)}</b>${esc(k)}</div>`).join("")}</div>`;
}

function imagesHtml(images) {
  if (!images || !images.length) return "";
  const wide = images.length === 1 ? " wide" : "";
  return `<div class="shots${wide}">${images.map((im) =>
    `<figure><img src="${im.src}" alt="${esc(im.label)}" loading="lazy"><figcaption>${esc(im.label)}</figcaption></figure>`
  ).join("")}</div>`;
}

function statusPill(v) {
  const s = String(v);
  if (/^(auto-confirmed|confirmed|yes|readable)$/.test(s)) return `<span class="pill ok">${esc(s)}</span>`;
  if (/^(rejected|no)$/.test(s)) return `<span class="pill bad">${esc(s)}</span>`;
  if (/review|flagged/.test(s)) return `<span class="pill warn">${esc(s)}</span>`;
  return esc(s);
}

function tableHtml(t) {
  if (!t || !t.rows.length) return "";
  const plateCols = new Set(t.columns.map((c, i) => /text|voted|corrected|raw|single/.test(c) ? i : -1));
  return `<div class="table-wrap"><table><thead><tr>${t.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
    <tbody>${t.rows.map((r) => `<tr>${r.map((v, i) =>
      `<td class="${plateCols.has(i) ? "plate" : ""}">${statusPill(v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function onStep(ev) {
  const card = $("card-" + ev.id);
  if (!card) return;
  const body = card.querySelector(".card-body");
  setStatus(ev.id, ev.status);
  if (ev.status === "running") {
    body.innerHTML = ev.text ? `<div class="progress"><span>${esc(ev.text)}</span></div>` : "";
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return;
  }
  const ms = ev.ms != null ? (ev.ms >= 1000 ? `${(ev.ms / 1000).toFixed(1)}s` : `${ev.ms}ms`) : "";
  card.querySelector(".ms").textContent = ev.status === "skipped" ? "" : ms;
  $("rail-" + ev.id).querySelector(".ms").textContent = ev.status === "skipped" ? "" : ms;
  body.innerHTML = `<div class="summary">${esc(ev.summary)}</div>` +
    metricsHtml(ev.metrics) + imagesHtml(ev.images) + tableHtml(ev.table);
}

function onProgress(ev) {
  const body = $("card-" + ev.id)?.querySelector(".card-body");
  if (!body) return;
  let p = body.querySelector(".progress");
  if (!p || !p.querySelector(".bar")) {
    body.innerHTML = `<div class="progress"><span class="ptext"></span><div class="bar"><i></i></div></div>`;
    p = body.querySelector(".progress");
  }
  p.querySelector(".ptext").textContent = ev.text || "";
  p.querySelector(".bar i").style.width = `${ev.pct}%`;
  if (ev.preview) {
    let img = body.querySelector("img.live");
    if (!img) {
      img = document.createElement("img");
      img.className = "live";
      img.alt = "Live preview";
      body.appendChild(img);
    }
    img.src = ev.preview;
  }
}

function onError(ev) {
  const running = plan.find((s) => $("card-" + s.id)?.classList.contains("running"));
  if (running) {
    setStatus(running.id, "error");
    $("card-" + running.id).querySelector(".card-body").innerHTML =
      `<div class="summary">${esc(ev.message)}</div>`;
  }
  showNotice(ev.message, true);
}

function onResult(ev) {
  const media = ev.mode === "video"
    ? `<video src="${ev.video}" controls autoplay muted loop playsinline></video>`
    : `<img src="${ev.annotated}" alt="Annotated result">`;
  $("report-media").innerHTML = media;
  $("downloads").innerHTML =
    `<a href="${ev.csv}" download>Download CSV</a>` +
    `<a href="${ev.mode === "video" ? ev.video : ev.annotated}" download>Download ${ev.mode}</a>`;
  const cols = ["", "id", "violation", "plate", "raw OCR", "confidence", "format", "status"];
  if (ev.mode === "video") cols.splice(6, 0, "frames", "no-helmet");
  const rows = ev.rows.map((r) => {
    const cells = [
      r.crop ? `<img class="crop" src="${r.crop}" alt="plate">` : "",
      esc(r.id), esc(r.violation),
      `<span class="mono"><b>${esc(r.plate)}</b></span>`,
      `<span class="mono">${esc(r.plate_raw || "-")}</span>`,
      r.ocr_confidence == null ? "-" : r.ocr_confidence.toFixed(2),
      r.valid_format == null ? "-" : statusPill(r.valid_format ? "yes" : "no"),
      statusPill(r.status),
    ];
    if (ev.mode === "video") cells.splice(6, 0, r.frames, `${Math.round(r.nohelm_ratio * 100)}%`);
    return `<tr>${cells.map((c) => `<td>${c}</td>`).join("")}</tr>`;
  });
  $("report-table").innerHTML = rows.length
    ? `<thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody>`
    : `<tbody><tr><td class="empty">No violations found.</td></tr></tbody>`;
  $("report").hidden = false;
  $("report").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function handle(ev) {
  switch (ev.type) {
    case "plan": buildPlan(ev); break;
    case "notice": showNotice(ev.text); break;
    case "step":
      onStep(ev);
      if (ev.status !== "running" && $("pace").checked) await sleep(PACE_MS);
      break;
    case "progress": onProgress(ev); break;
    case "error": onError(ev); break;
    case "result": onResult(ev); break;
  }
}

// ---------------------------------------------------------------- run

async function run() {
  if (!file || busy) return;
  busy = true;
  updateRunButton();
  showNotice(null);
  $("report").hidden = true;
  $("pipeline").hidden = true;

  // Events are queued and drained in order, so "presentation pace" can hold
  // finished steps on screen without slowing the server down.
  const queue = [];
  let streamDone = false;
  let wake = null;
  const drain = (async () => {
    while (true) {
      if (queue.length) {
        // during pacing, skip stale progress events so the queue never lags far behind
        const ev = queue.shift();
        if (ev.type === "progress" && queue.some((q) => q.type === "progress" && q.id === ev.id)) continue;
        await handle(ev);
      } else if (streamDone) break;
      else await new Promise((r) => (wake = r));
    }
  })();
  const push = (ev) => { queue.push(ev); wake?.(); wake = null; };

  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    if (!res.ok) {
      let msg = `Upload failed (${res.status})`;
      try { msg = (await res.json()).detail || msg; } catch {}
      throw new Error(msg);
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (line) push(JSON.parse(line));
      }
    }
  } catch (e) {
    push({ type: "error", message: e.message || String(e) });
  } finally {
    streamDone = true;
    wake?.();
    await drain;
    busy = false;
    updateRunButton();
  }
}

$("run").addEventListener("click", run);

// ---------------------------------------------------------------- lightbox

document.addEventListener("click", (e) => {
  const img = e.target.closest(".shots img, img.crop");
  if (img) {
    $("lightbox").querySelector("img").src = img.src;
    $("lightbox").hidden = false;
  }
});
$("lightbox").addEventListener("click", () => ($("lightbox").hidden = true));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("lightbox").hidden = true; });

checkHealth();
