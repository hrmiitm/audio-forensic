"use strict";

/* ═════════ Helpers ═════════ */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const icon = (id) => `<svg><use href="#i-${id}"/></svg>`;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const esc = (v) => String(v ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const colorOf = (c) => (c?.startsWith("--") ? cssVar(c) : c);
const cssColor = (c) => (c?.startsWith("--") ? `var(${c})` : c);
const fmt = (t) => {
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
};
const fmtDur = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const parseTime = (v) => {
  if (!v?.trim()) return null;
  const p = v.trim().split(":").map(Number);
  return p.some(isNaN) ? NaN : p.reduce((acc, x) => acc * 60 + x, 0);
};

let toastTimer;
function toast(msg, ms = 3200) {
  if (msg === DEMO_TEXT) return demoNotice();
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), ms);
}

async function api(path, opts = {}) {
  if (DEMO) return demoApi(path, opts);
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail ?? msg; } catch {}
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
}
const send = (method, body) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

// Debounced saves; while one is pending, polling won't overwrite the local value
const pending = {};
function saveLater(key, url, body) {
  if (DEMO) return; // demo edits stay in the page
  clearTimeout(pending[key]);
  pending[key] = setTimeout(() => {
    delete pending[key];
    api(url, send("PATCH", body())).catch((e) => toast(e.message));
  }, 350);
}

async function download(url) {
  toast("Preparing download…", 120000);
  try {
    const r = await api(url);
    const cd = r.headers.get("content-disposition") || "";
    const name = decodeURIComponent((/filename\*=utf-8''([^;]+)/i.exec(cd) || /filename="?([^";]+)"?/i.exec(cd) || [])[1] || "download");
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(await r.blob()), download: name });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    toast(`Saved ${name}`);
  } catch (e) { toast(e.message, 6000); }
}

/* GitHub Pages (or ?demo) runs the UI without a server */
const DEMO = location.hostname.endsWith("github.io") || new URLSearchParams(location.search).has("demo");
const REPO_URL = "https://github.com/hrmiitm/audio-forensic";
const DEMO_TEXT = "__demo__";

/* ═════════ State ═════════ */
const PPS = 50; // waveform peaks per second (server PEAKS_PER_SEC)
let MODELS = [], STATS = null, model = null, device = null;
let src = null, DUR = 0, tracks = [], jobs = [], splits = [], lastSeg = 0, lastError = null;
let selection = null, playT = 0, zoom = 1, spans = [];
const modelInfo = () => MODELS.find((m) => m.id === model);
const byId = (id) => tracks.find((t) => t.id === id);

/* ═════════ Models, devices, live usage ═════════ */
async function loadModels() {
  MODELS = await api("/api/models");
  const sel = $("#modelSelect");
  sel.innerHTML = MODELS.map((m) => `<option value="${m.id}" ${m.available ? "" : "disabled"}>${esc(m.label)}${m.available ? "" : " — unavailable"}</option>`).join("");
  model = (MODELS.find((m) => m.id === localStorage.getItem("af-model") && m.available) || MODELS.find((m) => m.available)).id;
  sel.value = model;
  renderModel();
}

function renderModel() {
  const m = modelInfo();
  $("#modelMeta").innerHTML = m.meta.map((x) => `<span>${esc(x)}</span>`).join("") + (m.note ? `<span class="dim">${esc(m.note)}</span>` : "");
  if (!m.devices.some((d) => d.id === device)) device = m.devices[0].id;
  const tag = (id) => ({ cuda: "CUDA", vulkan: "Vulkan", cpu: "CPU" })[id.split(":")[0]];
  $("#deviceList").innerHTML = m.devices.map((d) => `
    <label class="device ${d.id === device ? "on" : ""}">
      <input type="radio" name="device" value="${d.id}" ${d.id === device ? "checked" : ""}>
      ${icon("chip")}
      <span><span class="device-name">${esc(d.name)}</span><span class="device-detail">${esc(d.detail)}</span></span>
      <span class="device-tag">${tag(d.id)}</span>
      <span class="device-usage" data-dev="${d.id}"></span>
    </label>`).join("");
  $$(".field[data-for]").forEach((f) => (f.hidden = !f.dataset.for.split(" ").includes(model)));
  $("#optChunk").value = m.chunk;
  renderUsage();
  renderJob();
}

$("#modelSelect").addEventListener("change", (e) => {
  model = e.target.value;
  localStorage.setItem("af-model", model);
  renderModel();
});
$("#deviceList").addEventListener("change", (e) => {
  device = e.target.value;
  $$(".device").forEach((d) => d.classList.toggle("on", $("input", d).value === device));
});

/** Live usage % of a device (GPU: NVIDIA utilization, CPU: average over all threads). */
function usageOf(d) {
  if (!STATS) return null;
  if (d.id === "cpu") return STATS.cpu;
  const g = d.nv != null && STATS.gpus.find((x) => x.index === d.nv);
  return g ? g.util : null;
}

const meter = (pct, warn) => `<span class="meter ${warn && pct > 90 ? "high" : ""}"><span class="meter-fill" style="width:${clamp(pct, 0, 100)}%"></span></span>`;

function renderUsage() {
  for (const d of modelInfo()?.devices || []) {
    const el = $(`.device-usage[data-dev="${d.id}"]`), pct = usageOf(d);
    if (el) el.innerHTML = pct != null
      ? `<span>Usage</span>${meter(pct)}<span class="mono">${Math.round(pct)}%</span>`
      : `<span class="na">Live usage isn't available for this device</span>`;
  }
  const g = STATS?.gpus?.[0];
  $("#gpuMeter").hidden = !g;
  if (g) {
    const pct = (g.mem_used / g.mem_total) * 100;
    $("#gpuMeterName").textContent = `${g.name.replace(/NVIDIA GeForce |Laptop GPU/g, "").trim()} memory`;
    $("#gpuMeterFill").style.width = `${pct}%`;
    $("#gpuMeterFill").parentElement.classList.toggle("high", pct > 90);
    $("#gpuMeterText").textContent = `${g.mem_used.toFixed(1)} / ${g.mem_total.toFixed(1)} GB`;
    $("#gpuMeter").title = `GPU memory in use right now: ${g.mem_used.toFixed(2)} of ${g.mem_total.toFixed(1)} GB (all programs)`;
  }
  const stop = $("#stopBtn");
  stop.disabled = !(STATS?.active_jobs || STATS?.model_loaded);
  stop.title = STATS?.active_jobs ? "Stop all isolation and free GPU memory"
    : STATS?.model_loaded ? `Unload ${STATS.model_loaded} and free its memory` : "Nothing running";
}

async function pollStats() {
  try { STATS = await api("/api/stats"); } catch { STATS = null; } // server unreachable: keep polling
  renderUsage();
  setTimeout(pollStats, 1500);
}

$("#stopBtn").addEventListener("click", async () => {
  try {
    const r = await api("/api/jobs/stop", { method: "POST" });
    toast(r.stopped ? "Stopping… the model will be unloaded." : "Model unloaded, GPU memory freed.");
  } catch (e) { toast(e.message); }
  refreshSoon();
});

/* ═════════ Prompt: text, time hints, visual reference ═════════ */
const promptInput = $("#promptText");
$("#promptExamples").addEventListener("click", (e) => {
  if (e.target.tagName !== "BUTTON") return;
  promptInput.value = e.target.textContent;
  promptInput.focus();
});

function renderSpans() {
  $("#spanList").innerHTML = spans.map((s, i) => `
    <li class="span-row">
      <button class="span-sign ${s.sign === "+" ? "plus" : "minus"}" data-i="${i}" title="${s.sign === "+" ? "Sound present here (click to flip)" : "Sound absent here (click to flip)"}">${s.sign === "+" ? "+" : "−"}</button>
      <span class="mono">${fmt(s.a)} – ${fmt(s.b)}</span>
      <button class="icon-btn sm" data-del="${i}" title="Remove">${icon("x")}</button>
    </li>`).join("");
  $("#spanHint").hidden = spans.length > 0;
}
$("#spanList").addEventListener("click", (e) => {
  const sign = e.target.closest("[data-i]"), del = e.target.closest("[data-del]");
  if (sign) spans[sign.dataset.i].sign = spans[sign.dataset.i].sign === "+" ? "-" : "+";
  if (del) spans.splice(del.dataset.del, 1);
  renderSpans();
});
$("#addSpanBtn").addEventListener("click", () => {
  if (!selection) return;
  spans.push({ sign: "+", a: +selection.a.toFixed(2), b: +selection.b.toFixed(2) });
  renderSpans();
});

const visualFile = $("#visualFile");
function renderVisual() {
  const ok = !!src?.has_video;
  $("#visualSub").classList.toggle("disabled", !ok);
  $("#visualHint").textContent = ok
    ? "A copy of this video (or one frame) with everything except the sound source blacked out. The text can then be left empty."
    : "Available when the source has video.";
  const f = visualFile.files[0];
  $("#visualName").textContent = f ? f.name : "Masked image or video";
  $("#visualClear").hidden = !f;
}
visualFile.addEventListener("change", renderVisual);
$("#visualClear").addEventListener("click", (e) => {
  e.preventDefault();
  visualFile.value = "";
  renderVisual();
});

$$(".segmented").forEach((seg) => seg.addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (b && !b.disabled) $$("button", seg).forEach((x) => x.classList.toggle("on", x === b));
}));
const segValue = (id) => $(`#${id} .on`).dataset.v;
const setSeg = (id, v) => $$(`#${id} button`).forEach((b) => b.classList.toggle("on", b.dataset.v === v));
$("#seedRand").addEventListener("click", (e) => {
  e.preventDefault();
  $("#optSeed").value = Math.floor(Math.random() * 1e6);
});

/* ═════════ Isolation jobs ═════════ */
async function isolate() {
  if (!src || $("#isolateBtn").disabled) return;
  const prompt = promptInput.value.trim().toLowerCase(), visual = src.has_video ? visualFile.files[0] : null;
  if (!prompt && !visual) { promptInput.focus(); return toast("Name the sound to isolate (or add a visual reference)"); }
  const spec = {
    source_id: src.id, model, device, prompt,
    input: segValue("fromSeg"),
    range: segValue("rangeSeg") === "selection" && selection ? [selection.a, selection.b] : null,
    anchors: spans.map((s) => [s.sign, s.a, s.b]),
    options: { chunk: +$("#optChunk").value, overlap: +$("#optOverlap").value, steps: +$("#optSteps").value,
               seed: +$("#optSeed").value, bounded: $("#optBounded").checked },
  };
  const fd = new FormData();
  fd.append("spec", JSON.stringify(spec));
  if (visual) fd.append("visual", visual);
  try {
    await api("/api/jobs", { method: "POST", body: fd });
    refreshSoon();
  } catch (e) { toast(e.message, 6000); }
}
$("#isolateBtn").addEventListener("click", isolate);

function renderJob() {
  const active = jobs.filter((j) => j.status === "queued" || j.status === "running"), last = jobs.at(-1);
  const show = active[0] || (last && (last.status === "failed" || last.status === "cancelled") ? last : null);
  const box = $("#jobBox");
  box.hidden = !show;
  if (show) {
    box.classList.toggle("failed", show.status === "failed");
    $("#jobLabel").textContent = `“${show.prompt || "visual source"}” · ${show.message || show.status}${active.length > 1 ? ` · ${active.length - 1} more queued` : ""}`;
    $("#jobPct").textContent = show.status === "running" ? `${Math.round(show.progress * 100)}%` : "";
    $("#jobFill").style.width = `${show.progress * 100}%`;
  }
  const done = [...jobs].reverse().find((j) => j.status === "done" && j.model === model);
  $("#footNote").textContent = done
    ? `Last run: ${done.message}${done.peak_vram_mib ? ` · peak ${(done.peak_vram_mib / 1024).toFixed(1)} GB` : ""}`
    : modelInfo()?.note || "";
  $("#isolateBtn").disabled = !src;
}

/* ═════════ Sources: open, poll, history ═════════ */
const card = $("#sourceCard"), video = $("#video");

function setSourceTab(tab) {
  card.dataset.tab = tab;
  $$("#sourceTabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
}
function setSourceState(state) {
  card.dataset.state = state;
  $("#replaceBtn").textContent = state === "loaded" ? "Replace" : "Cancel";
  $("#sourceMeta").hidden = !src;
  if (state === "loaded" && src) setSourceTab(src.kind);
}
$("#sourceTabs").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  setSourceTab(b.dataset.tab);
  setSourceState("empty");
});
$("#replaceBtn").addEventListener("click", () => setSourceState(card.dataset.state === "loaded" ? "empty" : "loaded"));

async function openSource(id) {
  stopPlay();
  setWaiting(false);
  bufs.clear();
  src = null; tracks = []; jobs = []; splits = []; lastSeg = 0; lastError = null; selection = null; spans = []; playT = 0;
  if (id) {
    try {
      const d = await api(`/api/sources/${id}`);
      src = d.source; DUR = src.duration_s;
      applyState(d);
      await loadSegments();
      localStorage.setItem("af-source", id);
    } catch (e) {
      src = null;
      toast(e.message);
    }
  }
  if (!src) localStorage.removeItem("af-source");
  video.removeAttribute("src");
  if (src?.has_video) video.src = `/api/sources/${src.id}/video`;
  video.load();
  $("#videoFrame").hidden = !src?.has_video;
  $("#overview").hidden = !src || !!src.has_video;
  $("#sessionName").disabled = !src;
  $("#sessionName").value = src?.name || "";
  $("#exportBtn").disabled = !src;
  $("#exportMp4").disabled = !src?.has_video;
  setSourceState(src ? "loaded" : "empty");
  renderSpans(); renderVisual(); setSelection(null); setPlay(0); renderJob();
  renderTracks();
  refreshSoon();
}

function sourceInfo() {
  if (!src) return "";
  const kind = { audio: "audio", video: "video", youtube: src.yt_mode === "save" ? "YouTube · copy saved" : "YouTube · streamed" }[src.kind];
  return `${fmtDur(DUR)} · ${kind}`;
}

function applyState(d) {
  src = d.source;
  jobs = d.jobs;
  const old = new Map(tracks.map((t) => [t.id, t])), prevIds = tracks.map((t) => t.id).join();
  tracks = d.tracks.map((t) => {
    const o = old.get(t.id);
    if (!o) return { ...t, peaks: new Float32Array(Math.ceil(DUR * PPS) + 1), segs: [] };
    return Object.assign(o, pending[`t${t.id}`] ? { status: t.status, covered: t.covered } : t);
  });
  if (!pending.splits) splits = src.splits;
  if (!pending.name && document.activeElement !== $("#sessionName")) $("#sessionName").value = src.name;
  $("#metaName").textContent = src.name;
  $("#metaInfo").textContent = sourceInfo();
  $("#tcDur").textContent = fmt(DUR);
  const hasResidual = tracks.some((t) => t.role === "residual");
  $("#fromResidual").disabled = !hasResidual;
  if (!hasResidual) setSeg("fromSeg", "original");
  if (src.error && src.error !== lastError) toast(src.error, 7000);
  lastError = src.error;
  renderJob();
  return prevIds !== tracks.map((t) => t.id).join();
}

async function loadSegments() {
  const changed = new Set();
  for (const s of await api(`/api/sources/${src.id}/segments?after=${lastSeg}`)) {
    lastSeg = Math.max(lastSeg, s.id);
    const tr = byId(s.track_id);
    if (!tr) continue;
    tr.segs.push({ id: s.id, start: s.start, dur: s.dur });
    const p = Uint8Array.from(atob(s.peaks), (c) => c.charCodeAt(0)), at = Math.round(s.start * PPS);
    for (let i = 0; i < p.length && at + i < tr.peaks.length; i++) tr.peaks[at + i] = p[i] / 255;
    changed.add(tr);
  }
  return changed;
}

let pollTimer = null;
function refreshSoon(ms = 0) { if (DEMO) return; clearTimeout(pollTimer); pollTimer = setTimeout(refresh, ms); }
async function refresh() {
  if (!src) return;
  const id = src.id;
  try {
    const d = await api(`/api/sources/${id}`);
    if (src?.id !== id) return;
    const structural = applyState(d);
    const changed = await loadSegments();
    if (structural) renderTracks();
    else {
      tracks.forEach((t) => renderHead(t) && changed.add(t));
      changed.forEach(drawTrack);
      tracks.forEach(drawPending);
      applyGains();
      drawOverview();
    }
    resumeIfReady();
  } catch (e) {
    if (/not found/i.test(e.message)) return openSource(null);
  }
  const busy = tracks.some((t) => t.status === "partial") || jobs.some((j) => j.status === "queued" || j.status === "running");
  refreshSoon(busy ? 800 : 4000);
}

$("#sessionName").addEventListener("input", (e) => {
  if (src) saveLater("name", `/api/sources/${src.id}`, () => ({ name: e.target.value.trim() || "Untitled" }));
});

/* Uploads and YouTube */
function uploadFile(file, drop) {
  if (DEMO) return demoNotice();
  const label = $("strong", drop), sub = $(".drop-sub", drop), old = [label.innerHTML, sub.textContent];
  drop.classList.add("busy");
  const done = () => { drop.classList.remove("busy"); [label.innerHTML, sub.textContent] = old; };
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/sources/upload");
  xhr.upload.onprogress = (e) => { label.textContent = `Uploading ${file.name}`; sub.textContent = `${Math.round((e.loaded / e.total) * 100)}%`; };
  xhr.upload.onload = () => { sub.textContent = "Reading the file…"; };
  xhr.onload = () => {
    done();
    let r = {};
    try { r = JSON.parse(xhr.responseText); } catch {}
    xhr.status < 300 ? openSource(r.id) : toast(r.detail || "Upload failed", 6000);
  };
  xhr.onerror = () => { done(); toast("Upload failed"); };
  const fd = new FormData();
  fd.append("file", file);
  xhr.send(fd);
}
$$(".drop").forEach((drop) => {
  const take = (f) => f && uploadFile(f, drop);
  $("input", drop).addEventListener("change", (e) => { take(e.target.files[0]); e.target.value = ""; });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); take(e.dataTransfer.files[0]); });
});
$("#ytModes").addEventListener("change", () => $$(".mode").forEach((m) => m.classList.toggle("on", $("input", m).checked)));
$("#ytForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const start = parseTime($("#ytStart").value) ?? 0, end = parseTime($("#ytEnd").value);
  if (Number.isNaN(start) || Number.isNaN(end)) return toast("Use m:ss for start and end, e.g. 1:30");
  const btn = $("#ytFetch");
  btn.disabled = true;
  btn.textContent = "Opening…";
  try {
    const r = await api("/api/sources/youtube", send("POST", { url: $("#ytUrl").value, mode: $("input[name=ytmode]:checked").value, start, end }));
    $("#ytUrl").value = "";
    await openSource(r.id);
  } catch (err) { toast(err.message, 7000); }
  btn.disabled = false;
  btn.textContent = "Open";
});

/* History drawer */
async function renderHistory() {
  const list = await api("/api/sources");
  $("#historyList").innerHTML = list.length ? list.map((h) => `
    <li class="${h.id === src?.id ? "current" : ""}" data-id="${h.id}">
      <span class="h-icon">${icon(h.kind)}</span>
      <span><div class="h-name">${esc(h.name)}</div><div class="h-meta">${fmtDur(h.duration_s)} · ${h.isolated} isolated${h.kind === "youtube" ? ` · ${h.yt_mode === "save" ? "copy saved" : "streamed"}` : ""}</div></span>
      <span class="h-date">${esc(h.created_at.slice(5, 16))}</span>
      <button class="icon-btn sm h-del" data-del="${h.id}" title="Delete session and its files">${icon("trash")}</button>
    </li>`).join("") : `<li class="empty">Nothing here yet.</li>`;
}
function drawer(open) {
  $("#drawer").hidden = !open;
  $("#scrim").hidden = !open;
  if (open) renderHistory().catch((e) => toast(e.message));
}
$("#historyBtn").addEventListener("click", () => drawer(true));
$("#drawerClose").addEventListener("click", () => drawer(false));
$("#scrim").addEventListener("click", () => drawer(false));
$("#historyList").addEventListener("click", async (e) => {
  const del = e.target.closest("[data-del]"), li = e.target.closest("li[data-id]");
  if (del) {
    if (!confirm("Delete this session, its separated tracks and exports?")) return;
    await api(`/api/sources/${del.dataset.del}`, { method: "DELETE" }).catch((err) => toast(err.message));
    if (+del.dataset.del === src?.id) openSource(null);
    return renderHistory();
  }
  if (li) { drawer(false); openSource(+li.dataset.id); }
});

/* Export menu */
$("#exportBtn").addEventListener("click", (e) => {
  const m = $("#exportMenu"), open = m.hidden;
  closeMenus();
  m.hidden = !open;
  e.stopPropagation();
});
$("#exportMenu").addEventListener("click", (e) => {
  const item = e.target.closest("[data-format]");
  if (!item || item.disabled) return;
  closeMenus();
  download(`/api/sources/${src.id}/export?format=${item.dataset.format}`);
});
function closeMenus() { $$(".menu").forEach((m) => (m.hidden = true)); }
document.addEventListener("click", (e) => { if (!e.target.closest(".menu-wrap")) closeMenus(); });

/* ═════════ Timeline: tracks, waveforms, splits, clip edits ═════════ */
const heads = $("#tlHeads"), lanes = $("#lanes"), inner = $("#tlInner"), scroller = $("#tlScroll");
const pxPerSec = () => inner.clientWidth / Math.max(DUR, 0.001);
const timeAt = (clientX) => clamp((clientX - inner.getBoundingClientRect().left) / pxPerSec(), 0, DUR);
const bounds = () => [0, ...splits, DUR];
function clipAt(t) {
  const b = bounds();
  for (let i = 0; i < b.length - 1; i++) if (t >= b[i] && t < b[i + 1]) return { i, a: b[i], b: b[i + 1] };
  return { i: b.length - 2, a: b[b.length - 2], b: DUR };
}
const editAt = (tr, t) => tr.edits.find((e) => t >= e.a && t < e.b);
const saveTrack = (tr) => saveLater(`t${tr.id}`, `/api/tracks/${tr.id}`, () => ({ name: tr.name, muted: tr.muted, solo: tr.solo, gain: tr.gain, edits: tr.edits }));
const saveSplits = () => { saveLater("splits", `/api/sources/${src.id}`, () => ({ splits })); tracks.forEach(saveTrack); };

function trackSub(tr) {
  const job = jobs.find((j) => j.id === tr.job_id);
  const pct = DUR ? Math.min(100, Math.round((tr.covered / DUR) * 100)) : 0;
  const state = {
    partial: `<span class="status">${tr.role === "original" ? "loading" : "separating"} ${pct}%</span>`,
    failed: `<span class="status failed">failed</span>`,
    cancelled: `<span class="status failed">stopped at ${pct}%</span>`,
  }[tr.status] || "";
  const base = tr.role === "original" ? sourceInfo()
    : job ? `${job.model.toUpperCase()} · ${job.device}${tr.status === "complete" && job.elapsed_s ? ` · ${job.elapsed_s.toFixed(1)} s` : ""}` : "";
  return [base, state].filter(Boolean).join(" · ");
}

function headHTML(tr) {
  return `
    <div class="track-title"><span class="swatch" style="background:${cssColor(tr.color)}"></span><input class="track-name" value="${esc(tr.name)}" aria-label="Track name" spellcheck="false"></div>
    <div class="track-menu menu-wrap">
      <button class="icon-btn sm" data-act="menu" title="More">${icon("more")}</button>
      <div class="menu" hidden>
        <button class="menu-item" data-act="download">Download WAV</button>
        ${tr.role === "target" ? `<button class="menu-item" data-act="reuse">Reuse prompt</button>` : ""}
        <button class="menu-item" data-act="reset">Reset clip levels</button>
        ${tr.role === "original" ? "" : `<button class="menu-item danger" data-act="remove">Remove track</button>`}
      </div>
    </div>
    <div class="track-sub">${trackSub(tr)}</div>
    <div class="track-ctrls">
      <button class="icon-btn ${tr.muted ? "mute-on" : ""}" data-act="mute" title="${tr.muted ? "Unmute track" : "Mute track"}">${icon(tr.muted ? "mute" : "vol")}</button>
      <button class="icon-btn solo ${tr.solo ? "on" : ""}" data-act="solo" title="Solo: hear only soloed tracks">S</button>
      <input type="range" class="slider" min="0" max="200" value="${tr.gain}" data-act="gain" aria-label="Track volume">
      <span class="mono">${Math.round(tr.gain)}%</span>
    </div>`;
}

/** Re-render a track header if it changed and isn't being used. Returns true when it changed. */
function renderHead(tr, force = false) {
  const h = $(`.track-head[data-id="${tr.id}"]`);
  if (!h || (!force && (h.contains(document.activeElement) || !$(".menu", h)?.hidden))) return false;
  const html = headHTML(tr);
  h.classList.toggle("muted", !!tr.muted);
  if (h._html === html) return false;
  h.innerHTML = h._html = html;
  h.title = tr.role === "residual" ? $(".track-sub", h).textContent : ""; // residual rows hide the sub line
  return true;
}

function renderTracks() {
  $("#tlBody").classList.toggle("empty", !src);
  $$(".track-head", heads).forEach((n) => n.remove());
  lanes.innerHTML = "";
  for (const tr of tracks) {
    const h = Object.assign(document.createElement("div"), { className: `track-head ${tr.role}` });
    h.dataset.id = tr.id;
    heads.appendChild(h);
    renderHead(tr, true);
    const lane = Object.assign(document.createElement("div"), { className: `lane ${tr.role}`, innerHTML: "<canvas></canvas>" });
    lane.dataset.id = tr.id;
    lanes.appendChild(lane);
    tr.lane = lane;
    ensureNodes(tr);
  }
  layout();
  applyGains();
}

function drawTrack(tr) {
  const lane = tr.lane;
  if (!lane) return;
  const c = $("canvas", lane), w = lane.clientWidth, h = lane.clientHeight, dpr = devicePixelRatio || 1;
  c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
  const ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  const dim = !audible(tr), mid = h / 2, maxA = h / 2 - 7, n = tr.peaks.length;
  ctx.fillStyle = cssVar("--wave-mid");
  ctx.fillRect(0, Math.round(mid), w, 1);
  ctx.fillStyle = colorOf(tr.color);
  let lastAlpha = -1;
  for (let x = 0; x < w; x++) {
    const i0 = Math.floor((x / w) * n), i1 = Math.max(i0 + 1, Math.floor(((x + 1) / w) * n));
    let a = 0;
    for (let i = i0; i < i1 && i < n; i++) a = Math.max(a, tr.peaks[i]);
    if (!a) continue;
    const e = editAt(tr, ((x + 0.5) / w) * DUR), alpha = dim || e?.muted ? 0.2 : 0.9;
    if (alpha !== lastAlpha) { ctx.globalAlpha = alpha; lastAlpha = alpha; }
    const amp = Math.min(1, a * (tr.gain / 100) * (e ? e.gain / 100 : 1)) * maxA;
    ctx.fillRect(x, mid - amp, 1, Math.max(1, amp * 2));
  }
  $$(".clip-badge", lane).forEach((b) => b.remove());
  for (const e of tr.edits) {
    const b = Object.assign(document.createElement("span"), { className: `clip-badge${e.muted ? " mute" : ""}`, textContent: e.muted ? "Muted" : `${e.gain}%` });
    b.style.left = `${e.a * (w / DUR) + 4}px`;
    lane.appendChild(b);
  }
  drawPending(tr);
}

function drawPending(tr) {
  if (!tr.lane) return;
  let p = $(".lane-pending", tr.lane);
  if (tr.status === "complete" || tr.covered >= DUR - 0.05) return p?.remove();
  if (!p) tr.lane.appendChild((p = Object.assign(document.createElement("div"), { className: "lane-pending" })));
  p.style.left = `${tr.covered * pxPerSec()}px`;
  p.textContent = { partial: tr.role === "original" ? "Loading…" : "Separating…", failed: "Failed", cancelled: "Stopped" }[tr.status];
}
const drawAll = () => tracks.forEach(drawTrack);

function renderRuler() {
  const pps = pxPerSec();
  const minor = [0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60].find((s) => s * pps >= 10) ?? 60;
  const label = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600].find((s) => s * pps >= 64) ?? 600;
  let html = "";
  for (let k = 0; k * minor <= DUR + 1e-6; k++) {
    const t = k * minor, major = Math.abs(t / label - Math.round(t / label)) < 1e-6;
    html += `<span class="tick ${major ? "major" : ""}" style="left:${t * pps}px"></span>`;
    if (major) html += `<span class="tick-label" style="left:${t * pps}px">${fmtDur(Math.round(t))}</span>`;
  }
  html += splits.map((s) => `<span class="split-handle" data-t="${s}" style="left:${s * pps}px" title="Drag to move · double-click to remove"></span>`).join("");
  $("#ruler").innerHTML = html;
}

function renderOverlay() {
  const pps = pxPerSec(), ov = $("#overlay");
  $$(".split-line", ov).forEach((n) => n.remove());
  for (const s of splits) {
    const l = Object.assign(document.createElement("div"), { className: "split-line" });
    l.style.left = `${s * pps}px`;
    ov.appendChild(l);
  }
  const sel = $("#selection");
  sel.hidden = !selection;
  if (selection) { sel.style.left = `${selection.a * pps}px`; sel.style.width = `${(selection.b - selection.a) * pps}px`; }
  $("#playhead").style.left = `${playT * pps}px`;
}

function layout() {
  if (!src) return;
  inner.style.width = `${scroller.clientWidth * zoom}px`;
  renderRuler(); drawAll(); renderOverlay(); drawOverview();
}
new ResizeObserver(layout).observe(scroller);

function drawOverview() {
  const c = $("#overview"), orig = tracks.find((t) => t.role === "original");
  if (c.hidden || !orig) return;
  const w = c.clientWidth, h = c.clientHeight, dpr = devicePixelRatio || 1, n = orig.peaks.length;
  c.width = w * dpr; c.height = h * dpr;
  const ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.fillStyle = cssVar("--wave-neutral");
  for (let x = 0; x < w; x++) {
    let a = 0;
    for (let i = Math.floor((x / w) * n), e = Math.max(i + 1, Math.floor(((x + 1) / w) * n)); i < e; i++) a = Math.max(a, orig.peaks[i] || 0);
    const y = a * (h / 2 - 12);
    if (y) ctx.fillRect(x, h / 2 - y, 1, y * 2);
  }
  ctx.fillStyle = cssVar("--text");
  ctx.fillRect((playT / DUR) * w, 0, 1.5, h);
}
new ResizeObserver(drawOverview).observe($("#overview"));

function setSelection(sel) {
  selection = sel && sel.b - sel.a > 0.05 ? sel : null;
  $("#selLabel").textContent = selection ? `${fmt(selection.a)} – ${fmt(selection.b)}` : "";
  $("#splitSelBtn").disabled = !selection;
  $("#addSpanBtn").disabled = !selection;
  const rb = $("#rangeSelBtn");
  rb.disabled = !selection;
  if (!selection && rb.classList.contains("on")) setSeg("rangeSeg", "all");
  rb.textContent = selection ? `Selection · ${(selection.b - selection.a).toFixed(1)} s` : "Selection";
  renderOverlay();
}

function setPlay(t) {
  playT = clamp(t, 0, DUR);
  $("#tcNow").textContent = fmt(playT);
  $("#videoTime").textContent = fmt(playT);
  $("#playhead").style.left = `${playT * pxPerSec()}px`;
  drawOverview();
}

/* Track header interactions */
heads.addEventListener("click", async (e) => {
  const head = e.target.closest(".track-head"), btn = e.target.closest("[data-act]");
  if (!head || !btn) return;
  const tr = byId(+head.dataset.id), act = btn.dataset.act;
  if (btn.classList.contains("menu-item")) closeMenus();
  if (act === "mute" || act === "solo") {
    const key = act === "mute" ? "muted" : "solo";
    tr[key] = tr[key] ? 0 : 1;
    saveTrack(tr);
    renderHead(tr, true); drawAll(); applyGains();
  }
  if (act === "menu") {
    const m = $(".menu", head), open = m.hidden;
    closeMenus();
    m.hidden = !open;
    m.classList.toggle("up", head.getBoundingClientRect().bottom + 150 > $("#tlBody").getBoundingClientRect().bottom);
    e.stopPropagation();
  }
  if (act === "download") download(`/api/tracks/${tr.id}/download`);
  if (act === "reuse") { promptInput.value = jobs.find((j) => j.id === tr.job_id)?.prompt || tr.name.toLowerCase(); promptInput.focus(); }
  if (act === "reset") { tr.edits = []; saveTrack(tr); drawTrack(tr); applyGains(); }
  if (act === "remove" && confirm(`Remove “${tr.name}”${tr.status === "partial" ? " and stop its isolation" : ""}?`)) {
    await api(`/api/tracks/${tr.id}`, { method: "DELETE" }).catch((err) => toast(err.message));
    refreshSoon();
  }
});
heads.addEventListener("input", (e) => {
  const head = e.target.closest(".track-head"), tr = head && byId(+head.dataset.id);
  if (!tr) return;
  if (e.target.dataset.act === "gain") {
    tr.gain = +e.target.value;
    e.target.nextElementSibling.textContent = `${tr.gain}%`;
    drawTrack(tr); applyGains();
  }
  if (e.target.classList.contains("track-name")) tr.name = e.target.value;
  saveTrack(tr);
});
heads.addEventListener("change", (e) => { if (e.target.type === "range") e.target.blur(); });

/* Lanes: drag to select, click to seek, hover a clip for its controls */
let drag = null;
lanes.addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return;
  drag = { x: e.clientX, t: timeAt(e.clientX) };
  lanes.setPointerCapture(e.pointerId);
  hidePop(true);
});
lanes.addEventListener("pointermove", (e) => {
  if (!drag) return showPop(e);
  if (Math.abs(e.clientX - drag.x) > 3) {
    const t = timeAt(e.clientX);
    setSelection({ a: Math.min(drag.t, t), b: Math.max(drag.t, t) });
  }
});
lanes.addEventListener("pointerup", (e) => {
  if (drag && Math.abs(e.clientX - drag.x) <= 3) { seek(drag.t); setSelection(null); }
  drag = null;
});
lanes.addEventListener("pointerleave", () => hidePop());
scroller.addEventListener("wheel", (e) => {
  if (!e.ctrlKey) return;
  e.preventDefault();
  setZoom(zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15), e.clientX);
}, { passive: false });

const pop = $("#clipPop");
let popTarget = null, hideTimer;
function showPop(e) {
  const lane = e.target.closest(".lane");
  if (!lane) return;
  clearTimeout(hideTimer);
  const tr = byId(+lane.dataset.id), clip = clipAt(timeAt(e.clientX));
  if (popTarget && popTarget.tr === tr && popTarget.clip.i === clip.i && !pop.hidden) return;
  popTarget = { tr, clip };
  $$(".clip-hover").forEach((n) => n.remove());
  const pps = pxPerSec(), hl = Object.assign(document.createElement("div"), { className: "clip-hover" });
  hl.style.left = `${clip.a * pps}px`;
  hl.style.width = `${(clip.b - clip.a) * pps}px`;
  lane.appendChild(hl);
  const x = clamp(((clip.a + clip.b) / 2) * pps, scroller.scrollLeft + 130, scroller.scrollLeft + scroller.clientWidth - 130);
  const above = lane.offsetTop > 70;
  pop.style.left = `${x}px`;
  pop.style.top = `${above ? lane.offsetTop : lane.offsetTop + lane.offsetHeight}px`;
  pop.style.transform = above ? "translate(-50%, calc(-100% + 6px))" : "translate(-50%, -6px)";
  syncPop();
  pop.hidden = false;
}
function syncPop() {
  const { tr, clip } = popTarget, e = tr.edits.find((x) => x.a === clip.a && x.b === clip.b);
  const gain = e ? e.gain : 100, muted = !!e?.muted;
  $("#clipPopTitle").innerHTML = `<b>${esc(tr.name)}</b> clip ${clip.i + 1} · ${fmt(clip.a)}–${fmt(clip.b)}`;
  $("#clipGain").value = gain;
  $("#clipGainVal").textContent = muted ? "muted" : `${gain}%`;
  const m = $("#clipMute");
  m.classList.toggle("mute-on", muted);
  m.innerHTML = icon(muted ? "mute" : "vol");
  m.title = muted ? "Unmute this clip" : "Mute this clip";
}
function hidePop(now) {
  clearTimeout(hideTimer);
  const go = () => { pop.hidden = true; popTarget = null; $$(".clip-hover").forEach((n) => n.remove()); };
  now ? go() : (hideTimer = setTimeout(go, 220));
}
pop.addEventListener("pointerenter", () => clearTimeout(hideTimer));
pop.addEventListener("pointerleave", () => hidePop());
function editClip(change) {
  const { tr, clip } = popTarget;
  let e = tr.edits.find((x) => x.a === clip.a && x.b === clip.b);
  if (!e) tr.edits.push((e = { a: clip.a, b: clip.b, gain: 100, muted: false }));
  change(e);
  tr.edits = tr.edits.filter((x) => x.muted || x.gain !== 100);
  syncPop(); drawTrack(tr); saveTrack(tr); applyGains();
}
$("#clipGain").addEventListener("input", (ev) => editClip((e) => (e.gain = +ev.target.value)));
$("#clipMute").addEventListener("click", () => editClip((e) => (e.muted = !e.muted)));
$("#clipReset").addEventListener("click", () => editClip((e) => Object.assign(e, { gain: 100, muted: false })));

/* Splits: add, drag, remove (clip edits follow their clip) */
function addSplit(t) {
  if (t < 0.05 || t > DUR - 0.05 || splits.some((s) => Math.abs(s - t) < 0.05)) return false;
  t = +t.toFixed(3);
  splits = [...splits, t].sort((x, y) => x - y);
  for (const tr of tracks) {
    const e = tr.edits.find((x) => t > x.a && t < x.b);
    if (e) { tr.edits.push({ ...e, a: t }); e.b = t; }
  }
  return true;
}
function removeSplit(t) {
  splits = splits.filter((s) => s !== t);
  const next = bounds().find((x) => x > t) ?? DUR;
  for (const tr of tracks) {
    const left = tr.edits.find((e) => e.b === t), right = tr.edits.find((e) => e.a === t);
    if (right) tr.edits.splice(tr.edits.indexOf(right), 1);
    if (left) left.b = next; // the merged clip keeps the left clip's settings
  }
}
function moveSplit(oldT, newT) {
  const i = splits.indexOf(oldT);
  newT = +clamp(newT, (splits[i - 1] ?? 0) + 0.05, (splits[i + 1] ?? DUR) - 0.05).toFixed(3);
  splits[i] = newT;
  for (const tr of tracks) for (const e of tr.edits) { if (e.a === oldT) e.a = newT; if (e.b === oldT) e.b = newT; }
  return newT;
}
function splitsChanged() { renderRuler(); renderOverlay(); drawAll(); applyGains(); saveSplits(); }

const ruler = $("#ruler");
let handleDrag = null;
ruler.addEventListener("pointerdown", (e) => {
  const h = e.target.closest(".split-handle");
  if (h) { handleDrag = +h.dataset.t; ruler.setPointerCapture(e.pointerId); return; }
  seek(timeAt(e.clientX));
});
ruler.addEventListener("pointermove", (e) => {
  if (handleDrag === null) return;
  handleDrag = moveSplit(handleDrag, timeAt(e.clientX));
  renderRuler(); renderOverlay(); drawAll();
});
ruler.addEventListener("pointerup", () => { if (handleDrag !== null) splitsChanged(); handleDrag = null; });
ruler.addEventListener("dblclick", (e) => {
  const h = e.target.closest(".split-handle");
  if (h) { removeSplit(+h.dataset.t); splitsChanged(); }
});
$("#splitBtn").addEventListener("click", () => {
  if (src) addSplit(playT) ? splitsChanged() : toast("Move the playhead away from the edges or an existing split");
});
$("#splitSelBtn").addEventListener("click", () => { addSplit(selection.a); addSplit(selection.b); splitsChanged(); });

function setZoom(z, anchorX) {
  const old = zoom, rect = scroller.getBoundingClientRect();
  const ax = anchorX !== undefined ? anchorX - rect.left : scroller.clientWidth / 2;
  const frac = (scroller.scrollLeft + ax) / (scroller.clientWidth * old);
  zoom = clamp(z, 1, 40);
  $("#zoom").value = zoom;
  layout();
  scroller.scrollLeft = frac * scroller.clientWidth * zoom - ax;
}
$("#zoom").addEventListener("input", (e) => setZoom(+e.target.value));
$("#zoomIn").addEventListener("click", () => setZoom(zoom * 1.5));
$("#zoomOut").addEventListener("click", () => setZoom(zoom / 1.5));

/* ═════════ Player: Web Audio, block by block ═════════ */
const AC = new AudioContext();
const master = AC.createGain();
master.connect(AC.destination);
master.gain.value = $("#master").value / 100;
$("#master").addEventListener("input", (e) => master.gain.setTargetAtTime(e.target.value / 100, AC.currentTime, 0.02));

const bufs = new Map(); // segment id -> { seg, p: Promise<AudioBuffer> }
let play = null, gen = 0, loop = false, waitingAt = null, lastSchedule = 0;
const LOOKAHEAD = 4, PREFETCH = 20, KEEP_BEHIND = 15, KEEP_AHEAD = 45;
const videoOffset = () => (src?.kind === "youtube" ? src.offset_s : 0);
const now = () => (play ? play.t0 + (AC.currentTime - play.ctx0) : playT);
const audible = (tr) => !tr.muted && (tr.solo || !tracks.some((t) => t.solo));

function ensureNodes(tr) {
  if (tr.gainNode) return;
  tr.gainNode = AC.createGain();
  tr.editNode = AC.createGain();
  tr.gainNode.connect(tr.editNode).connect(master);
}

function loadBuf(seg) {
  if (!bufs.has(seg.id)) {
    bufs.set(seg.id, { seg, p: fetch(`/api/segments/${seg.id}`).then((r) => r.arrayBuffer()).then((b) => AC.decodeAudioData(b)) });
  }
  return bufs.get(seg.id).p;
}

const editValue = (tr, t) => { const e = editAt(tr, t); return e ? (e.muted ? 0 : e.gain / 100) : 1; };

function applyGains() {
  const t = now();
  for (const tr of tracks) {
    ensureNodes(tr);
    tr.gainNode.gain.setTargetAtTime(audible(tr) ? tr.gain / 100 : 0, AC.currentTime, 0.015);
    const p = tr.editNode.gain;
    p.cancelScheduledValues(0);
    p.setValueAtTime(editValue(tr, t), AC.currentTime);
    if (!play) continue;
    const edges = [...new Set(tr.edits.flatMap((e) => [e.a, e.b]))].filter((x) => x > t + 0.01).sort((a, b) => a - b);
    for (const x of edges) { // 5 ms ramps, same as the WAV export
      const at = play.ctx0 + (x - play.t0);
      p.setValueAtTime(editValue(tr, x - 1e-4), at - 0.0025);
      p.linearRampToValueAtTime(editValue(tr, x + 1e-4), at + 0.0025);
    }
  }
}

function schedule() {
  if (!play) return;
  const t = now(), g = gen;
  for (const tr of tracks) {
    for (const s of tr.segs) {
      if (s.start + s.dur <= t || s.start > t + PREFETCH) continue;
      loadBuf(s);
      if (s.start > t + LOOKAHEAD || play.srcs.has(s.id)) continue;
      play.srcs.set(s.id, null);
      loadBuf(s).then((buf) => {
        if (g !== gen || !play) return;
        const when = play.ctx0 + (s.start - play.t0), at = Math.max(when, AC.currentTime + 0.01), off = at - when;
        if (off >= buf.duration) return;
        const node = AC.createBufferSource();
        node.buffer = buf;
        node.connect(tr.gainNode);
        node.start(at, off);
        play.srcs.set(s.id, node);
      }).catch(() => play?.srcs.delete(s.id));
    }
  }
  for (const [id, b] of bufs) if (b.seg.start + b.seg.dur < t - KEEP_BEHIND || b.seg.start > t + KEEP_AHEAD) bufs.delete(id);
}

// Audible tracks that are still being produced must reach the playhead, otherwise we wait
const ready = (t) => tracks.every((tr) => !audible(tr) || tr.status !== "partial" || tr.covered >= Math.min(t, DUR - 0.05));

function startPlay(t) {
  if (DEMO) return demoNotice();
  stopPlay();
  if (!src || !tracks.length) return;
  if (t >= DUR - 0.01) t = 0;
  AC.resume();
  if (!ready(t + 0.5)) { setPlay(t); return setWaiting(true, t); }
  setWaiting(false);
  play = { ctx0: AC.currentTime + 0.12, t0: t, srcs: new Map() };
  if (src.has_video) { video.currentTime = t + videoOffset(); video.play().catch(() => {}); }
  applyGains();
  schedule();
  $("#playBtn").innerHTML = icon("pause");
  requestAnimationFrame(tick);
}

function stopPlay() {
  gen++;
  if (play) {
    playT = clamp(now(), 0, DUR);
    play.srcs.forEach((n) => { try { n?.stop(); } catch {} });
  }
  play = null;
  video.pause();
  $("#playBtn").innerHTML = icon("play");
}

function setWaiting(on, t = null) {
  waitingAt = on ? t : null;
  $("#buffering").hidden = !on;
  $("#playBtn").innerHTML = icon(on || play ? "pause" : "play");
}
function resumeIfReady() { if (waitingAt !== null && ready(waitingAt + 3)) startPlay(waitingAt); }

function tick(ts) {
  if (!play) return;
  const t = now();
  if (loop && selection && t >= selection.b) return startPlay(selection.a);
  if (t >= DUR) { stopPlay(); return setPlay(DUR); }
  if (!ready(t + 0.25)) { stopPlay(); setPlay(t); return setWaiting(true, t); }
  setPlay(t);
  if (src.has_video && !video.seeking && Math.abs(video.currentTime - (t + videoOffset())) > 0.3) video.currentTime = t + videoOffset();
  const x = t * pxPerSec();
  if (x > scroller.scrollLeft + scroller.clientWidth - 40 || x < scroller.scrollLeft) scroller.scrollLeft = x - 40;
  if (ts - lastSchedule > 200) { lastSchedule = ts; schedule(); }
  requestAnimationFrame(tick);
}

function togglePlay() {
  if (play || waitingAt !== null) { stopPlay(); return setWaiting(false); }
  startPlay(loop && selection && (playT < selection.a || playT >= selection.b) ? selection.a : playT);
}

function seek(t) {
  const wasPlaying = play || waitingAt !== null;
  stopPlay();
  setWaiting(false);
  setPlay(t);
  if (src?.has_video) video.currentTime = t + videoOffset();
  if (wasPlaying) startPlay(t);
}

$("#playBtn").addEventListener("click", togglePlay);
$("#toStart").addEventListener("click", () => { seek(selection ? selection.a : 0); scroller.scrollLeft = 0; });
$("#loopBtn").addEventListener("click", (e) => { loop = !loop; e.currentTarget.classList.toggle("on", loop); });

/* ═════════ Theme & resizable panels ═════════ */
function setTheme(theme, save) {
  document.documentElement.dataset.theme = theme;
  $("#themeBtn").innerHTML = icon(theme === "dark" ? "sun" : "moon");
  $("#themeBtn").title = theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
  if (save) localStorage.setItem("af-theme", theme);
  docsRendered = false; // diagrams re-render in the new theme
  if (!$("#docs").hidden) openDocs();
  if (src) { drawAll(); drawOverview(); }
}
$("#themeBtn").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true));

const layoutEl = $(".layout"), workspace = $(".workspace");
function makeResizer(handle, axis, apply, key) {
  const saved = localStorage.getItem(key);
  if (saved) apply(+saved);
  handle.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    handle.classList.add("active");
    document.body.classList.add("resizing", `resizing-${axis}`);
  });
  handle.addEventListener("pointermove", (e) => {
    if (!handle.hasPointerCapture(e.pointerId)) return;
    const size = axis === "x" ? e.clientX - layoutEl.getBoundingClientRect().left : e.clientY - workspace.getBoundingClientRect().top - 18;
    localStorage.setItem(key, apply(size));
  });
  handle.addEventListener("pointerup", () => {
    handle.classList.remove("active");
    document.body.classList.remove("resizing", "resizing-x", "resizing-y");
  });
  handle.addEventListener("dblclick", () => { apply(null); localStorage.removeItem(key); });
}
makeResizer($("#resizeX"), "x", (x) => {
  if (x === null) return layoutEl.style.removeProperty("--sidebar-w");
  const w = Math.round(clamp(x, 240, 440));
  layoutEl.style.setProperty("--sidebar-w", `${w}px`);
  return w;
}, "af-sidebar-w");
makeResizer($("#resizeY"), "y", (y) => {
  if (y === null) return workspace.style.removeProperty("--source-h");
  const h = Math.round(clamp(y, 140, workspace.getBoundingClientRect().height - 24 - 12 - 180));
  workspace.style.setProperty("--source-h", `${h}px`);
  return h;
}, "af-source-h");

/* ═════════ Hover help: bullet tips on [data-help] labels ═════════ */
const tip = $("#helpTip");
let tipTimer = null;
function hideTip() { clearTimeout(tipTimer); tip.hidden = true; tip._for = null; }
function showTip(el) {
  tip.innerHTML = `<ul>${el.dataset.help.split("|").map((l) => `<li>${esc(l)}</li>`).join("")}</ul>`;
  tip.hidden = false;
  tip._for = el;
  const r = el.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = `${clamp(r.left, 8, innerWidth - w - 8)}px`;
  tip.style.top = `${r.bottom + h + 16 > innerHeight ? r.top - h - 8 : r.bottom + 8}px`;
}
document.addEventListener("mouseover", (e) => {
  const el = e.target.closest?.("[data-help]");
  if (!el || el === tip._for) return;
  hideTip();
  tipTimer = setTimeout(() => showTip(el), 350);
});
document.addEventListener("mouseout", (e) => {
  const el = e.target.closest?.("[data-help]");
  if (el && !el.contains(e.relatedTarget)) hideTip();
});
["mousedown", "scroll", "wheel"].forEach((ev) => document.addEventListener(ev, hideTip, true));

/* ═════════ Models guide (Markdown + Mermaid, loaded on first open) ═════════ */
let docsRendered = false;
async function openDocs() {
  drawer(false);
  $("#docs").hidden = false;
  $("#scrim").hidden = false;
  if (docsRendered) return;
  const body = $("#docBody");
  const md = await fetch("static/models.md").then((r) => r.text()).catch(() => "");
  try {
    const { marked } = await import("https://cdn.jsdelivr.net/npm/marked@15/lib/marked.esm.js");
    body.innerHTML = marked.parse(md);
    const blocks = $$("code.language-mermaid", body);
    blocks.forEach((c) => Object.assign(c.parentElement, { className: "mermaid", textContent: c.textContent }));
    if (blocks.length) {
      const { default: mermaid } = await import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs");
      mermaid.initialize({ startOnLoad: false, theme: document.documentElement.dataset.theme === "dark" ? "dark" : "default" });
      await mermaid.run({ nodes: $$(".mermaid", body) });
    }
  } catch { // offline: show the Markdown as text
    body.innerHTML = `<pre>${esc(md || "The guide couldn't be loaded.")}</pre>`;
  }
  docsRendered = true;
}
function closeDocs() { $("#docs").hidden = true; if ($("#drawer").hidden) $("#scrim").hidden = true; }
$("#docsBtn").addEventListener("click", openDocs);
$("#docsClose").addEventListener("click", closeDocs);
$("#scrim").addEventListener("click", closeDocs);

/* ═════════ Demo mode (GitHub Pages): the UI with illustrative data, no server ═════════ */
function demoNotice() {
  const el = $("#toast");
  el.innerHTML = `This is an interface demo. To separate audio, run Audio Forensic on your computer or a server with a GPU: <a href="${REPO_URL}" target="_blank" rel="noopener">${REPO_URL.replace("https://", "")}</a>`;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), 9000);
}

const DEMO_DUR = 40;
const demoNoise = (i, k) => { const x = Math.sin(i * 12.9898 + k * 78.233) * 43758.5453; return x - Math.floor(x); };
function demoPeaks(kind) {
  const n = DEMO_DUR * PPS, out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / PPS;
    const voice = Math.sin(t * 0.8) > -0.25 ? (0.3 + 0.45 * Math.abs(Math.sin(t * 13))) * (0.6 + 0.4 * demoNoise(i, 1)) : 0.02;
    const dog = t > 4 && t % 7 < 0.45 ? 0.55 + 0.35 * demoNoise(i, 2) : 0.01;
    const bg = 0.05 + 0.04 * demoNoise(i, 3);
    const v = { voice, dog, rest: bg, rest2: Math.hypot(bg, dog * 0.1), orig: Math.hypot(voice, dog, bg) }[kind];
    out[i] = Math.min(1, v) * 255;
  }
  return btoa(String.fromCharCode(...out));
}
const DEMO_DEVICES = [
  { id: "cuda:0", name: "GPU", detail: "Unavailable in this demo", nv: null },
  { id: "cpu", name: "CPU", detail: "Unavailable in this demo", nv: null },
];
const DEMO_MODELS = [
  ["torch-fp16", "SAM-Audio small · FP16 (PyTorch)", ["0.6B params", "~2.5 GB GPU memory"], 5],
  ["large-fp8", "SAM-Audio large · FP8 (PyTorch)", ["3B params", "~4.3 GB GPU memory"], 5],
  ["large-fp16", "SAM-Audio large · FP16 (PyTorch)", ["3B params", "Weights streamed from RAM on 6 GB"], 20],
  ["q8", "SAM-Audio small · Q8 (GGUF)", ["0.6B params", "2.4 GB file"], 30],
  ["bf16", "SAM-Audio small · BF16 (GGUF)", ["0.6B params", "2.8 GB file"], 30],
  ["large-q8", "SAM-Audio large · Q8 (GGUF)", ["3B params", "5.0 GB file"], 30],
].map(([id, label, meta, chunk]) => ({ id, label, meta, chunk, available: true, note: "Demo: run the app locally to use this model", devices: DEMO_DEVICES }));
const DEMO_TRACKS = [
  [1, "original", "Original sound", "--wave-neutral", 1, null, "orig"],
  [2, "target", "Voice", "#4fb3a9", 0, 1, "voice"],
  [3, "target", "Dog barking", "#e07a5f", 0, 2, "dog"],
  [4, "residual", "Residual after “voice”", "--wave-residual", 1, 1, "rest2"],
  [5, "residual", "Residual after “dog barking”", "--wave-residual", 1, 2, "rest"],
];
function demoApi(path, opts) {
  const get = (opts.method || "GET") === "GET";
  if (get && path === "/api/models") return DEMO_MODELS;
  if (get && path === "/api/sources") return [{ id: 1, kind: "audio", name: "Demo session", duration_s: DEMO_DUR, isolated: 2, created_at: "2026-11-19 10:00", yt_mode: null }];
  if (get && path === "/api/sources/1") return {
    source: { id: 1, kind: "audio", name: "Demo session · illustrative waveforms, no audio", duration_s: DEMO_DUR, offset_s: 0, has_video: 0, splits: [12, 20], yt_mode: null, error: null },
    tracks: DEMO_TRACKS.map(([id, role, name, color, muted, job_id]) => ({ id, role, name, color, muted, solo: 0, gain: 100, job_id, status: "complete", covered: DEMO_DUR,
      edits: id === 3 ? [{ a: 12, b: 20, gain: 60, muted: false }] : [] })),
    jobs: [{ id: 1, model: "large-fp8", device: "cuda:0", prompt: "voice", status: "done", progress: 1, message: "0:40 separated in 89 s (example)", elapsed_s: 89, peak_vram_mib: 4319 },
           { id: 2, model: "torch-fp16", device: "cuda:0", prompt: "dog barking", status: "done", progress: 1, message: "0:40 separated in 9 s (example)", elapsed_s: 9, peak_vram_mib: 2407 }],
  };
  if (get && path.startsWith("/api/sources/1/segments")) {
    return path.endsWith("after=0") ? DEMO_TRACKS.map(([id, , , , , , kind]) => ({ id, track_id: id, start: 0, dur: DEMO_DUR, peaks: demoPeaks(kind) })) : [];
  }
  throw new Error(DEMO_TEXT);
}

/* ═════════ Keyboard & init ═════════ */
document.addEventListener("keydown", (e) => {
  const typing = e.target.matches("input:not([type=range]):not([type=checkbox]):not([type=radio]), textarea, select");
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); return isolate(); }
  if (e.key === "Escape") { closeMenus(); drawer(false); closeDocs(); hideTip(); setSelection(null); return; }
  if (typing || !src) return;
  if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  if (e.key === "s" || e.key === "S") $("#splitBtn").click();
  if (e.key === "Home") $("#toStart").click();
});

(async function init() {
  setTheme(document.documentElement.dataset.theme);
  renderSpans();
  try { await loadModels(); } catch (e) { toast(`Could not load models: ${e.message}`); }
  pollStats();
  if (DEMO) Object.assign($("#demoBadge"), { hidden: false, href: REPO_URL, title: "Interface demo: get the app on GitHub" });
  await openSource(DEMO ? 1 : +localStorage.getItem("af-source") || null);
})();
