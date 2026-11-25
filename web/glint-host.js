// Glint Browser Host:
//   - Mounts `glint.wasm` on `silk-ui-dom.js`
//   - IndexedDB (`glint-library-v1`) persistent library storage (folders, takes PCM, scores, stars, ratings, notes)
//   - Web Audio API (microphone capture, device listing, range/loop/speed playback, grain scrubbing, WAV download)
//   - WebGL2 3-tier spectrogram renderer (exact 1:1 port of Glint's WGSL shader)
import { mount } from "./silk-ui-dom.js";
import { resynthesizeVoice } from "./voice-synth.js";
import {
  computeAcousticVowelAnalysis,
  refineNeuralPhonemeSpans,
  buildIsolatedAudio,
  mapCompactTimeToSource,
} from "./vowel-isolator.js";
import { synthesizeKlattVowel, applyVoiceLabTransform } from "./voice-lab.js";

const decoder = new TextDecoder();
const encoder = new TextEncoder();

// ============================================================================
// 1. IndexedDB Persistent Library Store (synchronous in-memory mirror + async IDB commit)
// ============================================================================

const DB_NAME = "glint-library-v1";
const DB_VERSION = 1;

let db = null;
const lib = {
  // relPath -> { relPath, notes, createdAt }
  folders: new Map(),
  // fullPath -> { fullPath, relDir, name, displayName, sampleRate, samples: Float32Array, rating, starred, comment, scoresByRef, modified }
  clips: new Map(),
  rootNotes: "",
  scanRows: [],
  lastNotes: "",
};

function openIdb() {
  return new Promise((resolve) => {
    if (typeof indexedDB === "undefined") return resolve(null);
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const d = req.result;
      if (!d.objectStoreNames.contains("folders")) d.createObjectStore("folders", { keyPath: "relPath" });
      if (!d.objectStoreNames.contains("clips")) d.createObjectStore("clips", { keyPath: "fullPath" });
      if (!d.objectStoreNames.contains("meta")) d.createObjectStore("meta", { keyPath: "key" });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
  });
}

function idbGetAll(storeName) {
  return new Promise((resolve) => {
    if (!db) return resolve([]);
    try {
      const tx = db.transaction(storeName, "readonly");
      const req = tx.objectStore(storeName).getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => resolve([]);
    } catch {
      resolve([]);
    }
  });
}

function idbPut(storeName, value) {
  if (!db) return;
  try {
    const tx = db.transaction(storeName, "readwrite");
    tx.objectStore(storeName).put(value);
  } catch {}
}

function idbDelete(storeName, key) {
  if (!db) return;
  try {
    const tx = db.transaction(storeName, "readwrite");
    tx.objectStore(storeName).delete(key);
  } catch {}
}

async function seedDefaultDemosIfNeeded(seededVersion) {
  try {
    const resp = await fetch("./demos/manifest.json");
    if (!resp.ok) return;
    const manifest = await resp.json();
    const targetVer = manifest.version || 1;
    if (seededVersion >= targetVer) return;

    const folderRel = cleanRel(manifest.folder || "Harvard Sentences");
    const folderRec = {
      relPath: folderRel,
      createdAt: nowStamp(),
      notes: manifest.notes || "",
    };
    if (!lib.folders.has(folderRel)) {
      lib.folders.set(folderRel, folderRec);
      idbPut("folders", folderRec);
    }
    if (!lib.rootNotes && manifest.notes) {
      lib.rootNotes = manifest.notes;
      idbPut("meta", { key: "rootNotes", value: manifest.notes });
    }

    const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    const decodeCtx = OfflineCtx ? new OfflineCtx(1, 1, 44100) : ensureAudioCtx();

    for (const item of manifest.clips || []) {
      try {
        const r = await fetch(`./demos/${encodeURIComponent(item.file)}`);
        if (!r.ok) continue;
        const arr = await r.arrayBuffer();
        const audioBuf = await decodeCtx.decodeAudioData(arr);
        const mono = new Float32Array(audioBuf.length);
        const chCount = audioBuf.numberOfChannels;
        for (let ch = 0; ch < chCount; ch++) {
          const data = audioBuf.getChannelData(ch);
          for (let i = 0; i < mono.length; i++) mono[i] += data[i] / chCount;
        }
        const name = item.name || `${stripExt(item.file)}.wav`;
        const fullPath = `idb://${joinRel(folderRel, name)}`;
        if (lib.clips.has(fullPath)) continue;
        const rec = {
          fullPath,
          relDir: folderRel,
          name,
          displayName: stripExt(name),
          samples: mono,
          sampleRate: audioBuf.sampleRate,
          rating: 0,
          starred: false,
          comment: item.comment || "",
          scoresByRef: {},
          modified: nowStamp(),
        };
        lib.clips.set(fullPath, rec);
        idbPut("clips", rec);
      } catch (err) {
        console.warn("Failed to seed demo clip:", item.file, err);
      }
    }

    idbPut("meta", { key: "demosVersion", value: targetVer });
  } catch (err) {
    console.warn("Could not seed default demos:", err);
  }
}

async function hydrateLibrary() {
  db = await openIdb();
  const [folders, clips, metas] = await Promise.all([
    idbGetAll("folders"),
    idbGetAll("clips"),
    idbGetAll("meta"),
  ]);
  for (const f of folders) {
    if (f && typeof f.relPath === "string") lib.folders.set(f.relPath, f);
  }
  for (const c of clips) {
    if (c && typeof c.fullPath === "string") {
      if (!(c.samples instanceof Float32Array) && c.samples) {
        c.samples = new Float32Array(c.samples);
      }
      c.scoresByRef = c.scoresByRef || {};
      lib.clips.set(c.fullPath, c);
    }
  }
  let demosVersion = 0;
  for (const m of metas) {
    if (m.key === "rootNotes" && typeof m.value === "string") {
      lib.rootNotes = m.value;
    } else if (m.key === "demosVersion" && typeof m.value === "number") {
      demosVersion = m.value;
    }
  }
  await seedDefaultDemosIfNeeded(demosVersion);
}

function cleanRel(raw) {
  return (raw || "")
    .split("/")
    .filter((p) => p && p !== "." && p !== "..")
    .join("/");
}

function joinRel(parent, name) {
  const p = cleanRel(parent);
  return p ? `${p}/${name}` : name;
}

function stripExt(name) {
  return name.replace(/\.(wav|mp3|m4a|flac|aiff|ogg|webm)$/i, "");
}

function sanitizeName(raw) {
  return (raw || "").trim().replace(/[\/\\:*?"<>|]/g, "_");
}

function nowStamp() {
  return new Date().toISOString().slice(0, 16).replace("T", " ");
}

function scoreObjForRef(clip, activeRefPath) {
  const rec = activeRefPath && clip.scoresByRef ? clip.scoresByRef[activeRefPath] : null;
  if (!rec) return { overall: -1, pitch: -1, size: -1, weight: -1, vowel: -1, ver: 0 };
  return {
    overall: rec.overall ?? -1,
    pitch: rec.pitch ?? -1,
    size: rec.size ?? -1,
    weight: rec.weight ?? -1,
    vowel: rec.vowel ?? -1,
    ver: rec.ver ?? 0,
  };
}

function sortKeyForMode(clip, activeRefPath, sortMode) {
  const sc = scoreObjForRef(clip, activeRefPath);
  if (sortMode === 1) return sc.overall;
  if (sortMode === 2) return sc.pitch;
  if (sortMode === 3) return sc.size;
  if (sortMode === 4) return sc.weight;
  if (sortMode === 5) return sc.vowel;
  return -1;
}

function scanLibraryTree(expandedLines, notesDir, activeRefPath, sortMode) {
  const expandedSet = new Set(
    (expandedLines || "")
      .split("\n")
      .map((s) => cleanRel(s))
      .filter(Boolean),
  );
  const nd = cleanRel(notesDir);
  lib.lastNotes = nd ? (lib.folders.get(nd)?.notes || "") : lib.rootNotes;

  const rows = [];
  function walk(parentRel, depth) {
    // Direct child folders
    const childDirs = [];
    for (const f of lib.folders.values()) {
      const parts = f.relPath.split("/");
      const p = parts.length > 1 ? parts.slice(0, -1).join("/") : "";
      if (p === parentRel) childDirs.push(f);
    }
    childDirs.sort((a, b) => a.relPath.localeCompare(b.relPath));

    for (const d of childDirs) {
      const name = d.relPath.split("/").pop();
      const isExp = expandedSet.has(d.relPath);
      rows.push({
        isDir: 1,
        depth,
        expanded: isExp ? 1 : 0,
        rating: 0,
        starred: 0,
        score: -1,
        overall: -1,
        pitch: -1,
        size: -1,
        weight: -1,
        vowel: -1,
        ver: 0,
        name,
        relPath: d.relPath,
        parentRel,
        fullPath: `idb://${d.relPath}`,
        modified: d.createdAt || "",
        comment: "",
        displayName: name,
      });
      if (isExp) walk(d.relPath, depth + 1);
    }

    // Direct child clips
    const childClips = [];
    for (const c of lib.clips.values()) {
      if (cleanRel(c.relDir) === parentRel) childClips.push(c);
    }
    if (sortMode === 0) {
      childClips.sort((a, b) => a.name.localeCompare(b.name));
    } else {
      childClips.sort((a, b) => {
        const sb = sortKeyForMode(b, activeRefPath, sortMode);
        const sa = sortKeyForMode(a, activeRefPath, sortMode);
        if (sb !== sa) return sb - sa;
        return a.name.localeCompare(b.name);
      });
    }

    for (const c of childClips) {
      const sc = scoreObjForRef(c, activeRefPath);
      rows.push({
        isDir: 0,
        depth,
        expanded: 0,
        rating: c.rating || 0,
        starred: c.starred ? 1 : 0,
        score: sortKeyForMode(c, activeRefPath, sortMode),
        overall: sc.overall,
        pitch: sc.pitch,
        size: sc.size,
        weight: sc.weight,
        vowel: sc.vowel,
        ver: sc.ver,
        name: c.name,
        relPath: joinRel(parentRel, c.name),
        parentRel,
        fullPath: c.fullPath,
        modified: c.modified || "",
        comment: c.comment || "",
        displayName: c.displayName || stripExt(c.name),
      });
    }
  }
  walk("", 0);
  lib.scanRows = rows;
  return rows.length;
}

// ============================================================================
// 2. Web Audio API (Mic Capture, Playback, Grain Scrubbing, File Decoding, WAV Export)
// ============================================================================

let audioCtx = null;
function ensureAudioCtx() {
  if (!audioCtx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AC();
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume().catch(() => {});
  }
  return audioCtx;
}

const audioState = {
  inputs: [{ id: 1, deviceId: "", name: "Browser Microphone" }],
  outputs: [{ id: 1, deviceId: "", name: "System Output" }],
  selectedInputId: 1,
  selectedOutputId: 1,

  // Mic capture ring (hardware stream kept warm once opened to avoid CoreAudio/Bluetooth pops on repeat!)
  micRunning: false,
  micOpening: false,
  micStream: null,
  micStreamInputId: 0,
  micSource: null,
  micProc: null,
  micSilentGain: null,
  micRate: 48000,
  micLevelDb: -120,
  micError: "",
  micQueue: [],
  micQueuedSamples: 0,

  // Playback
  playing: false,
  finishedFlag: false,
  playNode: null,
  gainNode: null,
  playStartCtxTime: 0,
  playStartSec: 0,
  playEndSec: 0,
  playLoop: false,
  playPositionSec: 0,
  playTimeMap: null,
  volume: 0.85,
  speed: 1.0,

  // Scrubbing
  scrubSamples: null,
  scrubRate: 48000,
  scrubLastMs: 0,

  // File decode cache
  decodedFiles: new Map(),
  loaded: null,
};

async function enumerateBrowserDevices(promptPermission = false) {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    let list = await navigator.mediaDevices.enumerateDevices();
    const hasLabels = list.some((d) => d.kind === "audioinput" && d.label);
    if (promptPermission && !hasLabels && navigator.mediaDevices?.getUserMedia) {
      try {
        const tmp = await navigator.mediaDevices.getUserMedia({ audio: true });
        tmp.getTracks().forEach((t) => t.stop());
        list = await navigator.mediaDevices.enumerateDevices();
      } catch {}
    }
    const ins = [];
    const outs = [];
    let idxIn = 1;
    let idxOut = 1;
    for (const d of list) {
      if (d.kind === "audioinput") {
        ins.push({ id: idxIn++, deviceId: d.deviceId, name: d.label || `Microphone ${ins.length + 1}` });
      } else if (d.kind === "audiooutput") {
        outs.push({ id: idxOut++, deviceId: d.deviceId, name: d.label || `Output ${outs.length + 1}` });
      }
    }
    if (ins.length) audioState.inputs = ins;
    if (outs.length) audioState.outputs = outs;
  } catch {}
}

function isMicHardwareLive() {
  if (!audioState.micStream || !audioState.micProc) return false;
  if (audioState.micStream.active === false) return false;
  const tracks = audioState.micStream.getAudioTracks();
  return tracks.length > 0 && tracks.every((t) => t.readyState === "live");
}

function releaseMicHardware() {
  audioState.micRunning = false;
  audioState.micOpening = false;
  audioState.micLevelDb = -120;
  audioState.micQueue = [];
  audioState.micQueuedSamples = 0;
  try { audioState.micProc?.disconnect(); } catch {}
  try { audioState.micSource?.disconnect(); } catch {}
  try { audioState.micSilentGain?.disconnect(); } catch {}
  try { audioState.micStream?.getTracks().forEach((t) => t.stop()); } catch {}
  audioState.micProc = null;
  audioState.micSource = null;
  audioState.micSilentGain = null;
  audioState.micStream = null;
  audioState.micStreamInputId = 0;
}

function startMicAsync(preferredRate) {
  audioState.micError = "";
  if (!navigator.mediaDevices?.getUserMedia) {
    audioState.micError = "Microphone capture requires a browser with getUserMedia (HTTPS or localhost).";
    return 0;
  }
  const ctx = ensureAudioCtx();
  audioState.micRate = ctx.sampleRate || preferredRate || 48000;
  audioState.micRunning = true;
  audioState.micQueue = [];
  audioState.micQueuedSamples = 0;

  // Fast path: mic hardware is already warm for this input -> zero restart, zero pops, instant capture!
  if (isMicHardwareLive() && audioState.micStreamInputId === audioState.selectedInputId) {
    return 1;
  }

  // If device changed or previous stream ended, cleanly release old hardware before opening new stream
  if (audioState.micStream && (!isMicHardwareLive() || audioState.micStreamInputId !== audioState.selectedInputId)) {
    releaseMicHardware();
    audioState.micRunning = true;
  }

  if (audioState.micOpening) {
    return 1;
  }
  audioState.micOpening = true;

  const chosen = audioState.inputs.find((d) => d.id === audioState.selectedInputId);
  const desiredDevId = chosen?.deviceId || "";
  const openedInputId = audioState.selectedInputId;
  const constraints = {
    audio: {
      deviceId: desiredDevId ? { ideal: desiredDevId } : undefined,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  };

  navigator.mediaDevices
    .getUserMedia(constraints)
    .then((stream) => {
      audioState.micOpening = false;
      audioState.micStream = stream;
      audioState.micStreamInputId = openedInputId;
      audioState.micRate = ctx.sampleRate;
      const src = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(2048, 1, 1);
      const silentGain = ctx.createGain();
      silentGain.gain.value = 0.0;
      proc.onaudioprocess = (ev) => {
        // Keep output strictly silent
        const outCh = ev.outputBuffer.getChannelData(0);
        outCh.fill(0.0);
        // When logically stopped, simply ignore incoming frames without tearing down hardware!
        if (!audioState.micRunning) return;
        const ch = ev.inputBuffer.getChannelData(0);
        const copy = new Float32Array(ch.length);
        let sumSq = 0;
        for (let i = 0; i < ch.length; i++) {
          const v = ch[i];
          copy[i] = v;
          sumSq += v * v;
        }
        const rms = Math.sqrt(sumSq / Math.max(1, ch.length));
        audioState.micLevelDb = rms > 1e-6 ? 20 * Math.log10(rms) : -120;
        audioState.micQueue.push(copy);
        audioState.micQueuedSamples += copy.length;
      };
      src.connect(proc);
      proc.connect(silentGain);
      silentGain.connect(ctx.destination);
      audioState.micSource = src;
      audioState.micProc = proc;
      audioState.micSilentGain = silentGain;
      enumerateBrowserDevices().then(() => {
        if (mountedPage) mountedPage.post("refresh_devices", "");
      });
    })
    .catch((err) => {
      audioState.micOpening = false;
      audioState.micRunning = false;
      audioState.micError = err?.message || "Microphone permission denied.";
    });
  return 1;
}

function stopMic() {
  // Soft-gate: ignore incoming mic audio without stopping the MediaStreamTrack,
  // avoiding OS/Bluetooth audio restarts, pops, and silences between repeat rounds!
  audioState.micRunning = false;
  audioState.micLevelDb = -120;
  audioState.micQueue = [];
  audioState.micQueuedSamples = 0;
}

window.addEventListener("pagehide", releaseMicHardware);

function stopPlayback() {
  if (audioState.playNode) {
    try {
      audioState.playNode.onended = null;
      audioState.playNode.stop();
      audioState.playNode.disconnect();
    } catch {}
    audioState.playNode = null;
  }
  audioState.playing = false;
}

function startPlayback(samples, sampleRate, startSec, endSec, loop, timeMap = null) {
  stopPlayback();
  if (!samples || samples.length === 0 || sampleRate <= 0) return 0;
  const ctx = ensureAudioCtx();
  const dur = samples.length / sampleRate;
  const s0 = Math.max(0, Math.min(dur, startSec));
  const s1 = Math.max(s0 + 0.01, Math.min(dur, endSec > s0 ? endSec : dur));

  const buf = ctx.createBuffer(1, samples.length, Math.round(sampleRate));
  buf.copyToChannel(samples, 0);

  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.playbackRate.value = audioState.speed;
  src.loop = !!loop;
  if (loop) {
    src.loopStart = s0;
    src.loopEnd = s1;
  }
  const gain = ctx.createGain();
  gain.gain.value = audioState.volume;
  src.connect(gain);
  gain.connect(ctx.destination);

  audioState.playNode = src;
  audioState.gainNode = gain;
  audioState.playing = true;
  audioState.finishedFlag = false;
  audioState.playStartCtxTime = ctx.currentTime;
  audioState.playStartSec = s0;
  audioState.playEndSec = s1;
  audioState.playLoop = !!loop;
  audioState.playTimeMap = timeMap || null;
  audioState.playPositionSec = timeMap ? mapCompactTimeToSource(timeMap, s0) : s0;

  src.onended = () => {
    if (audioState.playNode === src && !audioState.playLoop) {
      audioState.playing = false;
      audioState.finishedFlag = true;
      audioState.playPositionSec = audioState.playTimeMap
        ? mapCompactTimeToSource(audioState.playTimeMap, audioState.playEndSec)
        : audioState.playEndSec;
    }
  };

  if (loop) {
    src.start(0, s0);
  } else {
    src.start(0, s0, s1 - s0);
  }
  return 1;
}

function currentPlayPosition() {
  if (!audioState.playing || !audioCtx) return audioState.playPositionSec;
  const elapsed = (audioCtx.currentTime - audioState.playStartCtxTime) * audioState.speed;
  const span = Math.max(0.01, audioState.playEndSec - audioState.playStartSec);
  let rawPos = audioState.playStartSec + elapsed;
  if (audioState.playLoop) {
    rawPos = audioState.playStartSec + (elapsed % span);
  } else if (rawPos >= audioState.playEndSec) {
    rawPos = audioState.playEndSec;
  }
  return audioState.playTimeMap ? mapCompactTimeToSource(audioState.playTimeMap, rawPos) : rawPos;
}

function playScrubGrain(centerSec) {
  const nowMs = performance.now();
  if (nowMs - audioState.scrubLastMs < 48) return;
  audioState.scrubLastMs = nowMs;
  if (!audioState.scrubSamples || audioState.scrubSamples.length < 64) return;
  const ctx = ensureAudioCtx();
  const sr = audioState.scrubRate;
  const grainLen = Math.min(audioState.scrubSamples.length, Math.floor(sr * 0.055));
  const centre = Math.floor(centerSec * sr);
  const start = Math.max(0, Math.min(audioState.scrubSamples.length - grainLen, centre - (grainLen >> 1)));
  const buf = ctx.createBuffer(1, grainLen, Math.round(sr));
  const out = buf.getChannelData(0);
  for (let i = 0; i < grainLen; i++) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / grainLen);
    out[i] = audioState.scrubSamples[start + i] * w;
  }
  const src = ctx.createBufferSource();
  src.buffer = buf;
  const gain = ctx.createGain();
  gain.gain.value = audioState.volume * 0.6;
  src.connect(gain);
  gain.connect(ctx.destination);
  src.start();
}

function encodeWav16(samples, sampleRate) {
  const numSamples = samples.length;
  const dataBytes = numSamples * 2;
  const buf = new ArrayBuffer(44 + dataBytes);
  const v = new DataView(buf);
  const writeStr = (off, s) => {
    for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  v.setUint32(4, 36 + dataBytes, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  writeStr(36, "data");
  v.setUint32(40, dataBytes, true);
  for (let i = 0; i < numSamples; i++) {
    const c = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, c < 0 ? Math.round(c * 32768) : Math.round(c * 32767), true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

function triggerWavDownload(name, samples, sampleRate) {
  const blob = encodeWav16(samples, sampleRate);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const clean = (name || "glint-take").replace(/\.wav$/i, "") + ".wav";
  a.href = url;
  a.download = clean;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

async function decodeBrowserFile(file) {
  const arr = await file.arrayBuffer();
  const ctx = ensureAudioCtx();
  const decoded = await ctx.decodeAudioData(arr.slice(0));
  const len = decoded.length;
  const chCount = decoded.numberOfChannels;
  const mono = new Float32Array(len);
  for (let c = 0; c < chCount; c++) {
    const ch = decoded.getChannelData(c);
    for (let i = 0; i < len; i++) mono[i] += ch[i] / chCount;
  }
  const key = `__file__/${Date.now()}/${file.name}`;
  audioState.decodedFiles.set(key, { name: file.name, sampleRate: decoded.sampleRate, samples: mono });
  return key;
}

// ============================================================================
// 3. WebGL2 3-Tier Spectrogram Renderer (1:1 GLSL ES 3.00 port of SPECTROGRAM_WGSL)
// ============================================================================

const VS_SRC = `#version 300 es
out vec2 vUv;
void main() {
  vec2 pos = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
  gl_Position = vec4(pos, 0.0, 1.0);
  vUv = (pos + 1.0) * 0.5;
}`;

const FS_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform float u[25];
uniform sampler2D coarseTex;
uniform sampler2D sharpTex;
uniform sampler2D lutTex;
uniform sampler2D liveTex;

float normToFreq(float norm) {
  if (u[13] > 0.5) {
    float lo = log(max(u[2], u[18]));
    float hi = log(max(u[3], u[18] * 2.0));
    return exp(lo + norm * (hi - lo));
  }
  return u[2] + norm * (u[3] - u[2]);
}

vec2 sampleTier(sampler2D tex, float time, float freq, float tierT0, float tierT1, float tierMaxHz, float uScale, float uOffset) {
  float span = tierT1 - tierT0;
  float x = span > 0.0 ? (time - tierT0) / span : 0.0;
  float y = freq / max(tierMaxHz, 1.0);
  ivec2 sz = textureSize(tex, 0);
  float halfTexel = 0.5 / max(float(sz.y), 1.0);
  float uTex = clamp(x, 0.0, 1.0) * uScale + uOffset;
  float vTex = clamp(1.0 - y - halfTexel, halfTexel, 1.0 - halfTexel);
  float v = texture(tex, vec2(uTex, vTex)).r;
  bool inside = span > 0.0 && time >= tierT0 && time <= tierT1 && y <= 1.0 && y >= 0.0;
  return vec2(v, inside ? 1.0 : 0.0);
}

void main() {
  float time = u[0] + vUv.x * (u[1] - u[0]);
  float freq = normToFreq(vUv.y);
  vec2 s = sampleTier(sharpTex, time, freq, u[8], u[9], u[12], u[23], u[24]);
  vec2 l = sampleTier(liveTex, time, freq, u[6], u[7], u[11], u[21], u[22]);
  vec2 c = sampleTier(coarseTex, time, freq, u[4], u[5], u[10], u[19], u[20]);
  bool useSharp = u[15] > 0.5 && s.y > 0.5;
  bool useLive = !useSharp && u[14] > 0.5 && l.y > 0.5;
  float val = c.x;
  float have = c.y;
  if (useLive) { val = l.x; have = 1.0; }
  if (useSharp) { val = s.x; have = 1.0; }
  float v = clamp((val + u[16]) * u[17], 0.0, 1.0);
  vec3 rgb = texture(lutTex, vec2(v, 0.5)).rgb;
  outColor = vec4(rgb * step(0.5, have), 1.0);
}`;

class WebGLSpectrogram {
  constructor() {
    this.canvas = document.createElement("canvas");
    this.canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block;pointer-events:none;";
    this.gl = this.canvas.getContext("webgl2", { alpha: false, antialias: false, preserveDrawingBuffer: true });
    this.lastUniforms = null;
    this.pendingLut = null;
    this.tiers = [null, null, null];
    this.lutTex = null;
    if (this.gl) this.initGl();
  }

  initGl() {
    const gl = this.gl;
    const compile = (type, src) => {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        console.error("Glint WebGL2 shader error:", gl.getShaderInfoLog(sh));
      }
      return sh;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS_SRC));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS_SRC));
    gl.linkProgram(prog);
    this.prog = prog;
    this.uLoc = gl.getUniformLocation(prog, "u[0]");
    this.coarseLoc = gl.getUniformLocation(prog, "coarseTex");
    this.sharpLoc = gl.getUniformLocation(prog, "sharpTex");
    this.lutLoc = gl.getUniformLocation(prog, "lutTex");
    this.liveLoc = gl.getUniformLocation(prog, "liveTex");
    this.vao = gl.createVertexArray();

    const makeTex = () => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
      return t;
    };
    this.tiers = [makeTex(), makeTex(), makeTex()];
    this.lutTex = makeTex();
  }

  attachToHost() {
    const host = document.querySelector('[data-testid="spectrogram-stage"]');
    if (!host) return false;
    if (this.canvas.parentElement !== host) {
      host.appendChild(this.canvas);
    }
    const rect = host.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(2, Math.round(rect.width * dpr));
    const h = Math.max(2, Math.round(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    return true;
  }

  setLut(bytes) {
    if (!this.gl) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, bytes.length / 4, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
  }

  allocTier(tier, w, h) {
    if (!this.gl || tier > 2) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tiers[tier]);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, Math.max(1, w), Math.max(1, h), 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  }

  writeTier(tier, x, w, h, bytes) {
    if (!this.gl || tier > 2 || w <= 0 || h <= 0) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tiers[tier]);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, x, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
  }

  clearTier(tier) {
    if (!this.gl || tier > 2) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tiers[tier]);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
  }

  draw(uniforms) {
    this.lastUniforms = new Float32Array(uniforms);
    if (!this.gl) return;
    this.attachToHost();
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);
    gl.uniform1fv(this.uLoc, this.lastUniforms);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.tiers[0]); gl.uniform1i(this.coarseLoc, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.tiers[2]); gl.uniform1i(this.sharpLoc, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.lutTex); gl.uniform1i(this.lutLoc, 2);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.tiers[1]); gl.uniform1i(this.liveLoc, 3);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}

// ============================================================================
// 4. WASM Memory Helpers & `glint#*` Import Table
// ============================================================================

let mountedPage = null;
let wasmMemory = null;
const specGl = new WebGLSpectrogram();

function memBuf() {
  return (wasmMemory || mountedPage.exports.memory).buffer;
}

function readStr(ptr, len) {
  if (!ptr || len <= 0) return "";
  return decoder.decode(new Uint8Array(memBuf(), ptr, len));
}

function writeStr(str, outPtr, cap) {
  if (!outPtr || cap <= 0) return 0;
  const bytes = encoder.encode(str || "");
  const n = Math.min(bytes.length, cap - 1);
  const dst = new Uint8Array(memBuf(), outPtr, cap);
  dst.set(bytes.subarray(0, n), 0);
  dst[n] = 0;
  return n;
}

function writeI32(ptr, val) {
  if (!ptr) return;
  new Int32Array(memBuf(), ptr, 1)[0] = val | 0;
}

function readF32Copy(ptr, count) {
  if (!ptr || count <= 0) return new Float32Array(0);
  return new Float32Array(new Float32Array(memBuf(), ptr, count));
}

const glintImports = {
  // Audio devices
  ov_audio_refresh_devices: () => {
    const prevSig = JSON.stringify([audioState.inputs, audioState.outputs]);
    enumerateBrowserDevices(true).then(() => {
      const nextSig = JSON.stringify([audioState.inputs, audioState.outputs]);
      if (nextSig !== prevSig && mountedPage) {
        mountedPage.post("refresh_devices", "");
      }
    });
    return 1;
  },
  ov_audio_input_count: () => audioState.inputs.length,
  ov_audio_input_id: (idx) => audioState.inputs[idx]?.id ?? 0,
  ov_audio_input_name: (idx, out, cap) => writeStr(audioState.inputs[idx]?.name || "", out, cap),
  ov_audio_output_count: () => audioState.outputs.length,
  ov_audio_output_id: (idx) => audioState.outputs[idx]?.id ?? 0,
  ov_audio_output_name: (idx, out, cap) => writeStr(audioState.outputs[idx]?.name || "", out, cap),
  ov_audio_get_input_device: () => audioState.selectedInputId,
  ov_audio_get_output_device: () => audioState.selectedOutputId,
  ov_audio_set_input_device: (devId) => {
    const nextId = devId >>> 0;
    if (nextId !== audioState.selectedInputId) {
      const wasRunning = audioState.micRunning;
      audioState.selectedInputId = nextId;
      releaseMicHardware();
      if (wasRunning) startMicAsync(audioState.micRate);
    }
    return 1;
  },
  ov_audio_set_output_device: (devId) => {
    audioState.selectedOutputId = devId >>> 0;
    const chosen = audioState.outputs.find((d) => d.id === audioState.selectedOutputId);
    if (audioCtx?.setSinkId && chosen?.deviceId) {
      audioCtx.setSinkId(chosen.deviceId).catch(() => {});
    }
    return 1;
  },

  // Mic capture
  ov_mic_start_rate: (preferredRate) => startMicAsync(preferredRate),
  ov_mic_start: () => startMicAsync(48000),
  ov_mic_stop: () => stopMic(),
  ov_mic_running: () => (audioState.micRunning ? 1 : 0),
  ov_mic_sample_rate: () => audioState.micRate,
  ov_mic_level_db: () => audioState.micLevelDb,
  ov_mic_available: () => audioState.micQueuedSamples,
  ov_mic_drain: (outPtr, maxCount) => {
    if (!outPtr || maxCount <= 0 || audioState.micQueuedSamples <= 0) return 0;
    const dst = new Float32Array(memBuf(), outPtr, maxCount);
    let written = 0;
    while (written < maxCount && audioState.micQueue.length > 0) {
      const chunk = audioState.micQueue[0];
      const need = maxCount - written;
      if (chunk.length <= need) {
        dst.set(chunk, written);
        written += chunk.length;
        audioState.micQueue.shift();
      } else {
        dst.set(chunk.subarray(0, need), written);
        audioState.micQueue[0] = chunk.subarray(need);
        written += need;
      }
    }
    audioState.micQueuedSamples = Math.max(0, audioState.micQueuedSamples - written);
    return written;
  },
  ov_mic_error: (out, cap) => writeStr(audioState.micError, out, cap),

  // Load audio from IndexedDB library or decoded browser file
  ov_load_file: (pathPtr, len) => {
    const p = readStr(pathPtr, len);
    const clip = lib.clips.get(p);
    if (clip && clip.samples) {
      audioState.loaded = { name: clip.displayName || stripExt(clip.name), sampleRate: clip.sampleRate, samples: clip.samples };
      return 1;
    }
    const dec = audioState.decodedFiles.get(p);
    if (dec && dec.samples) {
      audioState.loaded = dec;
      return 1;
    }
    return 0;
  },
  ov_loaded_count: () => audioState.loaded?.samples?.length || 0,
  ov_loaded_rate: () => audioState.loaded?.sampleRate || 48000,
  ov_loaded_copy: (outPtr, maxCount) => {
    if (!audioState.loaded?.samples || !outPtr || maxCount <= 0) return 0;
    const n = Math.min(maxCount, audioState.loaded.samples.length);
    new Float32Array(memBuf(), outPtr, n).set(audioState.loaded.samples.subarray(0, n));
    return n;
  },
  ov_loaded_name: (out, cap) => writeStr(audioState.loaded?.name || "Audio", out, cap),

  // Playback & Scrubbing
  ov_play_range: (ptr, count, sr, startSec, endSec, loop) => {
    const samples = readF32Copy(ptr, count);
    return startPlayback(samples, sr, startSec, endSec, loop !== 0);
  },
  ov_play_stop: () => stopPlayback(),
  ov_play_is_playing: () => (audioState.playing ? 1 : 0),
  ov_play_take_finished: () => {
    const f = audioState.finishedFlag ? 1 : 0;
    audioState.finishedFlag = false;
    return f;
  },
  ov_play_position: () => currentPlayPosition(),
  ov_play_set_volume: (v) => {
    audioState.volume = Math.max(0, Math.min(1, v));
    if (audioState.gainNode) audioState.gainNode.gain.value = audioState.volume;
  },
  ov_play_set_speed: (s) => {
    audioState.speed = Math.max(0.25, Math.min(2.0, s));
    if (audioState.playNode) audioState.playNode.playbackRate.value = audioState.speed;
  },
  ov_play_get_speed: () => audioState.speed,
  ov_scrub_begin: (ptr, count, sr, centerSec) => {
    audioState.scrubSamples = readF32Copy(ptr, count);
    audioState.scrubRate = sr;
    playScrubGrain(centerSec);
  },
  ov_scrub_move: (centerSec) => playScrubGrain(centerSec),
  ov_scrub_stop: () => {
    audioState.scrubSamples = null;
  },
  ov_export_wav: (pathPtr, pathLen, samplesPtr, count, sr) => {
    const name = readStr(pathPtr, pathLen) || "glint-take.wav";
    const samples = readF32Copy(samplesPtr, count);
    triggerWavDownload(name, samples, sr);
    return 1;
  },

  // Browser Open File / WAV Download triggers
  open_file_picker: () => {
    const el = document.getElementById("glint-file-input");
    if (el) {
      el.value = "";
      el.click();
    }
  },
  export_wav_download: (namePtr, nameLen, samplesPtr, count, sr) => {
    const name = readStr(namePtr, nameLen) || "glint-take";
    const samples = readF32Copy(samplesPtr, count);
    triggerWavDownload(name, samples, sr);
  },

  // IndexedDB-backed Library Tree
  ov_lib_scan_tree: (expPtr, expLen, notesPtr, notesLen, refPtr, refLen, sortMode) => {
    return scanLibraryTree(
      readStr(expPtr, expLen),
      readStr(notesPtr, notesLen),
      readStr(refPtr, refLen),
      sortMode,
    );
  },
  ov_lib_tree_meta: (idx, oDir, oDepth, oExp, oRating, oStar, oScore) => {
    const r = lib.scanRows[idx];
    if (!r) return;
    writeI32(oDir, r.isDir);
    writeI32(oDepth, r.depth);
    writeI32(oExp, r.expanded);
    writeI32(oRating, r.rating);
    writeI32(oStar, r.starred);
    writeI32(oScore, r.score);
  },
  ov_lib_tree_scores: (idx, oOv, oPi, oSz, oWt, oVw, oVer) => {
    const r = lib.scanRows[idx];
    if (!r) return;
    writeI32(oOv, r.overall);
    writeI32(oPi, r.pitch);
    writeI32(oSz, r.size);
    writeI32(oWt, r.weight);
    writeI32(oVw, r.vowel);
    writeI32(oVer, r.ver);
  },
  ov_lib_tree_str: (idx, field, out, cap) => {
    const r = lib.scanRows[idx];
    if (!r) return 0;
    const vals = [r.name, r.relPath, r.parentRel, r.fullPath, r.modified, r.comment, r.displayName];
    return writeStr(vals[field] || "", out, cap);
  },
  ov_lib_create_folder: (pPtr, pLen, nPtr, nLen, outRel, outCap) => {
    const parentRel = cleanRel(readStr(pPtr, pLen));
    const folderName = sanitizeName(readStr(nPtr, nLen));
    if (!folderName) return 0;
    const relPath = joinRel(parentRel, folderName);
    if (!lib.folders.has(relPath)) {
      const rec = { relPath, notes: "", createdAt: nowStamp() };
      lib.folders.set(relPath, rec);
      idbPut("folders", rec);
    }
    writeStr(relPath, outRel, outCap);
    return 1;
  },
  ov_lib_reveal_dir: () => {},
  ov_lib_reveal_entry: () => {},
  ov_lib_download_entry: (pPtr, pLen, nPtr, nLen) => {
    const parentRel = cleanRel(readStr(pPtr, pLen));
    const name = readStr(nPtr, nLen);
    const fp = `idb://${joinRel(parentRel, name)}`;
    const clip = lib.clips.get(fp);
    if (!clip || !clip.samples || clip.samples.length === 0) return 0;
    triggerWavDownload(clip.displayName || clip.name, clip.samples, clip.sampleRate || 48000);
    return 1;
  },
  ov_lib_share_entry: (pPtr, pLen, nPtr, nLen) => {
    const parentRel = cleanRel(readStr(pPtr, pLen));
    const name = readStr(nPtr, nLen);
    const fp = `idb://${joinRel(parentRel, name)}`;
    const clip = lib.clips.get(fp);
    if (!clip || !clip.samples || clip.samples.length === 0) return 0;
    const clean = (clip.displayName || clip.name || "prism-take").replace(/\.wav$/i, "") + ".wav";
    const blob = encodeWav16(clip.samples, clip.sampleRate || 48000);
    const file = new File([blob], clean, { type: "audio/wav" });
    if (typeof navigator !== "undefined" && typeof navigator.share === "function") {
      const canShareFiles = typeof navigator.canShare === "function" && navigator.canShare({ files: [file] });
      if (canShareFiles) {
        navigator.share({ title: clip.displayName || "Prism take", files: [file] }).catch((err) => {
          if (err && err.name !== "AbortError") triggerWavDownload(clean, clip.samples, clip.sampleRate || 48000);
        });
        return 1;
      }
    }
    triggerWavDownload(clean, clip.samples, clip.sampleRate || 48000);
    return 1;
  },
  ov_lib_save_wav: (dirPtr, dirLen, namePtr, nameLen, overwrite, samplesPtr, count, sr, score, outFn, fnCap, outFp, fpCap) => {
    const relDir = cleanRel(readStr(dirPtr, dirLen));
    if (relDir && !lib.folders.has(relDir)) {
      const f = { relPath: relDir, notes: "", createdAt: nowStamp() };
      lib.folders.set(relDir, f);
      idbPut("folders", f);
    }
    let base = sanitizeName(stripExt(readStr(namePtr, nameLen))) || "Take";
    let fileName = `${base}.wav`;
    let fullPath = `idb://${joinRel(relDir, fileName)}`;
    if (!overwrite) {
      let n = 1;
      while (lib.clips.has(fullPath)) {
        n++;
        fileName = `${base} ${String(n).padStart(2, "0")}.wav`;
        fullPath = `idb://${joinRel(relDir, fileName)}`;
      }
    }
    const existing = lib.clips.get(fullPath);
    const samples = readF32Copy(samplesPtr, count);
    const rec = {
      fullPath,
      relDir,
      name: fileName,
      displayName: stripExt(fileName),
      sampleRate: sr,
      samples,
      rating: existing?.rating || 0,
      starred: existing?.starred || false,
      comment: existing?.comment || "",
      scoresByRef: existing?.scoresByRef || {},
      modified: nowStamp(),
    };
    lib.clips.set(fullPath, rec);
    idbPut("clips", rec);
    writeStr(fileName, outFn, fnCap);
    writeStr(fullPath, outFp, fpCap);
    return 1;
  },
  ov_lib_rename_entry: (pPtr, pLen, oPtr, oLen, nPtr, nLen, isDir, outName, nCap, outRel, rCap, outFp, fpCap) => {
    const parentRel = cleanRel(readStr(pPtr, pLen));
    const oldName = readStr(oPtr, oLen);
    const newRaw = sanitizeName(readStr(nPtr, nLen));
    if (!oldName || !newRaw) return 0;

    if (isDir) {
      const oldRel = joinRel(parentRel, oldName);
      const newRel = joinRel(parentRel, newRaw);
      if (oldRel === newRel || lib.folders.has(newRel) || !lib.folders.has(oldRel)) return 0;
      const toMoveFolders = [...lib.folders.values()].filter((f) => f.relPath === oldRel || f.relPath.startsWith(oldRel + "/"));
      for (const f of toMoveFolders) {
        lib.folders.delete(f.relPath);
        idbDelete("folders", f.relPath);
        const updated = { ...f, relPath: newRel + f.relPath.slice(oldRel.length) };
        lib.folders.set(updated.relPath, updated);
        idbPut("folders", updated);
      }
      const toMoveClips = [...lib.clips.values()].filter((c) => c.relDir === oldRel || c.relDir.startsWith(oldRel + "/"));
      for (const c of toMoveClips) {
        lib.clips.delete(c.fullPath);
        idbDelete("clips", c.fullPath);
        const newDir = newRel + c.relDir.slice(oldRel.length);
        const newFp = `idb://${joinRel(newDir, c.name)}`;
        const updated = { ...c, relDir: newDir, fullPath: newFp };
        lib.clips.set(newFp, updated);
        idbPut("clips", updated);
      }
      writeStr(newRaw, outName, nCap);
      writeStr(newRel, outRel, rCap);
      writeStr(`idb://${newRel}`, outFp, fpCap);
      return 1;
    }

    const oldFp = `idb://${joinRel(parentRel, oldName)}`;
    const clip = lib.clips.get(oldFp);
    if (!clip) return 0;
    const newName = newRaw.endsWith(".wav") ? newRaw : `${newRaw}.wav`;
    const newRel = joinRel(parentRel, newName);
    const newFp = `idb://${newRel}`;
    if (newFp !== oldFp && lib.clips.has(newFp)) return 0;
    lib.clips.delete(oldFp);
    idbDelete("clips", oldFp);
    const updated = { ...clip, name: newName, displayName: stripExt(newName), fullPath: newFp, modified: nowStamp() };
    lib.clips.set(newFp, updated);
    idbPut("clips", updated);
    writeStr(newName, outName, nCap);
    writeStr(newRel, outRel, rCap);
    writeStr(newFp, outFp, fpCap);
    return 1;
  },
  ov_lib_move_entry: (spPtr, spLen, nPtr, nLen, dstPtr, dstLen, isDir, outRel, rCap, outFp, fpCap) => {
    const srcParent = cleanRel(readStr(spPtr, spLen));
    const name = readStr(nPtr, nLen);
    const dstFolder = cleanRel(readStr(dstPtr, dstLen));
    if (srcParent === dstFolder) return 0;

    if (isDir) {
      const oldRel = joinRel(srcParent, name);
      const newRel = joinRel(dstFolder, name);
      if (newRel === oldRel || newRel.startsWith(oldRel + "/") || lib.folders.has(newRel)) return 0;
      const toMoveFolders = [...lib.folders.values()].filter((f) => f.relPath === oldRel || f.relPath.startsWith(oldRel + "/"));
      for (const f of toMoveFolders) {
        lib.folders.delete(f.relPath);
        idbDelete("folders", f.relPath);
        const updated = { ...f, relPath: newRel + f.relPath.slice(oldRel.length) };
        lib.folders.set(updated.relPath, updated);
        idbPut("folders", updated);
      }
      const toMoveClips = [...lib.clips.values()].filter((c) => c.relDir === oldRel || c.relDir.startsWith(oldRel + "/"));
      for (const c of toMoveClips) {
        lib.clips.delete(c.fullPath);
        idbDelete("clips", c.fullPath);
        const newDir = newRel + c.relDir.slice(oldRel.length);
        const newFp = `idb://${joinRel(newDir, c.name)}`;
        const updated = { ...c, relDir: newDir, fullPath: newFp };
        lib.clips.set(newFp, updated);
        idbPut("clips", updated);
      }
      writeStr(newRel, outRel, rCap);
      writeStr(`idb://${newRel}`, outFp, fpCap);
      return 1;
    }

    const oldFp = `idb://${joinRel(srcParent, name)}`;
    const clip = lib.clips.get(oldFp);
    const newRel = joinRel(dstFolder, name);
    const newFp = `idb://${newRel}`;
    if (!clip || lib.clips.has(newFp)) return 0;
    lib.clips.delete(oldFp);
    idbDelete("clips", oldFp);
    const updated = { ...clip, relDir: dstFolder, fullPath: newFp };
    lib.clips.set(newFp, updated);
    idbPut("clips", updated);
    writeStr(newRel, outRel, rCap);
    writeStr(newFp, outFp, fpCap);
    return 1;
  },
  ov_lib_update_meta: (dirPtr, dirLen, fnPtr, fnLen, rating, starred, score, refPtr, refLen, cmtPtr, cmtLen) => {
    const fp = `idb://${joinRel(readStr(dirPtr, dirLen), readStr(fnPtr, fnLen))}`;
    const clip = lib.clips.get(fp);
    if (!clip) return 0;
    clip.rating = rating;
    clip.starred = starred !== 0;
    clip.comment = readStr(cmtPtr, cmtLen);
    const refPath = readStr(refPtr, refLen);
    if (refPath && score >= 0) {
      clip.scoresByRef = clip.scoresByRef || {};
      clip.scoresByRef[refPath] = { ...(clip.scoresByRef[refPath] || {}), overall: score };
    }
    idbPut("clips", clip);
    return 1;
  },
  ov_lib_update_scores: (dirPtr, dirLen, fnPtr, fnLen, refPtr, refLen, overall, pitch, size, weight, vowel, ver) => {
    const fp = `idb://${joinRel(readStr(dirPtr, dirLen), readStr(fnPtr, fnLen))}`;
    const clip = lib.clips.get(fp);
    if (!clip) return 0;
    const refPath = readStr(refPtr, refLen);
    if (refPath) {
      clip.scoresByRef = clip.scoresByRef || {};
      clip.scoresByRef[refPath] = { overall, pitch, size, weight, vowel, ver };
      idbPut("clips", clip);
    }
    return 1;
  },
  ov_lib_delete_entry: (pPtr, pLen, nPtr, nLen, isDir) => {
    const parentRel = cleanRel(readStr(pPtr, pLen));
    const name = readStr(nPtr, nLen);
    if (isDir) {
      const target = joinRel(parentRel, name);
      for (const f of [...lib.folders.keys()]) {
        if (f === target || f.startsWith(target + "/")) {
          lib.folders.delete(f);
          idbDelete("folders", f);
        }
      }
      for (const c of [...lib.clips.values()]) {
        if (c.relDir === target || c.relDir.startsWith(target + "/")) {
          lib.clips.delete(c.fullPath);
          idbDelete("clips", c.fullPath);
        }
      }
      return 1;
    }
    const fp = `idb://${joinRel(parentRel, name)}`;
    if (!lib.clips.has(fp)) return 0;
    lib.clips.delete(fp);
    idbDelete("clips", fp);
    return 1;
  },
  ov_lib_get_notes: (out, cap) => writeStr(lib.lastNotes, out, cap),
  ov_lib_set_notes: (dirPtr, dirLen, nPtr, nLen) => {
    const relDir = cleanRel(readStr(dirPtr, dirLen));
    const text = readStr(nPtr, nLen);
    lib.lastNotes = text;
    if (!relDir) {
      lib.rootNotes = text;
      idbPut("meta", { key: "rootNotes", value: text });
    } else {
      const existing = lib.folders.get(relDir) || { relPath: relDir, createdAt: nowStamp(), notes: "" };
      existing.notes = text;
      lib.folders.set(relDir, existing);
      idbPut("folders", existing);
    }
    return 1;
  },

  // WebGL2 Spectrogram hooks
  spec_set_lut: (ptr, len) => {
    specGl.setLut(new Uint8Array(memBuf(), ptr, len));
  },
  spec_alloc_tier: (tier, w, h) => {
    specGl.allocTier(tier, w, h);
  },
  spec_write_tier: (tier, x, w, h, ptr, len) => {
    specGl.writeTier(tier, x, w, h, new Uint8Array(memBuf(), ptr, len));
  },
  spec_clear_tier: (tier) => {
    specGl.clearTier(tier);
  },
  spec_draw: (ptr, count) => {
    specGl.draw(new Float32Array(memBuf(), ptr, count));
  },

  // Neural Pitch (SwiftF0) + Formants (FormantsTracker) Worker Bridge
  ov_neural_request: (jobId, isRef, samplesPtr, count, sampleRate, fmin, fmax) => {
    if (!neuralWorker || count < 64) return;
    const copy = new Float32Array(count);
    copy.set(new Float32Array(memBuf(), samplesPtr, count));
    neuralWorker.postMessage(
      { type: "analyze", jobId, isRef, samples: copy, sampleRate, fmin, fmax },
      [copy.buffer],
    );
  },
  ov_neural_pitch_count: () => (neuralResult?.f0Times ? neuralResult.f0Times.length : 0),
  ov_neural_copy_pitch: (timesPtr, hzPtr, confPtr, maxCount) => {
    if (!neuralResult?.f0Times) return 0;
    const n = Math.min(maxCount, neuralResult.f0Times.length);
    new Float64Array(memBuf(), timesPtr, n).set(neuralResult.f0Times.subarray(0, n));
    new Float64Array(memBuf(), hzPtr, n).set(neuralResult.f0Hz.subarray(0, n));
    new Float64Array(memBuf(), confPtr, n).set(neuralResult.f0Conf.subarray(0, n));
    return n;
  },
  ov_neural_formant_count: () => (neuralResult?.fmtTimes ? neuralResult.fmtTimes.length : 0),
  ov_neural_copy_formants: (timesPtr, f1Ptr, f2Ptr, f3Ptr, maxCount) => {
    if (!neuralResult?.fmtTimes) return 0;
    const n = Math.min(maxCount, neuralResult.fmtTimes.length);
    new Float64Array(memBuf(), timesPtr, n).set(neuralResult.fmtTimes.subarray(0, n));
    new Float64Array(memBuf(), f1Ptr, n).set(neuralResult.fmtF1.subarray(0, n));
    new Float64Array(memBuf(), f2Ptr, n).set(neuralResult.fmtF2.subarray(0, n));
    new Float64Array(memBuf(), f3Ptr, n).set(neuralResult.fmtF3.subarray(0, n));
    return n;
  },
  ov_neural_status: () => neuralStatusFlags,

  // Browser-local Source-Filter Voice Preview Resynthesizer (Pitch / Size / Weight)
  ov_synth_render: (
    samplesPtr,
    count,
    sampleRate,
    f0TimesPtr,
    f0HzPtr,
    f0Count,
    refF0Ptr,
    hasRefF0,
    pitchSemi,
    followContour,
    contourBlend,
    sizePct,
    weightDb,
  ) => {
    if (count < 64 || sampleRate <= 0) return 0;
    const pcm = new Float32Array(count);
    pcm.set(new Float32Array(memBuf(), samplesPtr, count));
    const f0Times = f0Count > 0 ? new Float64Array(memBuf(), f0TimesPtr, f0Count).slice() : new Float64Array(0);
    const f0Hz = f0Count > 0 ? new Float64Array(memBuf(), f0HzPtr, f0Count).slice() : new Float64Array(0);
    const refF0 = hasRefF0 && f0Count > 0 ? new Float64Array(memBuf(), refF0Ptr, f0Count).slice() : null;
    try {
      const rendered = resynthesizeVoice(pcm, sampleRate, f0Times, f0Hz, refF0, {
        pitchSemi,
        followContour: followContour !== 0,
        contourBlend,
        sizePct,
        weightDb,
      });
      synthAudio = { samples: rendered, sampleRate };
      return rendered.length;
    } catch (err) {
      console.warn("[Glint Synth] Resynthesis failed:", err);
      return 0;
    }
  },
  ov_synth_play: (startSec, endSec, loop) => {
    if (!synthAudio || !synthAudio.samples || synthAudio.samples.length === 0) return 0;
    return startPlayback(synthAudio.samples, synthAudio.sampleRate, startSec, endSec, loop !== 0);
  },
  ov_synth_copy: (outPtr, maxCount) => {
    if (!synthAudio || !synthAudio.samples) return 0;
    const n = Math.min(maxCount, synthAudio.samples.length);
    new Float32Array(memBuf(), outPtr, n).set(synthAudio.samples.subarray(0, n));
    return n;
  },

  // Vowel & Consonant Isolation (Option 1 instant acoustic + Option 2 lazy Allosaurus IPA)
  ov_phoneme_set_enabled: (enabled) => {
    if (enabled && neuralWorker) {
      neuralWorker.postMessage({ type: "enable_phonemes" });
    }
  },
  ov_vowel_analyze: (
    jobId,
    isRef,
    samplesPtr,
    count,
    sampleRate,
    f0TimesPtr,
    f0HzPtr,
    f0ConfPtr,
    f0Count,
    fmtTimesPtr,
    f1Ptr,
    f2Ptr,
    f3Ptr,
    fmtCount,
    wantNeural,
  ) => {
    const slot = isRef ? 1 : 0;
    if (count < 64 || sampleRate <= 0) {
      vowelSlots[slot] = { jobId, acoustic: null, spans: [], isNeural: false };
      return 0;
    }
    const pcm = new Float32Array(count);
    pcm.set(new Float32Array(memBuf(), samplesPtr, count));
    const f0Times = f0Count > 0 ? new Float64Array(memBuf(), f0TimesPtr, f0Count).slice() : null;
    const f0Hz = f0Count > 0 ? new Float64Array(memBuf(), f0HzPtr, f0Count).slice() : null;
    const f0Conf = f0Count > 0 ? new Float64Array(memBuf(), f0ConfPtr, f0Count).slice() : null;
    const fmtTimes = fmtCount > 0 ? new Float64Array(memBuf(), fmtTimesPtr, fmtCount).slice() : null;
    const fmtF1 = fmtCount > 0 ? new Float64Array(memBuf(), f1Ptr, fmtCount).slice() : null;
    const fmtF2 = fmtCount > 0 ? new Float64Array(memBuf(), f2Ptr, fmtCount).slice() : null;
    const fmtF3 = fmtCount > 0 ? new Float64Array(memBuf(), f3Ptr, fmtCount).slice() : null;

    const acoustic = computeAcousticVowelAnalysis(
      pcm,
      sampleRate,
      f0Times,
      f0Hz,
      f0Conf,
      fmtTimes,
      fmtF1,
      fmtF2,
      fmtF3,
    );
    vowelSlots[slot] = {
      jobId,
      acoustic,
      spans: acoustic.spans,
      isNeural: false,
      f0Times,
      f0Hz,
      f0Conf,
      fmtTimes,
      fmtF1,
      fmtF2,
      fmtF3,
    };

    if (wantNeural && neuralWorker) {
      const workerCopy = new Float32Array(pcm);
      neuralWorker.postMessage(
        { type: "analyze_phonemes", jobId, isRef: slot, samples: workerCopy, sampleRate },
        [workerCopy.buffer],
      );
    }
    return acoustic.spans.length;
  },
  ov_vowel_span_count: (isRef) => {
    const st = vowelSlots[isRef ? 1 : 0];
    return st && st.spans ? st.spans.length : 0;
  },
  ov_vowel_is_neural: (isRef) => {
    const st = vowelSlots[isRef ? 1 : 0];
    return st && st.isNeural ? 1 : 0;
  },
  ov_vowel_copy_spans: (isRef, t0Ptr, t1Ptr, labelBufPtr, labelStride, maxCount) => {
    const st = vowelSlots[isRef ? 1 : 0];
    if (!st || !st.spans || st.spans.length === 0) return 0;
    const n = Math.min(maxCount, st.spans.length);
    const outT0 = new Float64Array(memBuf(), t0Ptr, n);
    const outT1 = new Float64Array(memBuf(), t1Ptr, n);
    const labelMem = labelBufPtr && labelStride > 0 ? new Uint8Array(memBuf(), labelBufPtr, n * labelStride) : null;
    if (labelMem) labelMem.fill(0);
    for (let i = 0; i < n; i++) {
      const sp = st.spans[i];
      outT0[i] = sp.t0;
      outT1[i] = sp.t1;
      if (labelMem && sp.label) {
        const bytes = encoder.encode(sp.label);
        const len = Math.min(labelStride - 1, bytes.length);
        labelMem.set(bytes.subarray(0, len), i * labelStride);
      }
    }
    return n;
  },
  ov_play_filtered: (isRef, samplesPtr, count, sampleRate, startSec, endSec, loop, mode, timingMode, labMode, targetMonoHz) => {
    if (count < 64 || sampleRate <= 0) return 0;
    let pcm = readF32Copy(samplesPtr, count);
    const slot = isRef ? 1 : 0;
    let st = vowelSlots[slot];
    if (mode !== 0 && (!st || !st.acoustic || Math.abs(st.acoustic.durationSec - count / sampleRate) > 0.08)) {
      const ac = computeAcousticVowelAnalysis(pcm, sampleRate);
      st = {
        jobId: 0,
        acoustic: ac,
        spans: ac.spans,
        isNeural: false,
        f0Times: st?.f0Times || null,
        f0Hz: st?.f0Hz || null,
        f0Conf: st?.f0Conf || null,
        fmtTimes: st?.fmtTimes || null,
        fmtF1: st?.fmtF1 || null,
        fmtF2: st?.fmtF2 || null,
        fmtF3: st?.fmtF3 || null,
      };
      vowelSlots[slot] = st;
    }

    // Stage 1: Voice Lab acoustic transform (Whisperize, Monotone, Glottal, Solo F1/F2/F3, Breathiness)
    if (labMode > 0) {
      pcm = applyVoiceLabTransform(
        pcm,
        sampleRate,
        labMode,
        st?.f0Times || null,
        st?.f0Hz || null,
        st?.f0Conf || null,
        st?.fmtTimes || null,
        st?.fmtF1 || null,
        st?.fmtF2 || null,
        st?.fmtF3 || null,
        targetMonoHz || 0.0,
      );
    }

    // Stage 2: Vowel / Consonant isolation + Timing (0=In-place, 1=Compact, 2=TD-PSOLA Stretch/Hold)
    if (mode === 0) {
      return startPlayback(pcm, sampleRate, startSec, endSec, loop !== 0, null);
    }
    const iso = buildIsolatedAudio(
      pcm,
      sampleRate,
      st.spans,
      mode,
      timingMode | 0,
      startSec,
      endSec,
      st?.f0Times || null,
      st?.f0Hz || null,
    );
    return startPlayback(
      iso.samples,
      iso.sampleRate,
      iso.startSec,
      iso.endSec,
      loop !== 0,
      iso.timeMap,
    );
  },
  ov_klatt_play: (f0, f1, f2, f3, f4, durSec, tiltDb) => {
    try {
      const ctx = ensureAudioCtx();
      const sr = ctx.sampleRate || 48000;
      const samples = synthesizeKlattVowel(f0, f1, f2, f3, f4, durSec, tiltDb, sr);
      if (klattNode) {
        try {
          klattNode.stop();
          klattNode.disconnect();
        } catch {}
        klattNode = null;
      }
      const buf = ctx.createBuffer(1, samples.length, Math.round(sr));
      buf.copyToChannel(samples, 0);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      const gain = ctx.createGain();
      gain.gain.value = audioState.volume;
      src.connect(gain);
      gain.connect(ctx.destination);
      klattNode = src;
      src.onended = () => {
        if (klattNode === src) klattNode = null;
      };
      src.start();
      return 1;
    } catch (err) {
      console.warn("[Prism Klatt] Vowel synthesis failed:", err);
      return 0;
    }
  },
};

let synthAudio = null;
let klattNode = null;
const vowelSlots = [
  { jobId: 0, acoustic: null, spans: [], isNeural: false },
  { jobId: 0, acoustic: null, spans: [], isNeural: false },
];

let neuralWorker = null;
let neuralResult = null;
let neuralStatusFlags = 0;
const pendingNeuralPosts = [];

function initNeuralWorker() {
  if (typeof Worker === "undefined") return;
  try {
    neuralWorker = new Worker(new URL("./neural-worker.js", import.meta.url), { type: "module" });
    neuralWorker.addEventListener("message", (ev) => {
      const msg = ev.data;
      if (!msg) return;
      if (msg.type === "status") {
        neuralStatusFlags = msg.flags | 0;
        if (mountedPage) {
          mountedPage.post("neural_status", `${neuralStatusFlags}`);
        } else {
          pendingNeuralPosts.push(["neural_status", `${neuralStatusFlags}`]);
        }
      } else if (msg.type === "result") {
        neuralResult = msg;
        const payload = `${msg.jobId}|${msg.isRef ? 1 : 0}|${msg.hasPitch ? 1 : 0}|${msg.hasFormants ? 1 : 0}`;
        if (mountedPage) {
          mountedPage.post("neural_ready", payload);
        } else {
          pendingNeuralPosts.push(["neural_ready", payload]);
        }
      } else if (msg.type === "phonemes_result") {
        const slot = msg.isRef ? 1 : 0;
        const cur = vowelSlots[slot];
        if (cur && cur.jobId === msg.jobId && cur.acoustic) {
          const refined = refineNeuralPhonemeSpans(msg.phones || [], cur.acoustic);
          cur.spans = refined;
          cur.isNeural = true;
          const payload = `${msg.jobId}|${slot}|${refined.length}`;
          if (mountedPage) {
            mountedPage.post("phonemes_ready", payload);
          } else {
            pendingNeuralPosts.push(["phonemes_ready", payload]);
          }
        }
      }
    });
    neuralWorker.addEventListener("error", (err) => {
      console.warn("[Glint Neural] Worker error:", err.message || err);
    });
    neuralWorker.postMessage({ type: "warmup" });
  } catch (err) {
    console.warn("[Glint Neural] Failed to start worker:", err);
  }
}

// ============================================================================
// 5. Bootstrapping & Browser Event Wiring
// ============================================================================

async function boot() {
  initNeuralWorker();
  await hydrateLibrary();
  await enumerateBrowserDevices();

  const root = document.getElementById("app");
  const origInstantiate = WebAssembly.instantiate;
  WebAssembly.instantiate = async function (mod, importObj) {
    const res = await origInstantiate.call(WebAssembly, mod, importObj);
    const inst = res instanceof WebAssembly.Instance ? res : res.instance;
    if (inst?.exports?.memory) wasmMemory = inst.exports.memory;
    return res;
  };
  try {
    mountedPage = await mount({
      wasm: "./glint.wasm",
      root,
      imports: { glint: glintImports },
    });
  } finally {
    WebAssembly.instantiate = origInstantiate;
  }

  // Flush any neural events that arrived while WASM was mounting
  while (pendingNeuralPosts.length > 0) {
    const [evName, evPayload] = pendingNeuralPosts.shift();
    mountedPage.post(evName, evPayload);
  }

  const loader = document.getElementById("glint-loading");
  if (loader) loader.remove();

  // Keep WebGL2 canvas mounted & crisp after any DOM update or resize
  const obs = new MutationObserver(() => {
    if (specGl.attachToHost() && specGl.lastUniforms) {
      specGl.draw(specGl.lastUniforms);
    }
  });
  obs.observe(root, { childList: true, subtree: true });
  requestAnimationFrame(() => {
    if (specGl.attachToHost() && specGl.lastUniforms) {
      specGl.draw(specGl.lastUniforms);
    }
  });
  window.addEventListener("resize", () => {
    if (specGl.attachToHost() && specGl.lastUniforms) {
      specGl.draw(specGl.lastUniforms);
    }
  });

  // File picker handler
  const fileInput = document.getElementById("glint-file-input");
  fileInput?.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    try {
      const key = await decodeBrowserFile(file);
      mountedPage.post("file_chosen", key);
    } catch (err) {
      console.error("Failed to decode audio file:", err);
    }
  });

  // Drag-and-drop audio files anywhere on window
  const dropOverlay = document.getElementById("glint-drop-overlay");
  let dragDepth = 0;
  window.addEventListener("dragenter", (e) => {
    if (!e.dataTransfer?.types?.includes("Files")) return;
    e.preventDefault();
    dragDepth++;
    dropOverlay?.classList.add("visible");
  });
  window.addEventListener("dragover", (e) => {
    if (!e.dataTransfer?.types?.includes("Files")) return;
    e.preventDefault();
  });
  window.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) dropOverlay?.classList.remove("visible");
  });
  window.addEventListener("drop", async (e) => {
    e.preventDefault();
    dragDepth = 0;
    dropOverlay?.classList.remove("visible");
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    try {
      const key = await decodeBrowserFile(file);
      mountedPage.post("file_chosen", key);
    } catch (err) {
      console.error("Failed to decode dropped audio:", err);
    }
  });

  // Prevent browser page zoom on trackpad pinch / Ctrl+wheel / Cmd+wheel / Safari gestures
  // while still letting Loom handle the wheel event to zoom the spectrogram!
  window.addEventListener(
    "wheel",
    (e) => {
      if (e.ctrlKey || e.metaKey || e.target?.closest?.('[data-testid="spectrogram-stage"]')) {
        e.preventDefault();
      }
    },
    { passive: false },
  );
  for (const ev of ["gesturestart", "gesturechange", "gestureend"]) {
    window.addEventListener(ev, (e) => e.preventDefault(), { passive: false });
  }

  // Global keyboard shortcuts forwarded to Glint
  window.addEventListener("keydown", (e) => {
    const tag = e.target?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || e.target?.isContentEditable) return;
    const mods = (e.shiftKey ? 1 : 0) | (e.ctrlKey ? 2 : 0) | (e.altKey ? 4 : 0) | (e.metaKey ? 8 : 0);
    const key = e.key;
    const allowRepeat = key === "ArrowLeft" || key === "ArrowRight" || key === "=" || key === "+" || key === "-";
    if (e.repeat && !allowRepeat) return;
    if ((e.metaKey || e.ctrlKey) && (key === "=" || key === "+" || key === "-" || key === "0")) {
      e.preventDefault();
    }
    const handled = mountedPage.post("global_key", `${mods}|${key}`);
    if (handled || key === " " || ((e.metaKey || e.ctrlKey) && ["o", "s", "n", ","].includes(key.toLowerCase()))) {
      if (key === " " || ((e.metaKey || e.ctrlKey) && ["o", "s", ","].includes(key.toLowerCase()))) {
        e.preventDefault();
      }
    }
  });

  // Suppress native browser context menu on the spectrogram and library so Glint's menus show cleanly
  root.addEventListener("contextmenu", (e) => {
    e.preventDefault();
  });
}

boot().catch((err) => {
  console.error("Glint boot error:", err);
  const el = document.getElementById("glint-loading");
  if (el) {
    el.innerHTML = `<div class="brand">Glint</div><div class="sub" style="color:#d9381e">Failed to start: ${err.message}</div>`;
  }
});
