// Glint Neural Analysis Web Worker
// Runs SwiftF0 (132 KB ONNX, MIT) for pitch (F0) and FormantsTracker (4.4 MB W8 ONNX, MIT)
// for probabilistic heatmap formant tracking (F1-F3), completely off the UI main thread.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.webgpu.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";

const SWIFT_SR = 16000;
const SWIFT_HOP = 256; // 16 ms
const SWIFT_SILENCE_PEAK = 1e-3;
const FT_NFFT = 512;
const FT_HOP = 160; // 10 ms at 16 kHz
const FT_BINS = 257;
const FT_BIN_RES = (SWIFT_SR / 2) / FT_BINS;

let pitchSession = null;
let formantSession = null;
let formantBackend = "none";
let initPromise = null;
let busyCount = 0;

// Latest job id per target (0 = take, 1 = ref) so stale queued work is dropped immediately
const latestJobId = new Map();
// Simple LRU cache keyed by clip fingerprint -> { f0Times, f0Hz, f0Conf, fmtTimes, fmtF1, fmtF2, fmtF3 }
const resultCache = new Map();
const MAX_CACHE = 48;

function postStatus() {
  const flags =
    (pitchSession ? 1 : 0) |
    (formantSession ? 2 : 0) |
    (busyCount > 0 ? 4 : 0);
  self.postMessage({ type: "status", flags, backend: formantBackend });
}

async function ensureModels() {
  if (initPromise) return initPromise;
  initPromise = (async () => {
    try {
      const f0Resp = await fetch("./models/swift_f0.onnx");
      if (f0Resp.ok) {
        const f0Buf = await f0Resp.arrayBuffer();
        pitchSession = await ort.InferenceSession.create(f0Buf, {
          executionProviders: ["wasm"],
        });
        postStatus();
      }
    } catch (err) {
      console.warn("[Glint Neural] SwiftF0 load failed:", err);
    }

    try {
      const ftResp = await fetch("./models/formants_tracker.onnx");
      if (ftResp.ok) {
        const ftBuf = await ftResp.arrayBuffer();
        const hasWebGpu = typeof navigator !== "undefined" && !!navigator.gpu;
        if (hasWebGpu) {
          try {
            formantSession = await ort.InferenceSession.create(ftBuf, {
              executionProviders: ["webgpu"],
            });
            formantBackend = "webgpu";
          } catch (gpuErr) {
            console.warn("[Glint Neural] WebGPU init failed, falling back to WASM:", gpuErr);
            formantSession = await ort.InferenceSession.create(ftBuf, {
              executionProviders: ["wasm"],
            });
            formantBackend = "wasm";
          }
        } else {
          formantSession = await ort.InferenceSession.create(ftBuf, {
            executionProviders: ["wasm"],
          });
          formantBackend = "wasm";
        }
        postStatus();
      }
    } catch (err) {
      console.warn("[Glint Neural] FormantsTracker load failed:", err);
    }
  })();
  return initPromise;
}

// Fast 11-tap windowed-sinc anti-aliased resampler to 16 kHz mono
function resampleTo16k(src, srcRate) {
  if (!src || src.length === 0) return new Float32Array(0);
  if (Math.abs(srcRate - SWIFT_SR) < 1.0) {
    return src instanceof Float32Array ? src : new Float32Array(src);
  }
  const ratio = srcRate / SWIFT_SR;
  const outLen = Math.max(1, Math.round(src.length / ratio));
  const out = new Float32Array(outLen);
  const cutoff = Math.min(1.0, (SWIFT_SR / srcRate) * 0.92);
  const halfTaps = 6;
  const srcLen = src.length;
  for (let i = 0; i < outLen; i++) {
    const center = i * ratio;
    const cIdx = Math.floor(center);
    let acc = 0.0;
    let wSum = 0.0;
    const j0 = Math.max(0, cIdx - halfTaps);
    const j1 = Math.min(srcLen - 1, cIdx + halfTaps);
    for (let j = j0; j <= j1; j++) {
      const dx = j - center;
      const u = dx / (halfTaps + 1);
      const win = 0.54 + 0.46 * Math.cos(Math.PI * u);
      const x = Math.PI * cutoff * dx;
      const sinc = Math.abs(x) < 1e-6 ? 1.0 : Math.sin(x) / x;
      const w = win * sinc;
      acc += src[j] * w;
      wSum += w;
    }
    out[i] = wSum > 1e-8 ? acc / wSum : 0.0;
  }
  return out;
}

// Precomputed 512-point radix-2 FFT tables & periodic Hann window for FormantsTracker
const HANN_512 = new Float32Array(FT_NFFT);
const BIT_REV_512 = new Uint16Array(FT_NFFT);
const COS_512 = new Float32Array(FT_NFFT / 2);
const SIN_512 = new Float32Array(FT_NFFT / 2);
{
  for (let i = 0; i < FT_NFFT; i++) {
    HANN_512[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / FT_NFFT));
    let r = 0;
    let v = i;
    for (let b = 0; b < 9; b++) {
      r = (r << 1) | (v & 1);
      v >>= 1;
    }
    BIT_REV_512[i] = r;
  }
  for (let i = 0; i < FT_NFFT / 2; i++) {
    const ang = (-2.0 * Math.PI * i) / FT_NFFT;
    COS_512[i] = Math.cos(ang);
    SIN_512[i] = Math.sin(ang);
  }
}

function fft512Power(frame, re, im, outSpect, outOffset) {
  for (let i = 0; i < FT_NFFT; i++) {
    const r = BIT_REV_512[i];
    re[r] = frame[i];
    im[r] = 0.0;
  }
  for (let size = 2; size <= FT_NFFT; size <<= 1) {
    const half = size >> 1;
    const step = FT_NFFT / size;
    for (let i = 0; i < FT_NFFT; i += size) {
      let tw = 0;
      for (let j = 0; j < half; j++) {
        const wr = COS_512[tw];
        const wi = SIN_512[tw];
        tw += step;
        const a = i + j;
        const b = a + half;
        const tr = wr * re[b] - wi * im[b];
        const ti = wr * im[b] + wi * re[b];
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
  const scale = 1.0 / FT_NFFT; // matches torchaudio Spectrogram(power=2, normalized=True)
  for (let k = 0; k < FT_BINS; k++) {
    const rv = re[k];
    const iv = im[k];
    outSpect[outOffset + k] = (rv * rv + iv * iv) * scale;
  }
}

// Computes exact FormantsTracker input spectrogram: preemphasis(0.97), reflect-pad 256, n_fft=512, hop=160
function computeFormantSpectrogram(wav16k) {
  const n = wav16k.length;
  const emph = new Float32Array(n);
  let prev = 0.0;
  for (let i = 0; i < n; i++) {
    const cur = wav16k[i];
    emph[i] = cur - 0.97 * prev;
    prev = cur;
  }
  const pad = FT_NFFT >> 1; // 256
  const totalLen = n + pad * 2;
  const padded = new Float32Array(totalLen);
  for (let i = 0; i < pad; i++) {
    const srcIdx = Math.min(n - 1, pad - i);
    padded[i] = emph[srcIdx];
  }
  padded.set(emph, pad);
  for (let i = 0; i < pad; i++) {
    const srcIdx = Math.max(0, n - 2 - i);
    padded[pad + n + i] = emph[srcIdx];
  }
  const numFrames = Math.max(1, 1 + Math.floor((totalLen - FT_NFFT) / FT_HOP));
  const spect = new Float32Array(numFrames * FT_BINS);
  const frame = new Float32Array(FT_NFFT);
  const re = new Float32Array(FT_NFFT);
  const im = new Float32Array(FT_NFFT);
  for (let t = 0; t < numFrames; t++) {
    const base = t * FT_HOP;
    for (let k = 0; k < FT_NFFT; k++) {
      frame[k] = padded[base + k] * HANN_512[k];
    }
    fft512Power(frame, re, im, spect, t * FT_BINS);
  }
  return { spect, numFrames };
}

function decodeFormantHeatmaps(heatmapsData, numFrames, timeOffsetSec) {
  const fmtTimes = new Float64Array(numFrames);
  const fmtF1 = new Float64Array(numFrames);
  const fmtF2 = new Float64Array(numFrames);
  const fmtF3 = new Float64Array(numFrames);
  const outArrs = [fmtF1, fmtF2, fmtF3];
  const planeStride = numFrames * FT_BINS;

  for (let t = 0; t < numFrames; t++) {
    fmtTimes[t] = timeOffsetSec + t * (FT_HOP / SWIFT_SR);
  }
  for (let c = 0; c < 3; c++) {
    const cOff = c * planeStride;
    const dst = outArrs[c];
    for (let t = 0; t < numFrames; t++) {
      const rowOff = cOff + t * FT_BINS;
      let bestK = 0;
      let bestV = -1e30;
      for (let k = 0; k < FT_BINS; k++) {
        const v = heatmapsData[rowOff + k];
        if (v > bestV) {
          bestV = v;
          bestK = k;
        }
      }
      let shift = 0.0;
      if (bestK > 0 && bestK < FT_BINS - 1) {
        const a = heatmapsData[rowOff + bestK - 1];
        const b = bestV;
        const d = heatmapsData[rowOff + bestK + 1];
        const denom = a - 2.0 * b + d;
        if (Math.abs(denom) > 1e-8) {
          shift = Math.max(-0.5, Math.min(0.5, (0.5 * (a - d)) / denom));
        }
      }
      dst[t] = (bestK + shift + 0.5) * FT_BIN_RES;
    }
  }
  return { fmtTimes, fmtF1, fmtF2, fmtF3 };
}

function fingerprintClip(samples, sampleRate, fmin, fmax) {
  const n = samples.length;
  let h = 2166136261 >>> 0;
  const step = Math.max(1, Math.floor(n / 64));
  for (let i = 0; i < n; i += step) {
    const q = Math.round(samples[i] * 32767) & 0xffff;
    h ^= q;
    h = Math.imul(h, 16777619) >>> 0;
  }
  return `${n}:${Math.round(sampleRate)}:${Math.round(fmin)}:${Math.round(fmax)}:${h}`;
}

function cachePut(key, val) {
  if (resultCache.has(key)) resultCache.delete(key);
  resultCache.set(key, val);
  if (resultCache.size > MAX_CACHE) {
    const firstKey = resultCache.keys().next().value;
    resultCache.delete(firstKey);
  }
}

async function runSwiftF0(wav16k, fmin, fmax) {
  if (!pitchSession || wav16k.length < 64) return null;
  const clampedMin = Math.max(46.875, Math.min(1800.0, fmin || 65.0));
  const clampedMax = Math.max(clampedMin + 20.0, Math.min(2093.75, fmax || 600.0));
  const padded = wav16k.length < 256 ? (() => { const p = new Float32Array(256); p.set(wav16k); return p; })() : wav16k;
  const out = await pitchSession.run({
    audio: new ort.Tensor("float32", padded, [1, padded.length]),
    fmin: new ort.Tensor("float32", new Float32Array([clampedMin]), []),
    fmax: new ort.Tensor("float32", new Float32Array([clampedMax]), []),
  });
  const pitchRaw = out.pitch.data;
  const confRaw = out.confidence.data;
  const n = confRaw.length;
  const f0Times = new Float64Array(n);
  const f0Hz = new Float64Array(n);
  const f0Conf = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    f0Times[i] = i * (SWIFT_HOP / SWIFT_SR);
    f0Hz[i] = Number(pitchRaw[i]);
    // Silence guard matching swift_f0/core.py
    let peak = 0.0;
    const s0 = i * SWIFT_HOP;
    const s1 = Math.min(padded.length, s0 + SWIFT_HOP);
    for (let s = s0; s < s1; s++) {
      const a = Math.abs(padded[s]);
      if (a > peak) peak = a;
    }
    f0Conf[i] = peak < SWIFT_SILENCE_PEAK ? 0.0 : Number(confRaw[i]);
  }
  return { f0Times, f0Hz, f0Conf };
}

async function runFormantsTracker(wav16k, f0Result) {
  if (!formantSession || wav16k.length < 512) return null;
  // Crop leading/trailing silence to avoid wasting CNN compute on dead air
  let startSample = 0;
  let endSample = wav16k.length;
  if (f0Result && f0Result.f0Conf.length > 0) {
    let firstV = -1;
    let lastV = -1;
    for (let i = 0; i < f0Result.f0Conf.length; i++) {
      if (f0Result.f0Conf[i] >= 0.35) {
        if (firstV < 0) firstV = i;
        lastV = i;
      }
    }
    if (firstV >= 0) {
      startSample = Math.max(0, (firstV - 8) * SWIFT_HOP);
      endSample = Math.min(wav16k.length, (lastV + 10) * SWIFT_HOP);
    }
  }
  // Cap single pass to 12 seconds (192,000 samples) for predictable memory/latency
  if (endSample - startSample > 192000) {
    endSample = startSample + 192000;
  }
  if (endSample - startSample < 512) {
    startSample = 0;
    endSample = Math.min(wav16k.length, 192000);
  }
  const sub = wav16k.subarray(startSample, endSample);
  const { spect, numFrames } = computeFormantSpectrogram(sub);
  const out = await formantSession.run({
    spect: new ort.Tensor("float32", spect, [1, numFrames, FT_BINS]),
  });
  return decodeFormantHeatmaps(out.heatmaps.data, numFrames, startSample / SWIFT_SR);
}

// Serialize ONNX session execution (ORT WASM/WebGPU sessions shouldn't run concurrently)
let queueTail = Promise.resolve();

async function handleAnalyze(msg) {
  const { jobId, isRef, samples, sampleRate, fmin, fmax } = msg;
  if (latestJobId.get(isRef) !== jobId) return;

  const key = fingerprintClip(samples, sampleRate, fmin, fmax);
  const cached = resultCache.get(key);
  if (cached && cached.f0Times && cached.fmtTimes) {
    self.postMessage({
      type: "result",
      jobId,
      isRef,
      hasPitch: true,
      hasFormants: true,
      f0Times: cached.f0Times,
      f0Hz: cached.f0Hz,
      f0Conf: cached.f0Conf,
      fmtTimes: cached.fmtTimes,
      fmtF1: cached.fmtF1,
      fmtF2: cached.fmtF2,
      fmtF3: cached.fmtF3,
    });
    return;
  }

  busyCount++;
  postStatus();
  try {
    await ensureModels();
    if (latestJobId.get(isRef) !== jobId) return;

    const wav16k = resampleTo16k(samples, sampleRate);
    let f0Res = cached?.f0Times ? cached : await runSwiftF0(wav16k, fmin, fmax);
    if (latestJobId.get(isRef) !== jobId) return;

    // Stage 1: deliver SwiftF0 pitch immediately (~20-50ms) so pitch contour upgrades right away!
    if (f0Res) {
      self.postMessage({
        type: "result",
        jobId,
        isRef,
        hasPitch: true,
        hasFormants: false,
        f0Times: f0Res.f0Times,
        f0Hz: f0Res.f0Hz,
        f0Conf: f0Res.f0Conf,
      });
    }

    // Stage 2: run FormantsTracker for smooth neural F1/F2/F3
    const fmtRes = await runFormantsTracker(wav16k, f0Res);
    if (latestJobId.get(isRef) !== jobId) return;

    if (fmtRes) {
      const combined = {
        f0Times: f0Res?.f0Times || null,
        f0Hz: f0Res?.f0Hz || null,
        f0Conf: f0Res?.f0Conf || null,
        fmtTimes: fmtRes.fmtTimes,
        fmtF1: fmtRes.fmtF1,
        fmtF2: fmtRes.fmtF2,
        fmtF3: fmtRes.fmtF3,
      };
      cachePut(key, combined);
      self.postMessage({
        type: "result",
        jobId,
        isRef,
        hasPitch: !!f0Res,
        hasFormants: true,
        ...combined,
      });
    }
  } catch (err) {
    console.warn("[Glint Neural] Analysis failed:", err);
  } finally {
    busyCount = Math.max(0, busyCount - 1);
    postStatus();
  }
}

self.addEventListener("message", (ev) => {
  const msg = ev.data;
  if (!msg) return;
  if (msg.type === "warmup") {
    ensureModels();
    return;
  }
  if (msg.type === "analyze") {
    latestJobId.set(msg.isRef, msg.jobId);
    queueTail = queueTail.then(() => handleAnalyze(msg));
  }
});
