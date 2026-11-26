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
let phonemeSession = null;
let timbreSession = null;
let formantBackend = "none";
let initPromise = null;
let phonemeInitPromise = null;
let timbreInitPromise = null;
let busyCount = 0;
let phonemeBusyCount = 0;

// Latest job id per target (0 = take, 1 = ref) so stale queued work is dropped immediately
const latestJobId = new Map();
const latestPhoneJobId = new Map();
let latestLiveFormantJobId = 0;
let latestTimbreJobId = 0;
// Simple LRU cache keyed by clip fingerprint -> { f0Times, f0Hz, f0Conf, fmtTimes, fmtF1, fmtF2, fmtF3 }
const resultCache = new Map();
const phoneCache = new Map();
const timbreCache = new Map();
const MAX_CACHE = 48;

// Allosaurus uni2005 230-unit IPA table (index 0 = CTC blank, 1..229 = universal phones)
const ALLO_PHONES = ["","I","a","aː","ã","ă","b","bʲ","bʲj","bʷ","bʼ","bː","b̞","b̤","b̥","c","d","dʒ","dʲ","dː","d̚","d̥","d̪","d̯","d͡z","d͡ʑ","d͡ʒ","d͡ʒː","d͡ʒ̤","e","eː","e̞","f","fʲ","fʷ","fː","g","gʲ","gʲj","gʷ","gː","h","hʷ","i","ij","iː","i̞","i̥","i̯","j","k","kx","kʰ","kʲ","kʲj","kʷ","kʷʼ","kʼ","kː","k̟ʲ","k̟̚","k͡p̚","l","lʲ","lː","l̪","m","mʲ","mʲj","mʷ","mː","n","nj","nʲ","nː","n̪","n̺","o","oː","o̞","o̥","p","pf","pʰ","pʲ","pʲj","pʷ","pʷʼ","pʼ","pː","p̚","q","r","rː","s","sʲ","sʼ","sː","s̪","t","ts","tsʰ","tɕ","tɕʰ","tʂ","tʂʰ","tʃ","tʰ","tʲ","tʷʼ","tʼ","tː","t̚","t̪","t̪ʰ","t̪̚","t͡s","t͡sʼ","t͡ɕ","t͡ɬ","t͡ʃ","t͡ʃʲ","t͡ʃʼ","t͡ʃː","u","uə","uː","u͡w","v","vʲ","vʷ","vː","v̞","v̞ʲ","w","x","x̟ʲ","y","z","zj","zʲ","z̪","ä","æ","ç","çj","ð","ø","ŋ","ŋ̟","ŋ͡m","œ","œ̃","ɐ","ɐ̞","ɑ","ɑ̱","ɒ","ɓ","ɔ","ɔ̃","ɕ","ɕː","ɖ̤","ɗ","ə","ɛ","ɛ̃","ɟ","ɡ","ɡʲ","ɡ̤","ɡ̥","ɣ","ɣj","ɤ","ɤɐ̞","ɤ̆","ɥ","ɦ","ɨ","ɪ","ɫ","ɯ","ɯ̟","ɯ̥","ɰ","ɱ","ɲ","ɳ","ɴ","ɵ","ɸ","ɹ","ɹ̩","ɻ","ɻ̩","ɽ","ɾ","ɾj","ɾʲ","ɾ̠","ʀ","ʁ","ʁ̝","ʂ","ʃ","ʃʲː","ʃ͡ɣ","ʈ","ʉ̞","ʊ","ʋ","ʋʲ","ʌ","ʎ","ʏ","ʐ","ʑ","ʒ","ʒ͡ɣ","ʔ","ʝ","ː","β","β̞","θ","χ","ә","ḁ"];
const ALLO_IS_VOWEL = new Uint8Array([0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,1,1,0,1,1,1,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1]);

function postStatus() {
  const flags =
    (pitchSession ? 1 : 0) |
    (formantSession ? 2 : 0) |
    (busyCount > 0 ? 4 : 0) |
    (phonemeSession ? 8 : 0) |
    (phonemeBusyCount > 0 ? 16 : 0) |
    (timbreSession ? 32 : 0);
  self.postMessage({ type: "status", flags, backend: formantBackend });
}

async function ensureTimbreModel() {
  if (timbreSession) return timbreSession;
  if (timbreInitPromise) return timbreInitPromise;
  busyCount++;
  postStatus();
  timbreInitPromise = (async () => {
    try {
      const resp = await fetch("./models/wespeaker_campplus_q8.onnx");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const buf = await resp.arrayBuffer();
      timbreSession = await ort.InferenceSession.create(buf, {
        executionProviders: ["wasm"],
      });
      return timbreSession;
    } catch (err) {
      console.warn("[Glint Neural] WeSpeaker CAM++ load failed:", err);
      timbreInitPromise = null;
      return null;
    } finally {
      busyCount = Math.max(0, busyCount - 1);
      postStatus();
    }
  })();
  return timbreInitPromise;
}

async function ensurePhonemeModel() {
  if (phonemeSession) return phonemeSession;
  if (phonemeInitPromise) return phonemeInitPromise;
  phonemeBusyCount++;
  postStatus();
  phonemeInitPromise = (async () => {
    try {
      const resp = await fetch("./models/allosaurus_uni2005_q8.onnx");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const buf = await resp.arrayBuffer();
      phonemeSession = await ort.InferenceSession.create(buf, {
        executionProviders: ["wasm"],
      });
      return phonemeSession;
    } catch (err) {
      console.warn("[Prism Neural] Allosaurus IPA load failed:", err);
      phonemeInitPromise = null;
      return null;
    } finally {
      phonemeBusyCount = Math.max(0, phonemeBusyCount - 1);
      postStatus();
    }
  })();
  return phonemeInitPromise;
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
  // Cap total analyzed speech to 60 seconds (960,000 samples) and process in 10s chunks
  // with 0.5s context overlap so long clips get accurate neural F1-F3 all the way to the end!
  const MAX_TOTAL = 60 * SWIFT_SR;
  if (endSample - startSample > MAX_TOTAL) {
    endSample = startSample + MAX_TOTAL;
  }
  if (endSample - startSample < 512) {
    startSample = 0;
    endSample = Math.min(wav16k.length, MAX_TOTAL);
  }
  const CHUNK_SAMP = 10 * SWIFT_SR; // 160,000 samples (10s)
  const OVERLAP_SAMP = Math.round(0.5 * SWIFT_SR); // 8,000 samples (0.5s)
  if (endSample - startSample <= CHUNK_SAMP + OVERLAP_SAMP) {
    const sub = wav16k.subarray(startSample, endSample);
    const { spect, numFrames } = computeFormantSpectrogram(sub);
    const out = await formantSession.run({
      spect: new ort.Tensor("float32", spect, [1, numFrames, FT_BINS]),
    });
    return decodeFormantHeatmaps(out.heatmaps.data, numFrames, startSample / SWIFT_SR);
  }

  const allTimes = [];
  const allF1 = [];
  const allF2 = [];
  const allF3 = [];
  let cursor = startSample;
  let prevKeepEndSec = -1.0;
  while (cursor < endSample) {
    const winStart = Math.max(startSample, cursor - OVERLAP_SAMP);
    const winEnd = Math.min(endSample, cursor + CHUNK_SAMP + OVERLAP_SAMP);
    const keepEnd = Math.min(endSample, cursor + CHUNK_SAMP);
    const keepEndSec = keepEnd / SWIFT_SR;
    const sub = wav16k.subarray(winStart, winEnd);
    if (sub.length < 512) break;
    const { spect, numFrames } = computeFormantSpectrogram(sub);
    const out = await formantSession.run({
      spect: new ort.Tensor("float32", spect, [1, numFrames, FT_BINS]),
    });
    const dec = decodeFormantHeatmaps(out.heatmaps.data, numFrames, winStart / SWIFT_SR);
    for (let i = 0; i < dec.fmtTimes.length; i++) {
      const t = dec.fmtTimes[i];
      if (t > prevKeepEndSec && (keepEnd >= endSample || t <= keepEndSec)) {
        allTimes.push(t);
        allF1.push(dec.fmtF1[i]);
        allF2.push(dec.fmtF2[i]);
        allF3.push(dec.fmtF3[i]);
      }
    }
    prevKeepEndSec = keepEndSec;
    cursor = keepEnd;
  }
  return {
    fmtTimes: Float64Array.from(allTimes),
    fmtF1: Float64Array.from(allF1),
    fmtF2: Float64Array.from(allF2),
    fmtF3: Float64Array.from(allF3),
  };
}

async function handleAnalyzeLiveFormants(msg) {
  const { jobId, samples, sampleRate, startSec } = msg;
  if (latestLiveFormantJobId !== jobId) return;
  await ensureModels();
  if (!formantSession || latestLiveFormantJobId !== jobId) return;
  try {
    const wav16k = resampleTo16k(samples, sampleRate);
    if (wav16k.length < 512) return;
    const { spect, numFrames } = computeFormantSpectrogram(wav16k);
    const out = await formantSession.run({
      spect: new ort.Tensor("float32", spect, [1, numFrames, FT_BINS]),
    });
    if (latestLiveFormantJobId !== jobId) return;
    const dec = decodeFormantHeatmaps(out.heatmaps.data, numFrames, startSec || 0.0);
    self.postMessage({
      type: "live_formants_result",
      jobId,
      fmtTimes: dec.fmtTimes,
      fmtF1: dec.fmtF1,
      fmtF2: dec.fmtF2,
      fmtF3: dec.fmtF3,
    });
  } catch (err) {
    console.warn("[Glint Neural] Live formants failed:", err);
  }
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

// ============================================================================
// Allosaurus uni2005 (8 kHz Kaldi 40-MFCC + 3-frame stack stride 3 -> 30 ms)
// ============================================================================

const ALLO_SR = 8000;
const ALLO_WIN = 200; // 25 ms at 8 kHz
const ALLO_HOP = 80;  // 10 ms at 8 kHz
const ALLO_NFFT = 256;
const ALLO_BINS = 129;
const ALLO_MELS = 40;
const ALLO_CEP = 40;

const ALLO_POVEY = new Float64Array(ALLO_WIN);
const ALLO_BIT_REV = new Uint16Array(ALLO_NFFT);
const ALLO_COS = new Float64Array(ALLO_NFFT / 2);
const ALLO_SIN = new Float64Array(ALLO_NFFT / 2);
const ALLO_FB = new Float64Array(ALLO_MELS * ALLO_BINS);
const ALLO_DCT = new Float64Array(ALLO_CEP * ALLO_MELS);
const ALLO_LIFTER = new Float64Array(ALLO_CEP);
{
  for (let i = 0; i < ALLO_WIN; i++) {
    ALLO_POVEY[i] = Math.pow(0.5 - 0.5 * Math.cos((2.0 * Math.PI * i) / (ALLO_WIN - 1)), 0.85);
  }
  for (let i = 0; i < ALLO_NFFT; i++) {
    let r = 0;
    let v = i;
    for (let b = 0; b < 8; b++) {
      r = (r << 1) | (v & 1);
      v >>= 1;
    }
    ALLO_BIT_REV[i] = r;
  }
  for (let i = 0; i < ALLO_NFFT / 2; i++) {
    const a = (-2.0 * Math.PI * i) / ALLO_NFFT;
    ALLO_COS[i] = Math.cos(a);
    ALLO_SIN[i] = Math.sin(a);
  }
  const hz2mel = (f) => 1127.0 * Math.log(1.0 + f / 700.0);
  const melLo = hz2mel(40.0);
  const melHi = hz2mel(3800.0);
  const melPts = new Float64Array(ALLO_MELS + 2);
  for (let m = 0; m < ALLO_MELS + 2; m++) {
    melPts[m] = melLo + ((melHi - melLo) * m) / (ALLO_MELS + 1);
  }
  for (let m = 0; m < ALLO_MELS; m++) {
    const l = melPts[m];
    const c = melPts[m + 1];
    const r = melPts[m + 2];
    for (let k = 0; k < ALLO_BINS; k++) {
      const bm = hz2mel((k * ALLO_SR) / ALLO_NFFT);
      const w = Math.max(0.0, Math.min((bm - l) / (c - l), (r - bm) / (r - c)));
      ALLO_FB[m * ALLO_BINS + k] = w;
    }
  }
  for (let k = 0; k < ALLO_CEP; k++) {
    const sc = k === 0 ? Math.sqrt(1.0 / ALLO_MELS) : Math.sqrt(2.0 / ALLO_MELS);
    ALLO_LIFTER[k] = 1.0 + 0.5 * 22.0 * Math.sin((Math.PI * k) / 22.0);
    for (let m = 0; m < ALLO_MELS; m++) {
      ALLO_DCT[k * ALLO_MELS + m] = sc * Math.cos((Math.PI * k * (m + 0.5)) / ALLO_MELS);
    }
  }
}

function resampleToRate(src, srcRate, dstRate) {
  if (!src || src.length === 0) return new Float32Array(0);
  if (Math.abs(srcRate - dstRate) < 1.0) {
    return src instanceof Float32Array ? src : new Float32Array(src);
  }
  const ratio = srcRate / dstRate;
  const outLen = Math.max(1, Math.round(src.length / ratio));
  const out = new Float32Array(outLen);
  const cutoff = Math.min(1.0, (dstRate / srcRate) * 0.92);
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

function computeAllosaurusFeats(samples, sampleRate) {
  const wav8k = resampleToRate(samples, sampleRate, ALLO_SR);
  // Cap at 16s for snappy WASM BiLSTM latency
  const maxSamp = Math.min(wav8k.length, ALLO_SR * 16);
  if (maxSamp < ALLO_WIN) return null;
  const numFrames = 1 + Math.floor((maxSamp - ALLO_WIN) / ALLO_HOP);
  const mfcc = new Float32Array(numFrames * ALLO_CEP);

  const re = new Float64Array(ALLO_NFFT);
  const im = new Float64Array(ALLO_NFFT);
  const pow = new Float64Array(ALLO_BINS);
  const logMel = new Float64Array(ALLO_MELS);
  const emph = new Float64Array(ALLO_WIN);

  for (let t = 0; t < numFrames; t++) {
    const base = t * ALLO_HOP;
    let mean = 0.0;
    for (let i = 0; i < ALLO_WIN; i++) mean += wav8k[base + i] * 32768.0;
    mean /= ALLO_WIN;

    let prev = wav8k[base] * 32768.0 - mean;
    emph[0] = (prev - 0.97 * prev) * ALLO_POVEY[0];
    for (let i = 1; i < ALLO_WIN; i++) {
      const cur = wav8k[base + i] * 32768.0 - mean;
      emph[i] = (cur - 0.97 * prev) * ALLO_POVEY[i];
      prev = cur;
    }

    re.fill(0.0);
    im.fill(0.0);
    for (let i = 0; i < ALLO_WIN; i++) re[ALLO_BIT_REV[i]] = emph[i];
    for (let size = 2; size <= ALLO_NFFT; size <<= 1) {
      const half = size >> 1;
      const step = ALLO_NFFT / size;
      for (let i = 0; i < ALLO_NFFT; i += size) {
        let tw = 0;
        for (let j = 0; j < half; j++) {
          const wr = ALLO_COS[tw];
          const wi = ALLO_SIN[tw];
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
    for (let k = 0; k < ALLO_BINS; k++) pow[k] = re[k] * re[k] + im[k] * im[k];

    for (let m = 0; m < ALLO_MELS; m++) {
      let acc = 0.0;
      const off = m * ALLO_BINS;
      for (let k = 0; k < ALLO_BINS; k++) acc += ALLO_FB[off + k] * pow[k];
      logMel[m] = Math.log(acc > 1e-10 ? acc : 1e-10);
    }

    const row = t * ALLO_CEP;
    for (let c = 0; c < ALLO_CEP; c++) {
      let acc = 0.0;
      const off = c * ALLO_MELS;
      for (let m = 0; m < ALLO_MELS; m++) acc += ALLO_DCT[off + m] * logMel[m];
      mfcc[row + c] = acc * ALLO_LIFTER[c];
    }
  }

  // Utterance CMVN
  for (let c = 0; c < ALLO_CEP; c++) {
    let sum = 0.0;
    for (let t = 0; t < numFrames; t++) sum += mfcc[t * ALLO_CEP + c];
    const mu = sum / numFrames;
    let sq = 0.0;
    for (let t = 0; t < numFrames; t++) {
      const d = mfcc[t * ALLO_CEP + c] - mu;
      sq += d * d;
    }
    const invStd = 1.0 / (Math.sqrt(sq / numFrames) + 1e-8);
    for (let t = 0; t < numFrames; t++) {
      mfcc[t * ALLO_CEP + c] = (mfcc[t * ALLO_CEP + c] - mu) * invStd;
    }
  }

  // Stack (t-1, t, t+1) wrapped + subsample every 3rd frame -> (1, outT, 120)
  const outT = Math.ceil(numFrames / 3);
  const feats = new Float32Array(outT * 120);
  for (let i = 0; i < outT; i++) {
    const t = i * 3;
    const t0 = (t - 1 + numFrames) % numFrames;
    const t1 = t;
    const t2 = (t + 1) % numFrames;
    const dst = i * 120;
    feats.set(mfcc.subarray(t0 * ALLO_CEP, (t0 + 1) * ALLO_CEP), dst);
    feats.set(mfcc.subarray(t1 * ALLO_CEP, (t1 + 1) * ALLO_CEP), dst + 40);
    feats.set(mfcc.subarray(t2 * ALLO_CEP, (t2 + 1) * ALLO_CEP), dst + 80);
  }
  return { feats, outT };
}

async function handleAnalyzePhonemes(msg) {
  const { jobId, isRef, samples, sampleRate } = msg;
  if (latestPhoneJobId.get(isRef) !== jobId) return;

  const key = fingerprintClip(samples, sampleRate, 0, 0);
  const cached = phoneCache.get(key);
  if (cached) {
    self.postMessage({ type: "phonemes_result", jobId, isRef, phones: cached });
    return;
  }

  const sess = await ensurePhonemeModel();
  if (!sess || latestPhoneJobId.get(isRef) !== jobId) return;

  phonemeBusyCount++;
  postStatus();
  try {
    const prep = computeAllosaurusFeats(samples, sampleRate);
    if (!prep || prep.outT < 2) {
      self.postMessage({ type: "phonemes_result", jobId, isRef, phones: [] });
      return;
    }
    const out = await sess.run({
      feats: new ort.Tensor("float32", prep.feats, [1, prep.outT, 120]),
    });
    if (latestPhoneJobId.get(isRef) !== jobId) return;
    const logits = out.logits.data;
    const numUnits = 230;
    const phones = [];
    let prevUnit = 0;
    for (let i = 0; i < prep.outT; i++) {
      const row = i * numUnits;
      let bestU = 0;
      let bestV = logits[row];
      for (let u = 1; u < numUnits; u++) {
        const v = logits[row + u];
        if (v > bestV) {
          bestV = v;
          bestU = u;
        }
      }
      if (bestU !== 0 && bestU !== prevUnit) {
        phones.push({
          t: (i * 3 * ALLO_HOP + ALLO_WIN * 0.5) / ALLO_SR,
          unit: bestU,
          label: ALLO_PHONES[bestU] || "",
          isVowel: ALLO_IS_VOWEL[bestU] === 1,
        });
      }
      prevUnit = bestU;
    }
    if (phoneCache.has(key)) phoneCache.delete(key);
    phoneCache.set(key, phones);
    if (phoneCache.size > MAX_CACHE) {
      phoneCache.delete(phoneCache.keys().next().value);
    }
    self.postMessage({ type: "phonemes_result", jobId, isRef, phones });
  } catch (err) {
    console.warn("[Prism Neural] Phoneme inference failed:", err);
  } finally {
    phonemeBusyCount = Math.max(0, phonemeBusyCount - 1);
    postStatus();
  }
}

// ============================================================================
// WeSpeaker CAM++ LM (INT8, 8.0 MB) 80-dim Kaldi Fbank + 512-d Timbre Embedding
// ============================================================================
const TIMBRE_SR = 16000;
const TIMBRE_WIN = 400; // 25 ms
const TIMBRE_HOP = 160; // 10 ms
const TIMBRE_NFFT = 512;
const TIMBRE_BINS = 257;
const TIMBRE_MELS = 80;
const TIMBRE_POVEY = new Float64Array(TIMBRE_WIN);
const TIMBRE_FB = new Float64Array(TIMBRE_MELS * TIMBRE_BINS);
const TIMBRE_BIT_REV = new Uint16Array(TIMBRE_NFFT);
const TIMBRE_COS = new Float64Array(TIMBRE_NFFT >> 1);
const TIMBRE_SIN = new Float64Array(TIMBRE_NFFT >> 1);
{
  for (let i = 0; i < TIMBRE_WIN; i++) {
    TIMBRE_POVEY[i] = Math.pow(0.5 - 0.5 * Math.cos((2.0 * Math.PI * i) / (TIMBRE_WIN - 1)), 0.85);
  }
  const bits = 9;
  for (let i = 0; i < TIMBRE_NFFT; i++) {
    let r = 0, v = i;
    for (let b = 0; b < bits; b++) { r = (r << 1) | (v & 1); v >>= 1; }
    TIMBRE_BIT_REV[i] = r;
  }
  for (let i = 0; i < (TIMBRE_NFFT >> 1); i++) {
    const a = (-2.0 * Math.PI * i) / TIMBRE_NFFT;
    TIMBRE_COS[i] = Math.cos(a);
    TIMBRE_SIN[i] = Math.sin(a);
  }
  const hz2mel = (f) => 1127.0 * Math.log(1.0 + f / 700.0);
  const melLo = hz2mel(20.0);
  const melHi = hz2mel(7600.0);
  const melPts = new Float64Array(TIMBRE_MELS + 2);
  for (let i = 0; i < TIMBRE_MELS + 2; i++) {
    melPts[i] = melLo + ((melHi - melLo) * i) / (TIMBRE_MELS + 1);
  }
  for (let m = 0; m < TIMBRE_MELS; m++) {
    const l = melPts[m], c = melPts[m + 1], r = melPts[m + 2];
    for (let k = 0; k < TIMBRE_BINS; k++) {
      const bm = hz2mel((k * TIMBRE_SR) / TIMBRE_NFFT);
      TIMBRE_FB[m * TIMBRE_BINS + k] = Math.max(0.0, Math.min((bm - l) / (c - l), (r - bm) / (r - c)));
    }
  }
}

function computeTimbreFbank(samples, sampleRate) {
  const wav16k = resampleTo16k(samples, sampleRate);
  // Trim leading/trailing near-silence and cap to 16s for fast, speech-focused timbre embedding
  let start = 0;
  let end = wav16k.length;
  const thresh = 0.004;
  while (start < end && Math.abs(wav16k[start]) < thresh) start++;
  while (end > start && Math.abs(wav16k[end - 1]) < thresh) end--;
  if (end - start < TIMBRE_WIN * 2) {
    start = 0;
    end = wav16k.length;
  } else {
    start = Math.max(0, start - 1600);
    end = Math.min(wav16k.length, end + 1600);
  }
  const maxSamp = TIMBRE_SR * 16;
  if (end - start > maxSamp) end = start + maxSamp;
  const sub = wav16k.subarray(start, end);
  if (sub.length < TIMBRE_WIN) return null;

  const numFrames = 1 + Math.floor((sub.length - TIMBRE_WIN) / TIMBRE_HOP);
  const feats = new Float32Array(numFrames * TIMBRE_MELS);
  const re = new Float64Array(TIMBRE_NFFT);
  const im = new Float64Array(TIMBRE_NFFT);
  const pow = new Float64Array(TIMBRE_BINS);
  const emph = new Float64Array(TIMBRE_WIN);

  for (let t = 0; t < numFrames; t++) {
    const base = t * TIMBRE_HOP;
    let mean = 0.0;
    for (let i = 0; i < TIMBRE_WIN; i++) mean += sub[base + i] * 32768.0;
    mean /= TIMBRE_WIN;

    let prev = sub[base] * 32768.0 - mean;
    emph[0] = prev * 0.03 * TIMBRE_POVEY[0];
    for (let i = 1; i < TIMBRE_WIN; i++) {
      const cur = sub[base + i] * 32768.0 - mean;
      emph[i] = (cur - 0.97 * prev) * TIMBRE_POVEY[i];
      prev = cur;
    }

    re.fill(0.0);
    im.fill(0.0);
    for (let i = 0; i < TIMBRE_WIN; i++) re[TIMBRE_BIT_REV[i]] = emph[i];
    for (let size = 2; size <= TIMBRE_NFFT; size <<= 1) {
      const half = size >> 1;
      const step = TIMBRE_NFFT / size;
      for (let i = 0; i < TIMBRE_NFFT; i += size) {
        let tw = 0;
        for (let j = 0; j < half; j++) {
          const wr = TIMBRE_COS[tw];
          const wi = TIMBRE_SIN[tw];
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
    for (let k = 0; k < TIMBRE_BINS; k++) pow[k] = re[k] * re[k] + im[k] * im[k];

    const row = t * TIMBRE_MELS;
    for (let m = 0; m < TIMBRE_MELS; m++) {
      let acc = 0.0;
      const off = m * TIMBRE_BINS;
      for (let k = 0; k < TIMBRE_BINS; k++) acc += TIMBRE_FB[off + k] * pow[k];
      feats[row + m] = Math.log(acc > 1e-10 ? acc : 1e-10);
    }
  }

  // Per-utterance cepstral mean normalization (WeSpeaker standard)
  for (let m = 0; m < TIMBRE_MELS; m++) {
    let sum = 0.0;
    for (let t = 0; t < numFrames; t++) sum += feats[t * TIMBRE_MELS + m];
    const mu = sum / numFrames;
    for (let t = 0; t < numFrames; t++) feats[t * TIMBRE_MELS + m] -= mu;
  }
  return { feats, numFrames };
}

async function computeTimbreEmbedding(sess, samples, sampleRate) {
  const key = fingerprintClip(samples, sampleRate, 1, 80);
  const cached = timbreCache.get(key);
  if (cached) return cached;
  const prep = computeTimbreFbank(samples, sampleRate);
  if (!prep || prep.numFrames < 8) return null;
  const out = await sess.run({
    feats: new ort.Tensor("float32", prep.feats, [1, prep.numFrames, TIMBRE_MELS]),
  });
  const raw = out.embs.data;
  const emb = new Float32Array(raw.length);
  let sq = 0.0;
  for (let i = 0; i < raw.length; i++) {
    const v = Number(raw[i]);
    emb[i] = v;
    sq += v * v;
  }
  const inv = 1.0 / (Math.sqrt(sq) + 1e-12);
  for (let i = 0; i < emb.length; i++) emb[i] *= inv;
  if (timbreCache.has(key)) timbreCache.delete(key);
  timbreCache.set(key, emb);
  if (timbreCache.size > MAX_CACHE) timbreCache.delete(timbreCache.keys().next().value);
  return emb;
}

async function handleCompareTimbre(msg) {
  const { jobId, takeSamples, takeSr, refSamples, refSr } = msg;
  if (latestTimbreJobId !== jobId) return;
  const sess = await ensureTimbreModel();
  if (!sess || latestTimbreJobId !== jobId) return;
  try {
    const eTake = await computeTimbreEmbedding(sess, takeSamples, takeSr);
    if (!eTake || latestTimbreJobId !== jobId) return;
    const eRef = await computeTimbreEmbedding(sess, refSamples, refSr);
    if (!eRef || latestTimbreJobId !== jobId) return;
    let dot = 0.0;
    const n = Math.min(eTake.length, eRef.length);
    for (let i = 0; i < n; i++) dot += eTake[i] * eRef[i];
    const cos = Math.max(-1.0, Math.min(1.0, dot));
    const pct = Math.max(0, Math.min(100, Math.round(((cos - 0.10) / 0.68) * 100)));
    const cosPermille = Math.max(0, Math.min(1000, Math.round(Math.max(0, cos) * 1000)));
    self.postMessage({ type: "timbre_result", jobId, pct, cosPermille });
  } catch (err) {
    console.warn("[Glint Neural] Timbre embedding failed:", err);
  }
}

self.addEventListener("message", (ev) => {
  const msg = ev.data;
  if (!msg) return;
  if (msg.type === "warmup") {
    ensureModels();
    return;
  }
  if (msg.type === "enable_phonemes") {
    ensurePhonemeModel();
    return;
  }
  if (msg.type === "analyze") {
    latestJobId.set(msg.isRef, msg.jobId);
    queueTail = queueTail.then(() => handleAnalyze(msg));
    return;
  }
  if (msg.type === "analyze_live_formants") {
    latestLiveFormantJobId = msg.jobId;
    queueTail = queueTail.then(() => handleAnalyzeLiveFormants(msg));
    return;
  }
  if (msg.type === "compare_timbre") {
    latestTimbreJobId = msg.jobId;
    queueTail = queueTail.then(() => handleCompareTimbre(msg));
    return;
  }
  if (msg.type === "analyze_phonemes") {
    latestPhoneJobId.set(msg.isRef, msg.jobId);
    queueTail = queueTail.then(() => handleAnalyzePhonemes(msg));
  }
});
