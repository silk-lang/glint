// Prism Vowel & Consonant Isolation Engine
// Option 1: Instant (<10 ms) acoustic + SwiftF0 voicing + FormantsTracker vowel nucleus detector
// Option 2: Refines Allosaurus neural IPA phone events onto exact acoustic boundaries
// Audio filter: Click-free raised-cosine in-place gating, compact stitching, OR TD-PSOLA vowel stretch/hold
import { stretchVowelSegmentPsola } from "./voice-lab.js";

function makeFftTables(n) {
  const bits = Math.round(Math.log2(n));
  const bitRev = new Uint16Array(n);
  const cos = new Float32Array(n >> 1);
  const sin = new Float32Array(n >> 1);
  const hann = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    let v = i;
    for (let b = 0; b < bits; b++) {
      r = (r << 1) | (v & 1);
      v >>= 1;
    }
    bitRev[i] = r;
    hann[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (n - 1)));
  }
  for (let i = 0; i < (n >> 1); i++) {
    const a = (-2.0 * Math.PI * i) / n;
    cos[i] = Math.cos(a);
    sin[i] = Math.sin(a);
  }
  return { n, bitRev, cos, sin, hann };
}

const FFT_CACHE = new Map();
function getFft(n) {
  let t = FFT_CACHE.get(n);
  if (!t) {
    t = makeFftTables(n);
    FFT_CACHE.set(n, t);
  }
  return t;
}

function fftPower(frame, tbl, re, im, outPow) {
  const n = tbl.n;
  const bitRev = tbl.bitRev;
  for (let i = 0; i < n; i++) {
    re[bitRev[i]] = frame[i];
    im[bitRev[i]] = 0.0;
  }
  const cos = tbl.cos;
  const sin = tbl.sin;
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = n / size;
    for (let i = 0; i < n; i += size) {
      let tw = 0;
      for (let j = 0; j < half; j++) {
        const wr = cos[tw];
        const wi = sin[tw];
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
  const bins = (n >> 1) + 1;
  const inv = 1.0 / (n * n);
  for (let k = 0; k < bins; k++) {
    outPow[k] = (re[k] * re[k] + im[k] * im[k]) * inv;
  }
}

function lookupSeries(times, vals, t) {
  const n = times ? times.length : 0;
  if (n === 0) return 0.0;
  if (t <= times[0]) return vals[0];
  if (t >= times[n - 1]) return vals[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  const i1 = lo;
  const i0 = Math.max(0, lo - 1);
  const dt = times[i1] - times[i0];
  const a = dt > 1e-6 ? (t - times[i0]) / dt : 0.0;
  return vals[i0] * (1.0 - a) + vals[i1] * a;
}

// Fast normalized autocorrelation peak in human speech pitch range (65..600 Hz)
function estimateVoicingAcf(samples, center, winLen, sr) {
  const half = winLen >> 1;
  const s0 = Math.max(0, center - half);
  const s1 = Math.min(samples.length, s0 + winLen);
  const len = s1 - s0;
  if (len < 32) return 0.0;
  let energy = 0.0;
  for (let i = 0; i < len; i++) {
    const v = samples[s0 + i];
    energy += v * v;
  }
  if (energy < 1e-9) return 0.0;
  const minLag = Math.max(4, Math.floor(sr / 600));
  const maxLag = Math.min(len - 8, Math.ceil(sr / 65));
  if (maxLag <= minLag) return 0.0;
  // Subsample stride for speed on 44.1/48 kHz audio
  const stride = sr > 24000 ? 3 : 2;
  let best = 0.0;
  for (let lag = minLag; lag <= maxLag; lag += stride) {
    let num = 0.0;
    let eA = 0.0;
    let eB = 0.0;
    const lim = len - lag;
    for (let i = 0; i < lim; i += stride) {
      const a = samples[s0 + i];
      const b = samples[s0 + i + lag];
      num += a * b;
      eA += a * a;
      eB += b * b;
    }
    const den = Math.sqrt(eA * eB) + 1e-12;
    const r = num / den;
    if (r > best) best = r;
  }
  return Math.max(0.0, Math.min(1.0, best));
}

function clamp01(x) {
  return x < 0.0 ? 0.0 : x > 1.0 ? 1.0 : x;
}

function smoothstep(edge0, edge1, x) {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3.0 - 2.0 * t);
}

/**
 * Option 1: Instant acoustic + SwiftF0 + FormantsTracker vowel nucleus analyzer.
 * Returns frame features (also reused to sharpen Option 2 neural IPA boundaries)
 * and clean merged vowel spans [{ t0, t1, label: "" }].
 */
export function computeAcousticVowelAnalysis(
  samples,
  sampleRate,
  f0Times = null,
  f0Hz = null,
  f0Conf = null,
  fmtTimes = null,
  fmtF1 = null,
  fmtF2 = null,
  fmtF3 = null,
) {
  const n = samples ? samples.length : 0;
  const dur = sampleRate > 0 ? n / sampleRate : 0.0;
  if (n < 64 || sampleRate <= 0) {
    return {
      source: "acoustic",
      hopSec: 0.01,
      durationSec: dur,
      numFrames: 0,
      rmsDb: new Float32Array(0),
      voicing: new Float32Array(0),
      hfRatio: new Float32Array(0),
      vowelProb: new Float32Array(0),
      peakDb: -90,
      spans: [],
    };
  }

  const hopSec = 0.01;
  const hop = Math.max(16, Math.round(sampleRate * hopSec));
  const winLen = Math.max(64, Math.round(sampleRate * 0.025));
  let nfft = 256;
  while (nfft < winLen) nfft <<= 1;
  const tbl = getFft(nfft);
  const bins = (nfft >> 1) + 1;
  const binHz = sampleRate / nfft;

  const numFrames = Math.max(1, Math.floor(n / hop));
  const rmsDb = new Float32Array(numFrames);
  const voicing = new Float32Array(numFrames);
  const hfRatio = new Float32Array(numFrames);
  const zcrArr = new Float32Array(numFrames);
  const eLowArr = new Float32Array(numFrames);
  const eFormArr = new Float32Array(numFrames);
  const eF2Arr = new Float32Array(numFrames);
  const tonalityArr = new Float32Array(numFrames);

  const frame = new Float32Array(nfft);
  const re = new Float32Array(nfft);
  const im = new Float32Array(nfft);
  const pow = new Float32Array(bins);

  const bLow0 = Math.max(1, Math.floor(90 / binHz));
  const bLow1 = Math.min(bins - 1, Math.ceil(360 / binHz));
  const bForm0 = Math.max(1, Math.floor(280 / binHz));
  const bForm1 = Math.min(bins - 1, Math.ceil(3400 / binHz));
  const bF20 = Math.max(1, Math.floor(650 / binHz));
  const bF21 = Math.min(bins - 1, Math.ceil(3200 / binHz));
  const bHi0 = Math.min(bins - 1, Math.floor(3500 / binHz));
  const bHi1 = Math.min(bins - 1, Math.ceil(7800 / binHz));

  const hasF0 = !!(f0Times && f0Times.length > 0 && f0Conf && f0Conf.length === f0Times.length);
  const hasFmt = !!(fmtTimes && fmtTimes.length > 0 && fmtF1 && fmtF2);

  const halfWin = winLen >> 1;
  const winScale = (nfft - 1) / Math.max(1, winLen - 1);

  for (let t = 0; t < numFrames; t++) {
    const center = t * hop + (hop >> 1);
    const tSec = center / sampleRate;
    const s0 = center - halfWin;

    let sumSq = 0.0;
    let zc = 0;
    let prev = 0.0;
    frame.fill(0.0);
    for (let i = 0; i < winLen; i++) {
      const idx = s0 + i;
      const v = idx >= 0 && idx < n ? samples[idx] : 0.0;
      sumSq += v * v;
      if (i > 0 && ((v >= 0 && prev < 0) || (v < 0 && prev >= 0))) zc++;
      prev = v;
      const wIdx = Math.min(nfft - 1, Math.round(i * winScale));
      frame[i] = v * tbl.hann[wIdx];
    }
    const rms = Math.sqrt(sumSq / winLen);
    rmsDb[t] = 20.0 * Math.log10(rms + 1e-7);
    zcrArr[t] = zc / (winLen / sampleRate);

    fftPower(frame, tbl, re, im, pow);

    let eLow = 0.0;
    for (let k = bLow0; k <= bLow1; k++) eLow += pow[k];
    let eForm = 0.0;
    let maxForm = 0.0;
    let cntForm = 0;
    for (let k = bForm0; k <= bForm1; k++) {
      const p = pow[k];
      eForm += p;
      if (p > maxForm) maxForm = p;
      cntForm++;
    }
    let eF2 = 0.0;
    for (let k = bF20; k <= bF21; k++) eF2 += pow[k];
    let eHi = 0.0;
    for (let k = bHi0; k <= bHi1; k++) eHi += pow[k];

    eLowArr[t] = eLow;
    eFormArr[t] = eForm;
    eF2Arr[t] = eF2;
    hfRatio[t] = eHi / (eForm + eLow + 1e-12);
    const meanForm = eForm / Math.max(1, cntForm);
    tonalityArr[t] = maxForm / (meanForm + 1e-12);

    if (hasF0) {
      const conf = lookupSeries(f0Times, f0Conf, tSec);
      const hz = f0Hz ? lookupSeries(f0Times, f0Hz, tSec) : 0.0;
      voicing[t] = hz > 45.0 ? clamp01(conf) : clamp01(conf * 0.35);
    } else {
      voicing[t] = estimateVoicingAcf(samples, center, winLen, sampleRate);
    }
  }

  // Utterance 95th-percentile voiced energy reference
  const voicedDbs = [];
  for (let t = 0; t < numFrames; t++) {
    if (voicing[t] >= 0.45 && rmsDb[t] > -65) voicedDbs.push(rmsDb[t]);
  }
  voicedDbs.sort((a, b) => a - b);
  const peakDb =
    voicedDbs.length > 0
      ? voicedDbs[Math.min(voicedDbs.length - 1, Math.floor(voicedDbs.length * 0.92))]
      : -30.0;

  // Frame-level soft vowel probability
  const rawProb = new Float32Array(numFrames);
  for (let t = 0; t < numFrames; t++) {
    const tSec = (t * hop + (hop >> 1)) / sampleRate;
    const relDb = rmsDb[t] - peakDb;

    // 1. Energy gate (vowels carry syllable-nucleus energy; drop silence & stop closures)
    const sEnergy = smoothstep(-25.0, -10.5, relDb);
    if (sEnergy < 0.02) {
      rawProb[t] = 0.0;
      continue;
    }

    // 2. Voicing gate (drops all voiceless consonants /p, t, k, s, ʃ, f, θ, h/)
    const sVoice = smoothstep(0.32, 0.62, voicing[t]);

    // 3. Anti-friction gate (drops voiced fricatives /z, v, ð, ʒ/ and noisy bursts)
    const sCleanHf = 1.0 - smoothstep(0.18, 0.55, hfRatio[t]);
    const sCleanZcr = 1.0 - smoothstep(2300, 4200, zcrArr[t]);
    const sNoFric = Math.min(sCleanHf, 0.35 + 0.65 * sCleanZcr);

    // 4. Open vocal tract / anti-nasal-murmur gate
    // Closed nasals (/m, n, ŋ/) and voiced stop bars (/b, d, g/) pack energy below 360 Hz
    // while damping F2/F3 (650..3200 Hz) and dropping overall oral formant energy.
    const formFrac = eFormArr[t] / (eFormArr[t] + eLowArr[t] + 1e-12);
    const f2ToLow = eF2Arr[t] / (eLowArr[t] + 1e-12);
    // Keep /u/ and /o/ (high formFrac or strong energy) while damping quiet nasal murmurs
    let sOral = 1.0;
    if (f2ToLow < 0.08 && relDb < -6.5) {
      sOral = smoothstep(0.015, 0.08, f2ToLow) * 0.75 + 0.25 * smoothstep(-14.0, -5.0, relDb);
    } else if (formFrac < 0.28 && relDb < -7.5) {
      sOral = smoothstep(0.10, 0.30, formFrac);
    }

    // 5. Spectral peakiness (tonality) & formant plausibility boost
    const sTone = 0.7 + 0.3 * smoothstep(2.5, 9.0, tonalityArr[t]);
    let sFmt = 1.0;
    if (hasFmt) {
      const f1 = lookupSeries(fmtTimes, fmtF1, tSec);
      const f2 = lookupSeries(fmtTimes, fmtF2, tSec);
      if (f1 >= 210 && f1 <= 1100 && f2 >= 620 && f2 <= 3050 && f2 > f1 + 140) {
        sFmt = 1.12;
      } else if (f1 > 0 && (f1 < 180 || f2 < 520)) {
        sFmt = 0.78;
      }
    }

    rawProb[t] = clamp01(sEnergy * sVoice * sNoFric * sOral * sTone * sFmt);
  }

  // Smooth over 3 frames [0.2, 0.6, 0.2]
  const vowelProb = new Float32Array(numFrames);
  for (let t = 0; t < numFrames; t++) {
    const a = t > 0 ? rawProb[t - 1] : rawProb[t];
    const b = rawProb[t];
    const c = t + 1 < numFrames ? rawProb[t + 1] : rawProb[t];
    vowelProb[t] = 0.2 * a + 0.6 * b + 0.2 * c;
  }

  // Hysteresis thresholding -> spans
  const rawSpans = [];
  let inSpan = false;
  let startT = 0;
  for (let t = 0; t < numFrames; t++) {
    const p = vowelProb[t];
    if (!inSpan && p >= 0.54) {
      inSpan = true;
      startT = t;
    } else if (inSpan && p < 0.36) {
      rawSpans.push({ t0: startT * hopSec, t1: t * hopSec, label: "" });
      inSpan = false;
    }
  }
  if (inSpan) {
    rawSpans.push({ t0: startT * hopSec, t1: Math.min(dur, numFrames * hopSec), label: "" });
  }

  const spans = cleanSpans(rawSpans, dur, 0.024, 0.032, 0.010, 0.012);
  return {
    source: "acoustic",
    hopSec,
    durationSec: dur,
    numFrames,
    rmsDb,
    voicing,
    hfRatio,
    vowelProb,
    peakDb,
    spans,
  };
}

function cleanSpans(rawSpans, dur, mergeGapSec, minDurSec, padLeft, padRight) {
  if (!rawSpans || rawSpans.length === 0) return [];
  // Merge tiny gaps
  const merged = [{ ...rawSpans[0] }];
  for (let i = 1; i < rawSpans.length; i++) {
    const prev = merged[merged.length - 1];
    const cur = rawSpans[i];
    if (cur.t0 - prev.t1 <= mergeGapSec) {
      prev.t1 = Math.max(prev.t1, cur.t1);
      if (cur.label && cur.label !== prev.label) {
        prev.label = prev.label ? `${prev.label}${cur.label}` : cur.label;
      }
    } else {
      merged.push({ ...cur });
    }
  }
  // Drop tiny blips and pad edges cleanly without overlapping neighbors
  const kept = merged.filter((s) => s.t1 - s.t0 >= minDurSec);
  for (let i = 0; i < kept.length; i++) {
    const s = kept[i];
    const limL = i > 0 ? 0.5 * (kept[i - 1].t1 + s.t0) : 0.0;
    const limR = i + 1 < kept.length ? 0.5 * (s.t1 + kept[i + 1].t0) : dur;
    s.t0 = Math.max(limL, Math.max(0.0, s.t0 - padLeft));
    s.t1 = Math.min(limR, Math.min(dur, s.t1 + padRight));
  }
  return kept;
}

/**
 * Option 2: Convert Allosaurus neural IPA phone spikes into crisp, boundary-accurate
 * vowel spans using the acoustic energy/voicing/friction curves between adjacent phones.
 */
export function refineNeuralPhonemeSpans(phones, acoustic) {
  if (!phones || phones.length === 0 || !acoustic || acoustic.numFrames === 0) {
    return acoustic ? acoustic.spans : [];
  }
  const dur = acoustic.durationSec;
  const hopSec = acoustic.hopSec;
  const nf = acoustic.numFrames;
  const rmsDb = acoustic.rmsDb;
  const voicing = acoustic.voicing;
  const hfRatio = acoustic.hfRatio;
  const peakDb = acoustic.peakDb;

  const isVowelFriendlyFrame = (idx) => {
    if (idx < 0 || idx >= nf) return false;
    const relDb = rmsDb[idx] - peakDb;
    return relDb > -24.0 && voicing[idx] >= 0.35 && hfRatio[idx] < 0.42;
  };

  const out = [];
  for (let i = 0; i < phones.length; i++) {
    const p = phones[i];
    if (!p.isVowel) continue;

    // Neighbor bounds: stop halfway to adjacent consonant phones (tighter on consonants!)
    const prev = i > 0 ? phones[i - 1] : null;
    const next = i + 1 < phones.length ? phones[i + 1] : null;

    // CTC spikes fire slightly after phone onset (~15-30ms), so allow a bit more room rightward
    const leftHard = prev
      ? prev.isVowel
        ? 0.5 * (prev.t + p.t)
        : prev.t + Math.min(0.045, 0.45 * (p.t - prev.t))
      : Math.max(0.0, p.t - 0.14);
    const rightHard = next
      ? next.isVowel
        ? 0.5 * (p.t + next.t)
        : next.t - Math.min(0.035, 0.35 * (next.t - p.t))
      : Math.min(dur, p.t + 0.18);

    const loBound = Math.max(0.0, Math.max(leftHard, p.t - 0.22));
    const hiBound = Math.min(dur, Math.min(rightHard, p.t + 0.28));

    const centerIdx = Math.max(0, Math.min(nf - 1, Math.round(p.t / hopSec)));
    const minIdx = Math.max(0, Math.floor(loBound / hopSec));
    const maxIdx = Math.min(nf - 1, Math.ceil(hiBound / hopSec));

    // Walk outward from CTC peak across voiced/open frames
    let lIdx = centerIdx;
    let missL = 0;
    while (lIdx > minIdx) {
      if (isVowelFriendlyFrame(lIdx - 1)) {
        lIdx--;
        missL = 0;
      } else {
        missL++;
        if (missL > 1) break;
        lIdx--;
      }
    }
    let rIdx = centerIdx;
    let missR = 0;
    while (rIdx < maxIdx) {
      if (isVowelFriendlyFrame(rIdx + 1)) {
        rIdx++;
        missR = 0;
      } else {
        missR++;
        if (missR > 1) break;
        rIdx++;
      }
    }

    let t0 = Math.max(loBound, lIdx * hopSec);
    let t1 = Math.min(hiBound, (rIdx + 1) * hopSec);
    // Guarantee minimum sensible span around a confident neural vowel peak
    if (t1 - t0 < 0.042) {
      t0 = Math.max(loBound, p.t - 0.025);
      t1 = Math.min(hiBound, p.t + 0.035);
    }
    if (t1 - t0 >= 0.025) {
      out.push({ t0, t1, label: p.label || "" });
    }
  }

  // Merge immediately adjacent vowels only if overlap/touch (< 12 ms), preserving distinct labels
  if (out.length === 0) return [];
  const merged = [{ ...out[0] }];
  for (let i = 1; i < out.length; i++) {
    const prev = merged[merged.length - 1];
    const cur = out[i];
    if (cur.t0 <= prev.t1 + 0.012) {
      prev.t1 = Math.max(prev.t1, cur.t1);
      if (cur.label && !prev.label.endsWith(cur.label) && prev.label.length < 4) {
        prev.label = `${prev.label}${cur.label}`;
      }
    } else {
      merged.push({ ...cur });
    }
  }
  return merged;
}

/**
 * Builds isolated playback audio from `samples` and vowel `spans`.
 * mode: 0 = Full, 1 = Vowels only, 2 = Consonants only
 * timingMode: 0 = In-place (exact original timing), 1 = Compact (stitched segments),
 *             2 = Stretch / Hold (TD-PSOLA pitch-locked stretch to >= 0.48s per vowel!)
 */
export function buildIsolatedAudio(
  samples,
  sampleRate,
  spans,
  mode,
  timingMode,
  rangeStartSec = 0,
  rangeEndSec = -1,
  f0Times = null,
  f0Hz = null,
) {
  const n = samples ? samples.length : 0;
  if (n === 0 || sampleRate <= 0 || mode === 0) {
    return { samples, sampleRate, timeMap: null, startSec: rangeStartSec, endSec: rangeEndSec };
  }
  const compact = timingMode === 1;
  const stretch = timingMode === 2;
  const dur = n / sampleRate;
  const r0 = Math.max(0.0, Math.min(dur, rangeStartSec || 0.0));
  const r1 = rangeEndSec > r0 ? Math.min(dur, rangeEndSec) : dur;

  const vowelSpans = (spans || []).filter((s) => s.t1 > r0 && s.t0 < r1);

  // Build active kept segments in [r0, r1]
  let kept = [];
  if (mode === 1) {
    // Vowels only
    kept = vowelSpans.map((s) => ({
      t0: Math.max(r0, s.t0),
      t1: Math.min(r1, s.t1),
    }));
  } else {
    // Consonants only (exact complement of vowel spans within [r0, r1])
    let cursor = r0;
    for (const s of vowelSpans) {
      const a = Math.max(r0, s.t0);
      const b = Math.min(r1, s.t1);
      if (a > cursor + 0.008) {
        kept.push({ t0: cursor, t1: a });
      }
      cursor = Math.max(cursor, b);
    }
    if (r1 > cursor + 0.008) {
      kept.push({ t0: cursor, t1: r1 });
    }
  }

  // Filter out silent-only segments when compacting/stretching Consonants so dead air isn't kept
  if ((compact || stretch) && mode === 2 && kept.length > 0) {
    let globalPeak = 1e-5;
    for (let i = 0; i < n; i += 4) {
      const a = Math.abs(samples[i]);
      if (a > globalPeak) globalPeak = a;
    }
    const thresh = globalPeak * 0.035;
    const trimmed = [];
    const step = Math.max(1, Math.round(sampleRate * 0.005));
    for (const seg of kept) {
      const i0 = Math.max(0, Math.floor(seg.t0 * sampleRate));
      const i1 = Math.min(n, Math.ceil(seg.t1 * sampleRate));
      let firstActive = -1;
      let lastActive = -1;
      for (let i = i0; i < i1; i += step) {
        let pk = 0;
        const e = Math.min(i1, i + step);
        for (let k = i; k < e; k++) {
          const a = Math.abs(samples[k]);
          if (a > pk) pk = a;
        }
        if (pk >= thresh) {
          if (firstActive < 0) firstActive = i;
          lastActive = e;
        }
      }
      if (firstActive >= 0 && lastActive - firstActive >= sampleRate * 0.018) {
        trimmed.push({
          t0: Math.max(seg.t0, firstActive / sampleRate - 0.012),
          t1: Math.min(seg.t1, lastActive / sampleRate + 0.012),
        });
      }
    }
    kept = trimmed;
  }

  if (!compact && !stretch) {
    // In-place smooth raised-cosine mask over the entire clip
    const out = new Float32Array(n);
    const fadeSamp = Math.max(16, Math.round(sampleRate * 0.014));
    // Build mask directly from vowel spans for 100% complementary gating
    const mask = new Float32Array(n);
    for (const s of spans || []) {
      const a = Math.max(0, Math.floor(s.t0 * sampleRate));
      const b = Math.min(n, Math.ceil(s.t1 * sampleRate));
      if (b <= a) continue;
      const fIn = Math.min(fadeSamp, (b - a) >> 1);
      const fOut = Math.min(fadeSamp, (b - a) >> 1);
      for (let i = a; i < b; i++) {
        let g = 1.0;
        const dL = i - a;
        const dR = b - 1 - i;
        if (dL < fIn) {
          g = Math.min(g, 0.5 - 0.5 * Math.cos((Math.PI * (dL + 0.5)) / fIn));
        }
        if (dR < fOut) {
          g = Math.min(g, 0.5 - 0.5 * Math.cos((Math.PI * (dR + 0.5)) / fOut));
        }
        if (g > mask[i]) mask[i] = g;
      }
    }
    if (mode === 1) {
      for (let i = 0; i < n; i++) out[i] = samples[i] * mask[i];
    } else {
      for (let i = 0; i < n; i++) out[i] = samples[i] * (1.0 - mask[i]);
    }
    return { samples: out, sampleRate, timeMap: null, startSec: r0, endSec: r1 };
  }

  // Compacted OR Stretched mode
  if (kept.length === 0) {
    const empty = new Float32Array(Math.max(16, Math.round(sampleRate * 0.05)));
    return { samples: empty, sampleRate, timeMap: [{ out0: 0, out1: empty.length / sampleRate, src0: r0, src1: r0 }], startSec: 0, endSec: empty.length / sampleRate };
  }

  const gapSamp = Math.round(sampleRate * (stretch ? 0.058 : 0.024));
  const fadeSamp = Math.max(16, Math.round(sampleRate * (stretch ? 0.022 : 0.016)));
  const slices = [];
  let totalLen = 0;
  for (let idx = 0; idx < kept.length; idx++) {
    const seg = kept[idx];
    const segDur = seg.t1 - seg.t0;
    let buf = null;
    if (stretch) {
      // Stretch short vowels up to >= 0.48s (or 2.4x, capped at 1.15s) via pitch-locked TD-PSOLA!
      const targetDur = Math.min(1.15, Math.max(0.48, segDur * 2.4));
      buf = stretchVowelSegmentPsola(samples, sampleRate, seg.t0, seg.t1, targetDur, f0Times, f0Hz);
    } else {
      const i0 = Math.max(0, Math.floor(seg.t0 * sampleRate));
      const i1 = Math.min(n, Math.ceil(seg.t1 * sampleRate));
      if (i1 - i0 >= 16) buf = samples.subarray(i0, i1);
    }
    if (!buf || buf.length < 16) continue;
    if (slices.length > 0) totalLen += gapSamp;
    slices.push({ buf, len: buf.length, outOffset: totalLen, src0: seg.t0, src1: seg.t1 });
    totalLen += buf.length;
  }

  if (totalLen < 16) {
    const empty = new Float32Array(Math.max(16, Math.round(sampleRate * 0.05)));
    return { samples: empty, sampleRate, timeMap: null, startSec: 0, endSec: empty.length / sampleRate };
  }

  const out = new Float32Array(totalLen);
  const timeMap = [];
  for (const sl of slices) {
    const fLen = Math.min(fadeSamp, sl.len >> 1);
    for (let k = 0; k < sl.len; k++) {
      let g = 1.0;
      if (k < fLen) {
        g = 0.5 - 0.5 * Math.cos((Math.PI * (k + 0.5)) / fLen);
      } else if (sl.len - 1 - k < fLen) {
        const d = sl.len - 1 - k;
        g = 0.5 - 0.5 * Math.cos((Math.PI * (d + 0.5)) / fLen);
      }
      out[sl.outOffset + k] = sl.buf[k] * g;
    }
    timeMap.push({
      out0: sl.outOffset / sampleRate,
      out1: (sl.outOffset + sl.len) / sampleRate,
      src0: sl.src0,
      src1: sl.src1,
    });
  }

  return {
    samples: out,
    sampleRate,
    timeMap,
    startSec: 0.0,
    endSec: totalLen / sampleRate,
  };
}

export function mapCompactTimeToSource(timeMap, outSec) {
  if (!timeMap || timeMap.length === 0) return outSec;
  if (outSec <= timeMap[0].out0) return timeMap[0].src0;
  for (let i = 0; i < timeMap.length; i++) {
    const seg = timeMap[i];
    if (outSec <= seg.out1) {
      const span = Math.max(1e-6, seg.out1 - seg.out0);
      const u = Math.max(0.0, Math.min(1.0, (outSec - seg.out0) / span));
      return seg.src0 + u * (seg.src1 - seg.src0);
    }
    if (i + 1 < timeMap.length && outSec < timeMap[i + 1].out0) {
      return seg.src1;
    }
  }
  return timeMap[timeMap.length - 1].src1;
}
