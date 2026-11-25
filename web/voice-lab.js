// Prism Voice Lab DSP & Klatt Formant Vowel Synthesizer
//   - TD-PSOLA pitch-locked vowel time-stretcher (Hold / Stretch mode)
//   - Klatt-style cascade 5-formant vowel synthesizer (real-time F0/F1/F2/F3/F4/weight)
//   - Source-Filter Voice Lab transforms:
//       1: Remove F0 · Whisperize (pure vocal-tract spectral envelope excited by noise)
//       2: Flatten Pitch · Monotone (TD-PSOLA pitch lock preserving size & weight)
//       3: Pure Glottal Source (inverse cepstral filter — removes formants, keeps buzz/weight)
//       4: Solo F1 (dynamic formant-tracking bandpass)
//       5: Solo F2 (dynamic formant-tracking bandpass)
//       6: Solo F3+ (upper brightness / twang formant band)
//       7: Breathiness / Noise Only (harmonic-comb subtraction along k*F0)

function makeFft(n) {
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
    hann[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / n));
  }
  for (let i = 0; i < (n >> 1); i++) {
    const a = (-2.0 * Math.PI * i) / n;
    cos[i] = Math.cos(a);
    sin[i] = Math.sin(a);
  }
  return { n, bitRev, cos, sin, hann };
}

const FFT_MAP = new Map();
function getFft(n) {
  let t = FFT_MAP.get(n);
  if (!t) {
    t = makeFft(n);
    FFT_MAP.set(n, t);
  }
  return t;
}

function fftInPlace(re, im, tbl, inverse) {
  const n = tbl.n;
  const bitRev = tbl.bitRev;
  for (let i = 0; i < n; i++) {
    const j = bitRev[i];
    if (j > i) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr;
      const ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  const cos = tbl.cos;
  const sin = tbl.sin;
  const sign = inverse ? -1.0 : 1.0;
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = n / size;
    for (let i = 0; i < n; i += size) {
      let tw = 0;
      for (let j = 0; j < half; j++) {
        const wr = cos[tw];
        const wi = sign * sin[tw];
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
  if (inverse) {
    const inv = 1.0 / n;
    for (let i = 0; i < n; i++) {
      re[i] *= inv;
      im[i] *= inv;
    }
  }
}

function lookupPitch(times, hz, t) {
  const n = times ? times.length : 0;
  if (n === 0) return 0.0;
  if (t <= times[0]) return hz[0] > 45.0 ? hz[0] : 0.0;
  if (t >= times[n - 1]) return hz[n - 1] > 45.0 ? hz[n - 1] : 0.0;
  let lo = 0;
  let hi = n - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  const i1 = lo;
  const i0 = Math.max(0, lo - 1);
  const f0 = hz[i0];
  const f1 = hz[i1];
  const dt = times[i1] - times[i0];
  const a = dt > 1e-6 ? (t - times[i0]) / dt : 0.0;
  if (f0 > 45.0 && f1 > 45.0) {
    const r = f1 / f0;
    if (r > 0.65 && r < 1.55) return f0 * Math.pow(r, a);
    return a < 0.5 ? f0 : f1;
  }
  return a < 0.5 ? (f0 > 45.0 ? f0 : 0.0) : (f1 > 45.0 ? f1 : 0.0);
}

function lookupLinear(times, vals, t) {
  const n = times ? times.length : 0;
  if (n === 0 || !vals) return 0.0;
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

// ============================================================================
// 1. Klatt-Style Cascade 5-Formant Vowel Synthesizer
// ============================================================================

/**
 * Synthesizes a clean, warm, human-sounding sustained vowel from (F0, F1, F2, F3, F4)
 * using a Rosenberg-C glottal source + subtle aspiration noise + 5 cascaded resonators.
 */
export function synthesizeKlattVowel(
  f0Hz,
  f1Hz,
  f2Hz,
  f3Hz,
  f4Hz,
  durSec = 0.62,
  tiltDb = 0.0,
  sampleRate = 48000,
) {
  const sr = Math.max(16000, Math.round(sampleRate || 48000));
  const dur = Math.max(0.18, Math.min(2.5, durSec || 0.62));
  const n = Math.round(sr * dur);
  const f0Base = Math.max(65.0, Math.min(620.0, f0Hz > 45.0 ? f0Hz : 195.0));

  const f1 = Math.max(200.0, Math.min(1150.0, f1Hz > 150.0 ? f1Hz : 520.0));
  const f2 = Math.max(f1 + 180.0, Math.min(3100.0, f2Hz > f1 + 100.0 ? f2Hz : 1650.0));
  const f3 = Math.max(f2 + 250.0, Math.min(4400.0, f3Hz > f2 + 150.0 ? f3Hz : 2750.0));
  const f4 = Math.max(f3 + 550.0, Math.min(5400.0, f4Hz > f3 + 300.0 ? f4Hz : Math.max(3550.0, f3 + 850.0)));
  const f5 = Math.min(sr * 0.44, Math.max(4450.0, f4 + 900.0));

  // Physiological formant bandwidths (Hz)
  const formants = [
    { f: f1, bw: 52.0 + 0.045 * f1 },
    { f: f2, bw: 72.0 + 0.040 * f2 },
    { f: f3, bw: 110.0 + 0.032 * f3 },
    { f: f4, bw: 185.0 },
    { f: f5, bw: 270.0 },
  ];

  // Cascade 2nd-order digital resonator coefficients: y[n] = g*x[n] + a1*y[n-1] + a2*y[n-2]
  const res = formants.map(({ f, bw }) => {
    const r = Math.exp((-Math.PI * bw) / sr);
    const omega = (2.0 * Math.PI * f) / sr;
    const a1 = 2.0 * r * Math.cos(omega);
    const a2 = -(r * r);
    // Unity DC gain keeps cascaded amplitudes well-behaved across any F1/F2 combo
    const g = 1.0 - a1 - a2;
    return { g, a1, a2, y1: 0.0, y2: 0.0 };
  });

  // Spectral tilt one-pole lowpass to control vocal weight (heavier = brighter buzz, lighter = softer slope)
  // tiltDb > 0 -> lighter / softer upper harmonics; tiltDb < 0 -> heavier / buzzier
  const tiltClamped = Math.max(-10.0, Math.min(12.0, tiltDb || 0.0));
  const tiltCutHz = Math.max(350.0, Math.min(6000.0, 1600.0 * Math.pow(2.0, -tiltClamped / 6.0)));
  const tiltAlpha = Math.exp((-2.0 * Math.PI * tiltCutHz) / sr);

  // Rosenberg-C glottal pulse parameters (open quotient OQ ~ 0.56-0.68 depending on weight)
  const oq = Math.max(0.42, Math.min(0.76, 0.58 + tiltClamped * 0.012));
  const tp = oq * 0.72; // peak opening fraction
  const tn = oq;        // closing instant fraction

  const out = new Float32Array(n);
  let phase = 0.0;
  let prevGlot = 0.0;
  let tiltY = 0.0;
  let prevLip = 0.0;
  // Deterministic fast LCG noise for gentle breathiness
  let seed = 0x51a7c3d1 >>> 0;
  const rand = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return (seed / 4294967296.0) * 2.0 - 1.0;
  };
  let pinkB0 = 0.0;
  const breathAmt = Math.max(0.012, Math.min(0.08, 0.024 + tiltClamped * 0.003));

  const fadeIn = Math.min(n >> 2, Math.round(sr * 0.04));
  const fadeOut = Math.min(n >> 2, Math.round(sr * 0.065));

  for (let i = 0; i < n; i++) {
    const t = i / sr;
    // Gentle organic micro-vibrato (~5.2 Hz, ±0.45% pitch) so it sounds unmistakably vocal
    const vib = 1.0 + 0.0045 * Math.sin(2.0 * Math.PI * 5.2 * t) * Math.min(1.0, t / 0.15);
    const f0 = f0Base * vib;
    phase += f0 / sr;
    if (phase >= 1.0) phase -= Math.floor(phase);

    // Rosenberg smooth glottal flow waveform
    let glot = 0.0;
    if (phase < tp) {
      const u = phase / tp;
      glot = 0.5 * (1.0 - Math.cos(Math.PI * u));
    } else if (phase < tn) {
      const u = (phase - tp) / (tn - tp);
      glot = Math.cos(0.5 * Math.PI * u);
    }

    // Glottal flow derivative (excites the vocal tract)
    let src = (glot - prevGlot) * (sr / 8000.0);
    prevGlot = glot;

    // Gentle pink-ish aspiration noise
    pinkB0 = 0.85 * pinkB0 + 0.15 * rand();
    src += pinkB0 * breathAmt;

    // Weight spectral tilt
    tiltY = (1.0 - tiltAlpha) * src + tiltAlpha * tiltY;
    let x = tiltY;

    // Cascade 5 vocal-tract formant resonators
    for (let rIdx = 0; rIdx < res.length; rIdx++) {
      const flt = res[rIdx];
      const y = flt.g * x + flt.a1 * flt.y1 + flt.a2 * flt.y2;
      flt.y2 = flt.y1;
      flt.y1 = y;
      x = y;
    }

    // Lip radiation (+6 dB/oct slight high-shelf lift)
    const rad = x - 0.82 * prevLip;
    prevLip = x;

    let env = 1.0;
    if (i < fadeIn) env = 0.5 - 0.5 * Math.cos((Math.PI * (i + 0.5)) / fadeIn);
    else if (n - 1 - i < fadeOut) env = 0.5 - 0.5 * Math.cos((Math.PI * (n - 0.5 - i)) / fadeOut);

    out[i] = rad * env;
  }

  // Normalize to clean -8 dBFS peak so Klatt playback matches natural clip loudness
  let peak = 1e-6;
  for (let i = 0; i < n; i++) {
    const a = Math.abs(out[i]);
    if (a > peak) peak = a;
  }
  const scale = 0.42 / peak;
  for (let i = 0; i < n; i++) out[i] *= scale;

  return out;
}

// ============================================================================
// 2. TD-PSOLA Pitch-Locked Time-Stretcher (for Hold / Stretch Vowel Mode)
// ============================================================================

function findSegmentEpochs(samples, sr, startSamp, endSamp, f0Times, f0Hz) {
  const len = endSamp - startSamp;
  if (len < 32) return { marks: [], periods: [] };

  // Low-pass filter at 550 Hz to expose clean fundamental glottal peaks
  const lp = new Float32Array(len);
  const alpha = Math.min(0.95, Math.exp((-2.0 * Math.PI * 550.0) / sr));
  let y = 0.0;
  for (let i = 0; i < len; i++) {
    y = alpha * y + (1.0 - alpha) * samples[startSamp + i];
    lp[i] = y;
  }

  // Estimate fallback pitch inside the segment if f0Times is missing or sparse
  let fallbackF0 = 0.0;
  if (f0Times && f0Times.length > 0) {
    const vals = [];
    const t0 = startSamp / sr;
    const t1 = endSamp / sr;
    for (let i = 0; i < f0Times.length; i++) {
      if (f0Times[i] >= t0 && f0Times[i] <= t1 && f0Hz[i] > 50.0) vals.push(f0Hz[i]);
    }
    if (vals.length > 0) {
      vals.sort((a, b) => a - b);
      fallbackF0 = vals[vals.length >> 1];
    }
  }
  if (fallbackF0 < 50.0) {
    // Fast ACF pitch estimate on middle of segment
    const mid = len >> 1;
    const w = Math.min(len, Math.round(sr * 0.032));
    const s0 = Math.max(0, mid - (w >> 1));
    const minLag = Math.max(8, Math.floor(sr / 550));
    const maxLag = Math.min(w - 8, Math.ceil(sr / 70));
    let bestLag = Math.round(sr / 180);
    let bestR = -1e9;
    const step = sr > 24000 ? 2 : 1;
    for (let lag = minLag; lag <= maxLag; lag += step) {
      let num = 0.0;
      let ea = 0.0;
      let eb = 0.0;
      for (let i = 0; i < w - lag; i += step) {
        const a = lp[s0 + i];
        const b = lp[s0 + i + lag];
        num += a * b;
        ea += a * a;
        eb += b * b;
      }
      const r = num / (Math.sqrt(ea * eb) + 1e-12);
      if (r > bestR) {
        bestR = r;
        bestLag = lag;
      }
    }
    fallbackF0 = sr / bestLag;
  }

  const marks = [];
  const periods = [];
  let pos = Math.min(len - 1, Math.round((sr / fallbackF0) * 0.5));
  while (pos < len - 8) {
    const tSec = (startSamp + pos) / sr;
    const f0 = lookupPitch(f0Times, f0Hz, tSec);
    const useF0 = f0 > 50.0 && f0 < 750.0 ? f0 : fallbackF0;
    const T0 = sr / useF0;
    const halfSearch = Math.max(4, Math.round(T0 * 0.38));
    const j0 = Math.max(0, pos - halfSearch);
    const j1 = Math.min(len - 1, pos + halfSearch);
    let best = pos;
    let bestVal = -1e30;
    for (let j = j0; j <= j1; j++) {
      if (lp[j] > bestVal) {
        bestVal = lp[j];
        best = j;
      }
    }
    if (marks.length > 0 && best <= marks[marks.length - 1] + Math.max(4, Math.round(T0 * 0.45))) {
      best = Math.min(len - 1, marks[marks.length - 1] + Math.round(T0));
    }
    marks.push(best);
    periods.push(T0);
    pos = best + Math.round(T0);
  }
  return { marks, periods };
}

/**
 * Stretches a vowel segment `[t0, t1]` to `targetDurSec` via pitch-synchronous TD-PSOLA,
 * keeping pitch, formants, and vocal weight 100% locked!
 * Emphasizes stretching the steady-state core (15%..85%) so onsets/offsets remain crisp.
 */
export function stretchVowelSegmentPsola(samples, sr, t0, t1, targetDurSec, f0Times = null, f0Hz = null) {
  const n = samples.length;
  const i0 = Math.max(0, Math.floor(t0 * sr));
  const i1 = Math.min(n, Math.ceil(t1 * sr));
  const srcLen = i1 - i0;
  if (srcLen < 64) return samples.subarray(i0, i1).slice();

  const srcDur = srcLen / sr;
  const outDur = Math.max(srcDur, targetDurSec);
  const outLen = Math.max(srcLen, Math.round(outDur * sr));

  if (outLen <= srcLen + 32) {
    return samples.subarray(i0, i1).slice();
  }

  const { marks, periods } = findSegmentEpochs(samples, sr, i0, i1, f0Times, f0Hz);
  if (marks.length < 3) {
    // Very short clip fallback: simple hann-windowed OLA repeat
    const out = new Float32Array(outLen);
    const wsum = new Float32Array(outLen);
    const grain = Math.min(srcLen, Math.round(sr * 0.025));
    const hopOut = Math.max(16, grain >> 2);
    for (let op = 0; op < outLen; op += hopOut) {
      const frac = op / Math.max(1, outLen - 1);
      const sp = Math.round(frac * Math.max(0, srcLen - grain));
      for (let k = 0; k < grain && op + k < outLen; k++) {
        const w = 0.5 - 0.5 * Math.cos((2.0 * Math.PI * k) / Math.max(1, grain - 1));
        out[op + k] += samples[i0 + sp + k] * w;
        wsum[op + k] += w;
      }
    }
    for (let i = 0; i < outLen; i++) if (wsum[i] > 1e-4) out[i] /= wsum[i];
    return out;
  }

  // Non-linear time-warp curve: keep first/last 12% close to 1.25x, stretch middle core smoothly
  const mapOutFracToSrcFrac = (u) => {
    const edgeOut = 0.10;
    const edgeSrc = 0.14;
    if (u <= edgeOut) return (u / edgeOut) * edgeSrc;
    if (u >= 1.0 - edgeOut) return 1.0 - ((1.0 - u) / edgeOut) * edgeSrc;
    const midU = (u - edgeOut) / (1.0 - 2.0 * edgeOut);
    return edgeSrc + midU * (1.0 - 2.0 * edgeSrc);
  };

  const out = new Float32Array(outLen);
  const wsum = new Float32Array(outLen);

  let outPos = marks[0];
  let mIdx = 0;
  let safety = 0;
  const maxGrains = Math.ceil(outLen / 12) + 64;

  while (outPos < outLen && safety++ < maxGrains) {
    const u = Math.max(0.0, Math.min(1.0, outPos / Math.max(1, outLen - 1)));
    const targetSrc = mapOutFracToSrcFrac(u) * (srcLen - 1);

    // Find nearest analysis epoch mark
    while (mIdx + 1 < marks.length && marks[mIdx + 1] < targetSrc) mIdx++;
    let bestIdx = mIdx;
    if (mIdx + 1 < marks.length && Math.abs(marks[mIdx + 1] - targetSrc) < Math.abs(marks[mIdx] - targetSrc)) {
      bestIdx = mIdx + 1;
    }

    const centerSrc = i0 + marks[bestIdx];
    const T0 = periods[bestIdx];
    const halfWin = Math.max(8, Math.round(T0));
    const winSpan = halfWin * 2;
    const outCenter = Math.round(outPos);

    for (let k = -halfWin; k <= halfWin; k++) {
      const oi = outCenter + k;
      if (oi < 0 || oi >= outLen) continue;
      const si = centerSrc + k;
      if (si < 0 || si >= n) continue;
      const w = 0.5 * (1.0 + Math.cos((Math.PI * k) / halfWin));
      out[oi] += samples[si] * w;
      wsum[oi] += w;
    }
    outPos += T0;
  }

  for (let i = 0; i < outLen; i++) {
    if (wsum[i] > 1e-4) out[i] /= wsum[i];
  }
  return out;
}

// ============================================================================
// 3. Voice Lab Source-Filter DSP Transforms
// ============================================================================

/**
 * Computes smooth log-magnitude cepstral vocal-tract spectral envelope `outEnv[0..bins-1]`
 * (linear amplitude) from complex STFT frame `(re, im)`.
 */
function computeCepstralEnvelope(re, im, tbl, cepRe, cepIm, outEnv, lifterBins) {
  const n = tbl.n;
  const bins = (n >> 1) + 1;
  for (let k = 0; k < bins; k++) {
    const mag = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
    const lm = Math.log(mag + 1e-7);
    cepRe[k] = lm;
    cepIm[k] = 0.0;
    if (k > 0 && k < n - k) {
      cepRe[n - k] = lm;
      cepIm[n - k] = 0.0;
    }
  }
  fftInPlace(cepRe, cepIm, tbl, true);
  // Smooth raised-cosine low-time lifter to keep vocal tract envelope and remove pitch harmonics
  const cut = Math.max(6, Math.min((n >> 1) - 2, lifterBins));
  const trans = Math.max(4, cut >> 2);
  for (let q = 0; q < n; q++) {
    const dist = q <= (n >> 1) ? q : n - q;
    let w = 1.0;
    if (dist > cut + trans) w = 0.0;
    else if (dist > cut) {
      w = 0.5 * (1.0 + Math.cos((Math.PI * (dist - cut)) / trans));
    }
    cepRe[q] *= w;
    cepIm[q] = 0.0;
  }
  fftInPlace(cepRe, cepIm, tbl, false);
  for (let k = 0; k < bins; k++) {
    outEnv[k] = Math.exp(Math.max(-18.0, Math.min(18.0, cepRe[k])));
  }
}

function medianVoicedPitch(f0Hz) {
  if (!f0Hz || f0Hz.length === 0) return 190.0;
  const vals = [];
  for (let i = 0; i < f0Hz.length; i++) {
    if (f0Hz[i] > 55.0 && f0Hz[i] < 650.0) vals.push(f0Hz[i]);
  }
  if (vals.length === 0) return 190.0;
  vals.sort((a, b) => a - b);
  return vals[vals.length >> 1];
}

// Monotone TD-PSOLA: lock every voiced frame to a flat pitch `targetHz`, keeping size & weight untouched
function applyMonotonePsola(samples, sr, f0Times, f0Hz, targetHz) {
  const n = samples.length;
  const lockHz = targetHz > 55.0 && targetHz < 650.0 ? targetHz : medianVoicedPitch(f0Hz);
  const targetT0 = sr / lockHz;

  // Low-pass filter for epoch picking
  const lp = new Float32Array(n);
  const alpha = Math.min(0.95, Math.exp((-2.0 * Math.PI * 550.0) / sr));
  let y = 0.0;
  for (let i = 0; i < n; i++) {
    y = alpha * y + (1.0 - alpha) * samples[i];
    lp[i] = y;
  }

  const marks = [];
  const periods = [];
  const voiced = [];
  let pos = Math.round(0.01 * sr);
  const defT0 = Math.round(sr / 160.0);
  while (pos < n - 16) {
    const f0 = lookupPitch(f0Times, f0Hz, pos / sr);
    if (f0 > 50.0 && f0 < 750.0) {
      const T0 = sr / f0;
      const hs = Math.max(4, Math.round(T0 * 0.35));
      const s0 = Math.max(0, pos - hs);
      const s1 = Math.min(n - 1, pos + hs);
      let best = pos;
      let bestV = -1e30;
      for (let k = s0; k <= s1; k++) {
        if (lp[k] > bestV) {
          bestV = lp[k];
          best = k;
        }
      }
      if (marks.length > 0 && best <= marks[marks.length - 1] + Math.max(4, Math.round(T0 * 0.45))) {
        best = Math.min(n - 1, marks[marks.length - 1] + Math.round(T0));
      }
      marks.push(best);
      periods.push(T0);
      voiced.push(1);
      pos = best + Math.round(T0);
    } else {
      marks.push(pos);
      periods.push(defT0);
      voiced.push(0);
      pos += defT0;
    }
  }
  if (marks.length < 3) return samples.slice();

  const out = new Float32Array(n);
  const wsum = new Float32Array(n);
  let synPos = marks[0];
  let mIdx = 0;
  while (synPos < n) {
    while (mIdx + 1 < marks.length && marks[mIdx + 1] < synPos) mIdx++;
    let best = mIdx;
    if (mIdx + 1 < marks.length && Math.abs(marks[mIdx + 1] - synPos) < Math.abs(marks[mIdx] - synPos)) {
      best = mIdx + 1;
    }
    const isV = voiced[best] === 1;
    const srcCenter = marks[best];
    const T0 = periods[best];
    const halfWin = Math.max(8, Math.round(isV ? T0 : defT0));
    const oc = Math.round(synPos);
    for (let k = -halfWin; k <= halfWin; k++) {
      const oi = oc + k;
      const si = srcCenter + k;
      if (oi < 0 || oi >= n || si < 0 || si >= n) continue;
      const w = 0.5 * (1.0 + Math.cos((Math.PI * k) / halfWin));
      out[oi] += samples[si] * w;
      wsum[oi] += w;
    }
    synPos += isV ? targetT0 : defT0;
  }
  for (let i = 0; i < n; i++) {
    out[i] = wsum[i] > 1e-4 ? out[i] / wsum[i] : samples[i];
  }
  return out;
}

/**
 * Applies a Voice Lab acoustic transform (`labMode` 1..7) cleanly to `samples`.
 *   1: Remove F0 · Whisperize (pure vocal tract resonance)
 *   2: Flatten Pitch · Monotone (TD-PSOLA pitch lock)
 *   3: Pure Glottal Source (inverse cepstral filter — formants removed)
 *   4: Solo F1 (dynamic formant-tracking bandpass)
 *   5: Solo F2 (dynamic formant-tracking bandpass)
 *   6: Solo F3+ (upper brightness / twang formant band)
 *   7: Breathiness / Noise Only (harmonic-comb removal along k*F0)
 */
export function applyVoiceLabTransform(
  samples,
  sampleRate,
  labMode,
  f0Times = null,
  f0Hz = null,
  f0Conf = null,
  fmtTimes = null,
  fmtF1 = null,
  fmtF2 = null,
  fmtF3 = null,
  targetMonoHz = 0.0,
) {
  const n = samples ? samples.length : 0;
  if (n < 64 || sampleRate <= 0 || labMode === 0) return samples;

  if (labMode === 2) {
    return applyMonotonePsola(samples, sampleRate, f0Times, f0Hz, targetMonoHz);
  }

  // STFT engine (2048 for sharp harmonic comb in Breathiness mode; 1024 for cepstral & formants)
  const nfft = labMode === 7 ? 2048 : 1024;
  const hop = nfft >> 2;
  const bins = (nfft >> 1) + 1;
  const binHz = sampleRate / nfft;
  const tbl = getFft(nfft);

  const re = new Float32Array(nfft);
  const im = new Float32Array(nfft);
  const cepRe = new Float32Array(nfft);
  const cepIm = new Float32Array(nfft);
  const env = new Float32Array(bins);

  const out = new Float32Array(n);
  const wsum = new Float32Array(n);

  // Cepstral lifter cutoff: ~1.5 ms keeps vocal tract formants & strips pitch harmonics above ~110 Hz
  const lifterBins = Math.max(10, Math.round(sampleRate * 0.00145));

  // Deterministic LCG for whisper random phase
  let seed = 0x2f6e2b1 >>> 0;
  const nextRandAngle = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return (seed / 4294967296.0) * 2.0 * Math.PI;
  };

  for (let pos = -nfft + hop; pos < n; pos += hop) {
    let inEnergy = 0.0;
    for (let i = 0; i < nfft; i++) {
      const idx = pos + i;
      const v = idx >= 0 && idx < n ? samples[idx] : 0.0;
      const wv = v * tbl.hann[i];
      re[i] = wv;
      im[i] = 0.0;
      inEnergy += wv * wv;
    }
    if (inEnergy < 1e-12) continue;

    const tSec = (pos + (nfft >> 1)) / sampleRate;
    fftInPlace(re, im, tbl, false);

    if (labMode === 1) {
      // Remove F0 · Whisperize: vocal-tract cepstral envelope * random-phase broadband noise
      computeCepstralEnvelope(re, im, tbl, cepRe, cepIm, env, lifterBins);
      let outPow = 0.0;
      for (let k = 0; k < bins; k++) {
        const hz = k * binHz;
        // High-pass gently below 180 Hz so sub-bass rumble doesn't muddy the whispered resonance
        const hp = hz < 180.0 ? (hz / 180.0) * (hz / 180.0) : 1.0;
        const mag = env[k] * hp;
        const ang = nextRandAngle();
        const rv = mag * Math.cos(ang);
        const iv = mag * Math.sin(ang);
        re[k] = rv;
        im[k] = iv;
        if (k > 0 && k < nfft - k) {
          re[nfft - k] = rv;
          im[nfft - k] = -iv;
          outPow += 2.0 * mag * mag;
        } else {
          outPow += mag * mag;
        }
      }
      // Match original frame RMS energy cleanly
      const scale = Math.sqrt((inEnergy * nfft) / Math.max(1e-12, outPow)) * 0.92;
      for (let i = 0; i < nfft; i++) {
        re[i] *= scale;
        im[i] *= scale;
      }
    } else if (labMode === 3) {
      // Pure Glottal Source: inverse-filter out vocal-tract spectral envelope!
      computeCepstralEnvelope(re, im, tbl, cepRe, cepIm, env, lifterBins);
      // Compute mean envelope level to prevent division explosions in dead stopbands
      let maxEnv = 1e-6;
      for (let k = 1; k < bins; k++) if (env[k] > maxEnv) maxEnv = env[k];
      const floor = maxEnv * 0.04;
      let outPow = 0.0;
      for (let k = 0; k < bins; k++) {
        const hz = k * binHz;
        // Divide by vocal-tract envelope & apply natural glottal -6 dB/oct rolloff above 350 Hz
        const inv = 1.0 / Math.max(floor, env[k]);
        const roll = hz > 350.0 ? Math.pow(350.0 / hz, 0.85) : 1.0;
        const lp = hz > 4800.0 ? Math.max(0.0, 1.0 - (hz - 4800.0) / 1400.0) : 1.0;
        const g = inv * roll * lp;
        re[k] *= g;
        im[k] *= g;
        const p = re[k] * re[k] + im[k] * im[k];
        if (k > 0 && k < nfft - k) {
          re[nfft - k] = re[k];
          im[nfft - k] = -im[k];
          outPow += 2.0 * p;
        } else {
          outPow += p;
        }
      }
      const scale = Math.sqrt((inEnergy * nfft) / Math.max(1e-12, outPow)) * 0.85;
      for (let i = 0; i < nfft; i++) {
        re[i] *= scale;
        im[i] *= scale;
      }
    } else if (labMode === 4 || labMode === 5 || labMode === 6) {
      // Dynamic formant-tracking bandpass: Solo F1 / Solo F2 / Solo F3+
      const rawF1 = lookupLinear(fmtTimes, fmtF1, tSec);
      const rawF2 = lookupLinear(fmtTimes, fmtF2, tSec);
      const rawF3 = lookupLinear(fmtTimes, fmtF3, tSec);
      const f1 = rawF1 >= 180 && rawF1 <= 1150 ? rawF1 : 520.0;
      const f2 = rawF2 >= 650 && rawF2 <= 3100 ? rawF2 : 1650.0;
      const f3 = rawF3 >= 1700 && rawF3 <= 4500 ? rawF3 : 2750.0;

      for (let k = 0; k < bins; k++) {
        const hz = k * binHz;
        let g = 0.0;
        if (labMode === 4) {
          const d = (hz - f1) / 155.0;
          g = Math.exp(-0.5 * d * d);
        } else if (labMode === 5) {
          const d = (hz - f2) / 245.0;
          g = Math.exp(-0.5 * d * d) * 1.45;
        } else {
          // F3+ upper brilliance & twang band
          const lo = Math.max(1900.0, f3 - 380.0);
          if (hz < lo) {
            const d = (lo - hz) / 240.0;
            g = Math.exp(-0.5 * d * d);
          } else if (hz > 5600.0) {
            const d = (hz - 5600.0) / 700.0;
            g = Math.exp(-0.5 * d * d);
          } else {
            g = 1.0;
          }
          g *= 1.85;
        }
        re[k] *= g;
        im[k] *= g;
        if (k > 0 && k < nfft - k) {
          re[nfft - k] = re[k];
          im[nfft - k] = -im[k];
        }
      }
    } else if (labMode === 7) {
      // Breathiness / Noise Only: harmonic comb filter notches out k*F0 on voiced frames
      const f0 = lookupPitch(f0Times, f0Hz, tSec);
      const conf = f0Conf && f0Times ? lookupLinear(f0Times, f0Conf, tSec) : (f0 > 50 ? 0.8 : 0.0);
      if (f0 > 55.0 && f0 < 750.0 && conf >= 0.35) {
        const notchW = Math.max(binHz * 1.35, f0 * 0.24);
        for (let k = 0; k < bins; k++) {
          const hz = k * binHz;
          let g = 1.0;
          if (hz < f0 * 0.65) {
            g = 0.08;
          } else if (hz < Math.min(5800.0, f0 * 24.0)) {
            const nearestH = Math.max(1, Math.round(hz / f0));
            const dist = Math.abs(hz - nearestH * f0);
            const u = dist / notchW;
            // Deep notch right on every harmonic peak, transparent in inter-harmonic valleys
            const notch = 1.0 - 0.94 * Math.exp(-0.5 * u * u);
            g = notch;
          }
          // Gentle makeup gain on residual breath
          g *= 1.55;
          re[k] *= g;
          im[k] *= g;
          if (k > 0 && k < nfft - k) {
            re[nfft - k] = re[k];
            im[nfft - k] = -im[k];
          }
        }
      }
    }

    fftInPlace(re, im, tbl, true);
    for (let i = 0; i < nfft; i++) {
      const idx = pos + i;
      if (idx >= 0 && idx < n) {
        const w = tbl.hann[i];
        out[idx] += re[i] * w;
        wsum[idx] += w * w;
      }
    }
  }

  for (let i = 0; i < n; i++) {
    if (wsum[i] > 1e-4) out[i] /= wsum[i];
  }
  // Gentle soft limiter to guarantee zero clipping after formant makeup gain
  let peak = 0.0;
  for (let i = 0; i < n; i++) {
    const a = Math.abs(out[i]);
    if (a > peak) peak = a;
  }
  if (peak > 0.92) {
    const g = 0.92 / peak;
    for (let i = 0; i < n; i++) out[i] *= g;
  }
  return out;
}
