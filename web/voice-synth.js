// Glint Browser-Local Source-Filter Voice Resynthesizer
// Applies independent, natural-sounding Pitch (TD-PSOLA), Vocal-Tract Size (Cepstral Spectral
// Envelope Warping), and Vocal Weight (Voiced Glottal Spectral Tilt) adjustments to a user's
// own voice recording, 100% locally in <100ms with zero external dependencies.

function makeRadix2Tables(n) {
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

const FFT_CACHE = new Map();
function getFftTables(n) {
  let t = FFT_CACHE.get(n);
  if (!t) {
    t = makeRadix2Tables(n);
    FFT_CACHE.set(n, t);
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

// Lookup helper over uniform or sorted (times, vals) pitch arrays
function lookupPitchAt(times, hz, t) {
  const n = times.length;
  if (n === 0) return 0.0;
  if (t <= times[0]) return hz[0] > 40.0 ? hz[0] : 0.0;
  if (t >= times[n - 1]) return hz[n - 1] > 40.0 ? hz[n - 1] : 0.0;
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
  if (f0 > 40.0 && f1 > 40.0) {
    const ratio = f1 / f0;
    if (ratio > 0.65 && ratio < 1.55) {
      return f0 * Math.pow(ratio, a);
    }
    return a < 0.5 ? f0 : f1;
  }
  return a < 0.5 ? (f0 > 40.0 ? f0 : 0.0) : (f1 > 40.0 ? f1 : 0.0);
}

// ============================================================================
// Stage 1: Pitch-Synchronous Overlap-Add (TD-PSOLA) for natural pitch shifting
// Preserves original vocal-tract formants & keeps unvoiced consonants 100% crisp.
// ============================================================================

function findEpochMarks(samples, sr, f0Times, f0Hz) {
  const n = samples.length;
  // Simple low-pass smoothed signal (highlights fundamental pulse peaks reliably)
  const lp = new Float32Array(n);
  const alpha = Math.min(0.95, Math.exp((-2.0 * Math.PI * 550.0) / sr));
  let y = 0.0;
  for (let i = 0; i < n; i++) {
    y = alpha * y + (1.0 - alpha) * samples[i];
    lp[i] = y;
  }

  const marks = [];
  const periods = [];
  const voicedFlags = [];

  let pos = Math.round(0.01 * sr);
  const defaultPeriod = Math.round(sr / 150.0);

  while (pos < n - 16) {
    const t = pos / sr;
    const f0 = lookupPitchAt(f0Times, f0Hz, t);
    if (f0 > 45.0 && f0 < 900.0) {
      const T0 = sr / f0;
      // Search around `pos` within +/- 0.35 * T0 for strongest positive peak in LP signal
      const halfSearch = Math.max(4, Math.round(T0 * 0.35));
      const s0 = Math.max(0, pos - halfSearch);
      const s1 = Math.min(n - 1, pos + halfSearch);
      let bestIdx = pos;
      let bestVal = -1e30;
      for (let k = s0; k <= s1; k++) {
        if (lp[k] > bestVal) {
          bestVal = lp[k];
          bestIdx = k;
        }
      }
      // Enforce monotonic progression
      if (marks.length > 0 && bestIdx <= marks[marks.length - 1] + Math.max(4, Math.round(T0 * 0.45))) {
        bestIdx = Math.min(n - 1, marks[marks.length - 1] + Math.round(T0));
      }
      marks.push(bestIdx);
      periods.push(T0);
      voicedFlags.push(1);
      pos = bestIdx + Math.round(T0);
    } else {
      marks.push(pos);
      periods.push(defaultPeriod);
      voicedFlags.push(0);
      pos += defaultPeriod;
    }
  }
  return { marks, periods, voicedFlags };
}

function tdPsolaShift(samples, sr, f0Times, f0Hz, refF0Hz, pitchSemi, followContour, contourBlend) {
  const n = samples.length;
  const ratioConst = Math.pow(2.0, pitchSemi / 12.0);
  const needShift = Math.abs(pitchSemi) > 0.03 || (followContour && refF0Hz && refF0Hz.length > 0 && contourBlend > 0.01);
  if (!needShift || f0Times.length < 2) {
    return new Float32Array(samples);
  }

  const { marks, periods, voicedFlags } = findEpochMarks(samples, sr, f0Times, f0Hz);
  if (marks.length < 3) {
    return new Float32Array(samples);
  }

  const out = new Float32Array(n);
  const norm = new Float32Array(n);

  // Target F0 helper at time t (either constant semitone shift or blended toward ref contour)
  function targetF0At(t, srcF0) {
    if (srcF0 <= 45.0) return 0.0;
    let tgt = srcF0 * ratioConst;
    if (followContour && refF0Hz && refF0Hz.length === f0Times.length) {
      const rf = lookupPitchAt(f0Times, refF0Hz, t);
      if (rf > 45.0 && rf < 900.0) {
        // Log-frequency blend from srcF0 -> rf by contourBlend, plus any manual pitchSemi tweak
        const blended = srcF0 * Math.pow(rf / srcF0, contourBlend);
        tgt = blended * ratioConst;
      }
    }
    return Math.max(55.0, Math.min(850.0, tgt));
  }

  let synPos = marks[0];
  let mIdx = 0;

  while (synPos < n - 8) {
    while (mIdx + 1 < marks.length && marks[mIdx + 1] < synPos) {
      mIdx++;
    }
    let bestM = mIdx;
    if (mIdx + 1 < marks.length && Math.abs(marks[mIdx + 1] - synPos) < Math.abs(marks[mIdx] - synPos)) {
      bestM = mIdx + 1;
    }

    const anaCenter = marks[bestM];
    const T0 = periods[bestM];
    const isVoiced = voicedFlags[bestM] === 1;

    if (!isVoiced) {
      // Unvoiced segment: copy with Hann crossfade at native period (100% crisp consonants!)
      const halfW = Math.round(T0);
      const len = halfW * 2;
      for (let k = -halfW; k <= halfW; k++) {
        const si = synPos + k;
        const ai = synPos + k;
        if (si >= 0 && si < n && ai >= 0 && ai < n) {
          const w = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * (k + halfW)) / Math.max(1, len)));
          out[si] += samples[ai] * w;
          norm[si] += w;
        }
      }
      synPos += Math.max(8, Math.round(T0));
      continue;
    }

    const srcF0 = lookupPitchAt(f0Times, f0Hz, synPos / sr) || (sr / Math.max(8.0, T0));
    const tgtF0 = targetF0At(synPos / sr, srcF0);
    const synT0 = tgtF0 > 45.0 ? sr / tgtF0 : T0;

    // 2-period Hann grain centered on analysis epoch `anaCenter`
    const halfW = Math.max(8, Math.round(T0));
    const denom = Math.max(1, halfW * 2);
    const synInt = Math.round(synPos);
    for (let k = -halfW; k <= halfW; k++) {
      const si = synInt + k;
      const ai = anaCenter + k;
      if (si >= 0 && si < n && ai >= 0 && ai < n) {
        const w = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * (k + halfW)) / denom));
        out[si] += samples[ai] * w;
        norm[si] += w;
      }
    }
    synPos += Math.max(8.0, synT0);
  }

  for (let i = 0; i < n; i++) {
    if (norm[i] > 1e-4) {
      out[i] /= norm[i];
    } else {
      out[i] = samples[i];
    }
  }
  return out;
}

// ============================================================================
// Stage 2: Pitch-Adaptive Cepstral Envelope Warping (Size) + Voiced Spectral Tilt (Weight)
// Zero-phase magnitude modification -> preserves harmonic phase, zero metallic phasiness!
// ============================================================================

function spectralSizeWeightProcess(samples, sr, f0Times, f0Hz, pitchSemi, sizePct, weightDb) {
  const n = samples.length;
  const sizeScale = Math.max(0.72, Math.min(1.38, 1.0 + sizePct / 100.0));
  const doSize = Math.abs(sizePct) > 0.25;
  const doWeight = Math.abs(weightDb) > 0.08;
  if (!doSize && !doWeight) {
    return new Float32Array(samples);
  }

  // Choose FFT size ~21-25 ms (1024 at 44.1/48 kHz -> 21.3ms, 512 at 16-24 kHz)
  const fftSize = sr >= 32000 ? 1024 : 512;
  const hop = fftSize >> 2; // 75% overlap for exact COLA reconstruction
  const bins = (fftSize >> 1) + 1;
  const tbl = getFftTables(fftSize);
  const hann = tbl.hann;
  const hzPerBin = sr / fftSize;

  const out = new Float32Array(n);
  const winSum = new Float32Array(n);

  const re = new Float32Array(fftSize);
  const im = new Float32Array(fftSize);
  const logMag = new Float32Array(fftSize);
  const cepIm = new Float32Array(fftSize);
  const envLog = new Float32Array(bins);
  const pitchRatio = Math.pow(2.0, pitchSemi / 12.0);

  // Subtle deterministic PRNG for soft breathiness hint when making weight much lighter
  let rng = 123456789;
  const nextRand = () => {
    rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0;
    return (rng / 4294967296.0) * 2.0 - 1.0;
  };

  for (let start = 0; start < n; start += hop) {
    let origEnergy = 0.0;
    for (let i = 0; i < fftSize; i++) {
      const idx = start + i;
      const s = idx < n ? samples[idx] : 0.0;
      const w = s * hann[i];
      re[i] = w;
      im[i] = 0.0;
      origEnergy += w * w;
    }

    if (origEnergy < 1e-10) {
      for (let i = 0; i < fftSize; i++) {
        const idx = start + i;
        if (idx < n) {
          const w2 = hann[i] * hann[i];
          winSum[idx] += w2;
        }
      }
      continue;
    }

    fftInPlace(re, im, tbl, false);

    const centerSec = (start + fftSize * 0.5) / sr;
    const rawF0 = lookupPitchAt(f0Times, f0Hz, centerSec);
    const curF0 = rawF0 > 45.0 ? Math.max(55.0, Math.min(850.0, rawF0 * pitchRatio)) : 0.0;
    const isVoiced = curF0 > 45.0;

    // 1. Compute smooth pitch-adaptive cepstral spectral envelope (log-magnitude)
    if (doSize) {
      for (let k = 0; k < bins; k++) {
        const mag = Math.hypot(re[k], im[k]);
        const lm = Math.log(mag + 1e-6);
        logMag[k] = lm;
        if (k > 0 && k < bins - 1) {
          logMag[fftSize - k] = lm;
        }
      }
      cepIm.fill(0.0);
      fftInPlace(logMag, cepIm, tbl, true); // real cepstrum

      // Pitch-adaptive lifter cutoff: keeps smooth formants, removes harmonic ripples!
      const effF0 = isVoiced ? curF0 : 220.0;
      const qCut = Math.max(10, Math.min(Math.floor(fftSize * 0.14), Math.round((0.82 * sr) / effF0)));
      const qFade = Math.max(4, Math.round(qCut * 0.25));
      for (let q = qCut; q <= fftSize - qCut; q++) {
        const dist = Math.min(q - qCut, (fftSize - qCut) - q);
        if (dist < qFade) {
          const fade = 0.5 * (1.0 + Math.cos((Math.PI * (dist + 1)) / (qFade + 1)));
          logMag[q] *= fade;
        } else {
          logMag[q] = 0.0;
        }
      }
      cepIm.fill(0.0);
      fftInPlace(logMag, cepIm, tbl, false);
      for (let k = 0; k < bins; k++) {
        envLog[k] = logMag[k];
      }
    }

    // 2. Apply Size envelope warp + Weight glottal spectral tilt
    const effSize = isVoiced ? sizeScale : 1.0 + 0.35 * (sizeScale - 1.0);
    let modEnergy = 0.0;

    for (let k = 1; k < bins - 1; k++) {
      const freq = k * hzPerBin;
      let gain = 1.0;

      if (doSize) {
        // Taper formant warp smoothly above 5.8 kHz so high air/fricatives stay natural
        const highBlend = freq < 5500.0 ? 1.0 : Math.max(0.0, 1.0 - (freq - 5500.0) / 2200.0);
        const localScale = 1.0 + highBlend * (effSize - 1.0);
        const srcBin = k / localScale;
        const b0 = Math.max(1, Math.min(bins - 2, Math.floor(srcBin)));
        const frac = Math.max(0.0, Math.min(1.0, srcBin - b0));
        const warpedLog = envLog[b0] + frac * (envLog[b0 + 1] - envLog[b0]);
        const dLog = Math.max(-1.8, Math.min(1.8, warpedLog - envLog[k]));
        gain *= Math.exp(dLog);
      }

      if (doWeight && isVoiced) {
        // Glottal weight tilt: positive weightDb = lighter (boost H1, soften H2..H8);
        // negative weightDb = heavier/fuller (boost H2..H8 relative to H1).
        const h1Center = curF0;
        const h1Width = Math.max(35.0, curF0 * 0.42);
        const h1Bell = Math.exp(-0.5 * Math.pow((freq - h1Center) / h1Width, 2));
        // Smooth octave tilt above 1.35 * F0 up to 4 kHz
        const octAbove = Math.max(0.0, Math.min(3.5, Math.log2(Math.max(1.0, freq / (curF0 * 1.35)))));
        const tiltTaper = freq < 4200.0 ? 1.0 : Math.max(0.0, 1.0 - (freq - 4200.0) / 2000.0);
        const dbShift = weightDb * (0.55 * h1Bell - 0.38 * octAbove * tiltTaper);
        gain *= Math.pow(10.0, Math.max(-12.0, Math.min(12.0, dbShift)) / 20.0);
      }

      re[k] *= gain;
      im[k] *= gain;

      // Subtle high-band breathiness shimmer only when pushing substantially lighter (> +2.5 dB)
      if (doWeight && isVoiced && weightDb > 2.5 && freq > 1800.0 && freq < 5200.0) {
        const breathAmt = Math.min(0.035, (weightDb - 2.5) * 0.005) * Math.sqrt(origEnergy / fftSize);
        re[k] += nextRand() * breathAmt;
        im[k] += nextRand() * breathAmt;
      }

      // Maintain conjugate symmetry for real IFFT
      re[fftSize - k] = re[k];
      im[fftSize - k] = -im[k];
      modEnergy += (re[k] * re[k] + im[k] * im[k]) * 2.0;
    }

    fftInPlace(re, im, tbl, true);

    // Gentle per-frame RMS matching so spectral tilt / resonance shifts keep natural loudness
    const synthEnergy = modEnergy / fftSize;
    const rmsRatio = synthEnergy > 1e-10 ? Math.sqrt(origEnergy / synthEnergy) : 1.0;
    const clampedRms = Math.max(0.75, Math.min(1.45, rmsRatio));

    for (let i = 0; i < fftSize; i++) {
      const idx = start + i;
      if (idx < n) {
        const w = hann[i];
        out[idx] += re[i] * clampedRms * w;
        winSum[idx] += w * w;
      }
    }
  }

  // Normalize COLA window sum and soft-limit peaks to [-0.96, +0.96]
  let peak = 0.0;
  for (let i = 0; i < n; i++) {
    if (winSum[i] > 1e-5) {
      out[i] /= winSum[i];
    }
    const a = Math.abs(out[i]);
    if (a > peak) peak = a;
  }
  if (peak > 0.95) {
    const sc = 0.95 / peak;
    for (let i = 0; i < n; i++) out[i] *= sc;
  }
  return out;
}

/**
 * Resynthesizes a recording with independent Pitch, Size (Resonance), and Vocal Weight edits.
 * @param {Float32Array} samples Mono PCM audio
 * @param {number} sampleRate Sample rate in Hz
 * @param {Float64Array} f0Times Pitch frame timestamps (sec)
 * @param {Float64Array} f0Hz Pitch frame frequencies (Hz, 0 = unvoiced)
 * @param {Float64Array|null} refF0Hz Optional DTW-aligned target F0 (same length as f0Times)
 * @param {{ pitchSemi: number, followContour: boolean, contourBlend: number, sizePct: number, weightDb: number }} opts
 * @returns {Float32Array}
 */
export function resynthesizeVoice(samples, sampleRate, f0Times, f0Hz, refF0Hz, opts) {
  if (!samples || samples.length < 128) return new Float32Array(samples || 0);
  const pitchSemi = Number(opts?.pitchSemi || 0);
  const followContour = !!opts?.followContour;
  const contourBlend = Math.max(0.0, Math.min(1.0, Number(opts?.contourBlend ?? 1.0)));
  const sizePct = Number(opts?.sizePct || 0);
  const weightDb = Number(opts?.weightDb || 0);

  // Stage 1: Pitch shift via TD-PSOLA (formant-preserving, crisp unvoiced consonants)
  const pitched = tdPsolaShift(
    samples,
    sampleRate,
    f0Times || new Float64Array(0),
    f0Hz || new Float64Array(0),
    refF0Hz || null,
    pitchSemi,
    followContour,
    contourBlend,
  );

  // Stage 2: Size (cepstral spectral envelope warp) + Weight (voiced spectral tilt)
  return spectralSizeWeightProcess(
    pitched,
    sampleRate,
    f0Times || new Float64Array(0),
    f0Hz || new Float64Array(0),
    pitchSemi,
    sizePct,
    weightDb,
  );
}
