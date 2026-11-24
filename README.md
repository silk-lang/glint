# Glint

Vocal resonance, pitch, vocal-weight, and formant analysis studio built in **Silk**, **Loom**, and **Tailwind CSS 4** — targeting both native **macOS** (zero-copy Metal IOSurface WebGPU + CoreAudio + Accelerate vDSP) and **WebAssembly** (`wasm32-unknown-unknown` + `loom-dom` + WebGL2 + Web Audio + IndexedDB).

## Features

- **Muted Light-Chrome Studio + Darkroom Spectrogram Well**: Calm `#FFFFFF` workspace framing a `#0E0E10` spectrogram well with Glint's default **Muted** colormap (`#1E2A55 → #5C9FB0 → #D8CF8A → #9B3A4A`) plus 7 classic palettes (`VoceVista`, `Inferno`, `Magma`, `Viridis`, `Ocean`, `Grayscale`, `Praat light`).
- **Acoustic Analysis**:
  - Boersma (1993) autocorrelation pitch tracking with Viterbi path refinement (`F0`, musical note, cents tuner, harmonics).
  - Praat Burg 5-formant LPC tracker (`F1–F4`), Hillenbrand (1995) gendered vowel space (`Fem` / `Masc` / `Both`), clickable IPA vowel targets that drop F1/F2/F3 guides.
  - Real-time **Size** (Vocal Tract Length `VTL` in cm), **Weight** (`H1–H2` dB), **HNR**, **Size × Weight** fullness pad, and **Slice + LPC / LTAS** spectra.
- **Reference Comparison & Hands-Free Practice**:
  - Pin any take as **Reference**, DTW time-warp alignment, **Ref waveform ribbon** along the top of the spectrogram, **Plot mine on ref** mode, composite **Match vs Ref** score with weakest-metric vermilion coaching hints.
  - **Repeat Drill** and **Echo Mirror** hands-free voice-activated practice loops.
- **Persistent Library**:
  - **macOS native**: Real folders & WAV files in `~/Documents/Glint Library/` (automatically reads existing `~/Documents/Overtone Library/` if present).
  - **Browser WASM**: Full browser persistence in **IndexedDB** (`glint-library-v1`) — folders, recorded/saved PCM takes, ratings, stars, match scores, comments, and script notes survive page reloads.

## Building

```bash
# 1. Compile Tailwind CSS 4 (generates styles/app.css + src/app_css.silk)
./scripts/build-css.sh

# 2. Build native macOS release binary -> dist/macos/glint
silk build --release

# 3. Build WASM web bundle -> dist/web/
silk build --manifest-path web-pkg/silk.toml --target wasm32-unknown-unknown --optimization size --debug false
```
