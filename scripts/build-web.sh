#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/dist/web"

cd "$ROOT_DIR"

"$ROOT_DIR/scripts/build-css.sh"

silk build \
  --manifest-path "$ROOT_DIR/web-pkg/silk.toml" \
  --target wasm32-unknown-unknown \
  --optimization size \
  --debug false

DOM_JS="$(silk package-path --manifest "$ROOT_DIR/web-pkg/silk.toml" --dependency loom_dom --path js/silk-ui-dom.js | python3 -c 'import json,sys; print(json.load(sys.stdin))')"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
cp "$ROOT_DIR/web/index.html" "$OUT_DIR/index.html"
cp "$ROOT_DIR/web/glint-host.js" "$OUT_DIR/glint-host.js"
cp "$ROOT_DIR/web/neural-worker.js" "$OUT_DIR/neural-worker.js"
cp "$DOM_JS" "$OUT_DIR/silk-ui-dom.js"
cp "$ROOT_DIR/styles/app.css" "$OUT_DIR/app.css"
cp "$ROOT_DIR/web-pkg/build/wasm32-unknown-unknown/size/glint-web.wasm" "$OUT_DIR/glint.wasm"
if [ -d "$ROOT_DIR/web/demos" ]; then
  cp -R "$ROOT_DIR/web/demos" "$OUT_DIR/demos"
fi
if [ -d "$ROOT_DIR/web/models" ]; then
  cp -R "$ROOT_DIR/web/models" "$OUT_DIR/models"
fi

echo "Built $OUT_DIR:"
ls -lh "$OUT_DIR"
