#!/usr/bin/env bash
# Build WhisperApp.dmg for macOS distribution
# Prerequisites:
#   pip install pyinstaller
#   brew install create-dmg
# Usage: bash installer/build_mac.sh

set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=$(python -c "import whisperapp; print(getattr(whisperapp, '__version__', '1.1.0'))")
DMG_NAME="WhisperApp-${VERSION}-macOS"

echo "==> Building PyInstaller bundle..."
pyinstaller whisperapp.spec --noconfirm

echo "==> Creating DMG: ${DMG_NAME}.dmg"
create-dmg \
  --volname "WhisperApp ${VERSION}" \
  --volicon "assets/icon.icns" \
  --window-pos 200 120 \
  --window-size 600 400 \
  --icon-size 100 \
  --icon "WhisperApp.app" 175 190 \
  --hide-extension "WhisperApp.app" \
  --app-drop-link 425 190 \
  --no-internet-enable \
  "dist/${DMG_NAME}.dmg" \
  "dist/WhisperApp.app"

echo "==> Done: dist/${DMG_NAME}.dmg"
