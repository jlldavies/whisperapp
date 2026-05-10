# App Icons

Place your icon files here before building installers:

- `icon.icns` — macOS app bundle icon (1024×1024 recommended, ICNS format)
- `icon.ico` — Windows executable and installer icon (256×256 recommended, ICO format)

To convert a PNG:
- macOS: `sips -s format icns icon.png --out icon.icns`
  or use Xcode's asset catalog / `iconutil`
- Windows: use ImageMagick: `magick icon.png -define icon:auto-resize=256,128,64,48,32,16 icon.ico`

Without these files the installer builds succeed but produce apps with no icon.
