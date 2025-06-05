# Assets Directory

This directory contains images and other assets for the Xin Vault Analyzer.

## Adding a Banner Image

To add a custom banner/header image to your vault analyzer:

### Option 1: Full-Width Banner
1. Save your banner image as `banner.png` (or `banner.jpg`, `banner.svg`, etc.)
2. The image will be displayed full-width at the top of the page
3. The title will appear smaller below the banner

### Option 2: Centered Logo
1. Save your logo/image as `logo.png` (or `logo.jpg`, `logo.svg`, etc.)
2. The image will be displayed centered with columns on the sides
3. The title will appear below the centered image

### Recommended Image Specifications

**For Banner (`banner.png`):**
- **Width**: 1200-1920px (will scale to container width)
- **Height**: 200-400px 
- **Format**: PNG, JPG, or SVG
- **Style**: Wide banner design, possibly with text/logo

**For Logo (`logo.png`):**
- **Width**: 300-600px
- **Height**: 150-300px  
- **Format**: PNG (with transparency), JPG, or SVG
- **Style**: Square or rectangular logo

### Supported Image Formats
- PNG (recommended for logos with transparency)
- JPG/JPEG (good for photos/banners)
- SVG (great for scalable graphics)
- GIF (basic support)
- WEBP (modern format)

### Example Usage
1. Place your image file in this `assets/` directory
2. Restart the Streamlit app
3. The image will automatically appear in the header

### File Priority
The app will look for images in this order:
1. `banner.png` (full-width banner)
2. `logo.png` (centered logo)
3. If neither exists, shows default text title

### Tips
- Keep file sizes reasonable (< 2MB) for faster loading
- Use high-quality images that look good when scaled
- Consider your brand colors and the app's overall design
- Test different image sizes to see what works best 