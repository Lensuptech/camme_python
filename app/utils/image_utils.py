import cv2
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter


# ---------------------------------------------------
# Parameter Validation
# ---------------------------------------------------
def validate_param(value, min_val=-100, max_val=100, default=0):
    """Safely convert a value to float within a range."""
    try:
        val = float(value)
        return max(min(val, max_val), min_val)
    except Exception:
        return default


# ---------------------------------------------------
# I/O Helpers
# ---------------------------------------------------
def load_image_from_bytes(file_bytes):
    """Load incoming image bytes into a float32 NumPy array."""
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return np.array(img, dtype=np.float32)
    except Exception as e:
        print("Image load error:", e)
        return None


def save_image_to_buffer(arr):
    """Convert np array → JPEG bytes for response."""
    try:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        return buf
    except Exception as e:
        print("Save buffer error:", e)
        return None


# ---------------------------------------------------
# Main Adjustment Engine
# ---------------------------------------------------
def adjust_image(arr, params):
    if arr is None:
        return None

    # Ensure RGB
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=2)
    if arr.shape[2] < 3:
        arr = np.repeat(arr[:, :, :1], 3, axis=2)

    out = arr.astype(np.float32).copy()

    try:
        # ---------------------------------------------------
        # Parameter Parsing
        # ---------------------------------------------------
        p = lambda k, d=0, mn=-100, mx=100: validate_param(params.get(k, d), mn, mx, d)

        brightness = p("brightness") * 2.55
        contrast = p("contrast") / 100.0
        saturation = p("saturation") / 100.0
        exposure = p("exposure") * 1.5
        highlights = p("highlights") / 100.0
        shadows = p("shadows") / 100.0
        vibrance = p("vibrance") / 100.0
        temperature = p("temperature") * 0.5
        fading = p("fading") * 2.55
        enhance = p("enhance") * 2.0
        smoothness = p("smoothness") / 100.0
        ambiance = p("ambiance") * 2.0
        noise = max(0.0, p("noise", 0, 0, 100) * 0.3)
        color_noise = max(0.0, p("color_noise", 0, 0, 100) * 0.3)
        texture = p("texture") / 100.0
        clarity = p("clarity") / 100.0
        dehaze = p("dehaze") / 100.0
        hue = p("hue", 0, -100, 100)

        vignette_amount = p("vignette_amount") / 100.0
        vignette_midpoint = p("vignette_midpoint", 50) / 100.0
        vignette_roundness = p("vignette_roundness") / 100.0
        vignette_highlights = p("vignette_highlights") / 100.0

        vignette_feather = int(p("vignette_feather", 50))
        if vignette_feather % 2 == 0:
            vignette_feather += 1

        grain_amount = p("grain_amount") / 100.0
        grain_size = int(p("grain_size", 1, 1, 10))
        grain_roughness = p("grain_roughness", 1, 1, 10)

        sharpen_amount = p("sharpen_amount")
        sharpen_radius = int(p("sharpen_radius", 1))
        sharpen_detail = p("sharpen_detail", 1)
        sharpen_masking = p("sharpen_masking", 0)

        # ---------------------------------------------------
        # Basic Adjustments
        # ---------------------------------------------------
        out += brightness + exposure + fading
        out = np.clip(out, 0, 255)

        # Temperature Tint
        if temperature != 0:
            out[:, :, 0] = np.clip(out[:, :, 0] + temperature, 0, 255)  # Red
            out[:, :, 2] = np.clip(out[:, :, 2] - temperature, 0, 255)  # Blue

        # Tint (Green)
        tint = p("tint") * 0.5
        if tint != 0:
            out[:, :, 1] = np.clip(out[:, :, 1] + tint, 0, 255)

        # Highlights/Shadows
        gray = np.mean(out, axis=2, keepdims=True)
        bright_mask = (gray > 128).astype(np.float32)
        dark_mask = (gray <= 128).astype(np.float32)

        out += bright_mask * (highlights * 50)
        out += dark_mask * (shadows * 50)
        out = np.clip(out, 0, 255)

        # ---------------------------------------------------
        # Convert to PIL for Enhancers
        # ---------------------------------------------------
        img = Image.fromarray(out.astype(np.uint8))

        if contrast != 0:
            img = ImageEnhance.Contrast(img).enhance(1 + contrast)
        if saturation != 0:
            img = ImageEnhance.Color(img).enhance(1 + saturation)
        if enhance != 0:
            img = ImageEnhance.Sharpness(img).enhance(1 + enhance / 50.0)
        if ambiance != 0:
            img = ImageEnhance.Brightness(img).enhance(1 + ambiance / 50.0)

        # Texture
        if texture != 0:
            img = img.filter(ImageFilter.DETAIL if texture > 0 else ImageFilter.SMOOTH)

        # Clarity
        if clarity != 0:
            img = ImageEnhance.Contrast(img).enhance(1 + clarity)

        # Dehaze
        if dehaze != 0:
            img = ImageOps.autocontrast(img, cutoff=int(abs(dehaze) * 10))

        # ---------------------------------------------------
        # Convert back to NumPy
        # ---------------------------------------------------
        np_img = np.array(img).astype(np.float32)

        # Vibrance
        if vibrance != 0:
            gray = np.mean(np_img, axis=2, keepdims=True)
            np_img += (np_img - gray) * vibrance

        # Smoothness
        if smoothness != 0:
            np_img = cv2.GaussianBlur(np_img, sigma=[smoothness * 5, smoothness * 5, 0])

        # Noise
        if noise > 0:
            np_img += np.random.normal(0, noise, np_img.shape)
        if color_noise > 0:
            np_img += np.random.normal(0, color_noise, np_img.shape)

        # Grain
        if grain_amount > 0:
            h, w = np_img.shape[:2]
            grain = np.random.normal(0, grain_amount * 30, (h, w))
            k = max(1, int(grain_size * 2) + 1)
            grain = cv2.GaussianBlur(grain, (k, k), 0) * grain_roughness
            np_img += grain[:, :, None]

        # ---------------------------------------------------
        # Vignette
        # ---------------------------------------------------
        if vignette_amount > 0:
            h, w = np_img.shape[:2]
            X, Y = np.meshgrid(np.arange(w), np.arange(h))
            cx, cy = w // 2, h // 2
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            mask = 1 - vignette_amount * (
                np.clip((dist / np.max(dist) - vignette_midpoint) / (1 - vignette_midpoint), 0, 1)
            )
            if vignette_feather > 1:
                mask = cv2.GaussianBlur(mask, (vignette_feather, vignette_feather), 0)

            np_img *= mask[:, :, None]

        # ---------------------------------------------------
        # Sharpen
        # ---------------------------------------------------
        if sharpen_amount != 0:
            pil_img = Image.fromarray(np.clip(np_img, 0, 255).astype(np.uint8))
            pil_img = ImageEnhance.Sharpness(pil_img).enhance(1 + sharpen_amount / 50.0)
            np_img = np.array(pil_img).astype(np.float32)

        # ---------------------------------------------------
        # Hue
        # ---------------------------------------------------
        if hue != 0:
            hsv = cv2.cvtColor(np_img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 180
            np_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        return np.clip(np_img, 0, 255)

    except Exception as e:
        print("Adjustment error:", e, "Params:", params)
        return arr