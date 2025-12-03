from PIL import ImageEnhance, Image, ImageOps, ImageFilter
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import io


def validate_param(value, min_val=-100, max_val=100, default=0):
    """Ensure numeric parameters are within expected range"""
    try:
        val = float(value)
        return max(min(val, max_val), min_val)
    except Exception:
        return default


def apply_pil_enhancements(img, contrast, saturation, vibrance, enhance, smoothness, ambiance, texture, clarity, dehaze):
    """Apply Pillow-based image enhancements."""
    if contrast: img = ImageEnhance.Contrast(img).enhance(1 + contrast / 100)
    if saturation: img = ImageEnhance.Color(img).enhance(1 + saturation / 100)
    if vibrance:
        np_img = np.array(img, dtype=np.float32)
        mean_val = np.mean(np_img, axis=2, keepdims=True)
        np_img += (np_img - mean_val) * (vibrance / 100)
        img = Image.fromarray(np.clip(np_img, 0, 255).astype(np.uint8))
    if enhance: img = ImageEnhance.Sharpness(img).enhance(1 + enhance / 50)
    if smoothness:
        np_img = np.array(img, dtype=np.float32)
        np_img = gaussian_filter(np_img, sigma=[smoothness * 0.5, smoothness * 0.5, 0])
        img = Image.fromarray(np.clip(np_img, 0, 255).astype(np.uint8))
    if ambiance: img = ImageEnhance.Brightness(img).enhance(1 + ambiance / 50)
    if texture: img = img.filter(ImageFilter.DETAIL if texture > 0 else ImageFilter.SMOOTH)
    if clarity: img = ImageEnhance.Contrast(img).enhance(1 + clarity / 100)
    if dehaze: img = ImageOps.autocontrast(img, cutoff=int(abs(dehaze * 10)))
    return img


def add_noise_and_grain(img_np, noise, color_noise, grain_amount, grain_size, grain_roughness):
    """Add random noise and film-like grain effects."""
    if noise: img_np += np.random.normal(0, noise * 0.3, img_np.shape)
    if color_noise: img_np += np.random.normal(0, color_noise * 0.3, img_np.shape)

    if grain_amount:
        h, w = img_np.shape[:2]
        grain = np.random.normal(0, grain_amount * 30, (h, w))
        k = max(1, int(grain_size * 2) | 1)  # ensure odd kernel
        grain = cv2.GaussianBlur(grain, (k, k), 0) * grain_roughness
        img_np += np.repeat(grain[:, :, None], 3, axis=2)

    return np.clip(img_np, 0, 255)


def apply_vignette(img_np, amount, midpoint, feather, roundness):
    """Apply vignette lighting effect."""
    if amount <= 0:
        return img_np

    h, w = img_np.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    dist = np.sqrt((X - w/2)**2 + (Y - h/2)**2)
    mask = 1 - amount * np.clip((dist / dist.max() - midpoint) / (1 - midpoint), 0, 1)
    if feather > 1:
        mask = cv2.GaussianBlur(mask, (feather | 1, feather | 1), 0)
    return img_np * mask[..., None]


def apply_gaussian_blur(img, strength):
    """Apply Gaussian blur with scaled strength (0–100 → sigma 0.1–25)."""
    sigma = np.clip((strength / 4) + 0.1, 0.1, 25)  # simpler scaling
    k = int(max(3, round(sigma * 6 + 1)))           # auto odd kernel size
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), sigma)


def generate_filter_name(settings: dict) -> str:
    """Generate a unique name for the applied filter settings"""
    active_params = {
        k: float(v) for k, v in settings.items()
        if v not in (None, "", "0", 0) and str(v).replace('.', '', 1).isdigit()
    }

    if not active_params:
        return "No_Adjustments"

    sorted_items = sorted(active_params.items())
    param_list = " | ".join([f"{k}={v:.1f}" for k, v in sorted_items])
    unique_name = "_".join([f"{k}{int(round(v))}" for k, v in sorted_items])

    if len(unique_name) > 100:
        unique_name = unique_name[:100] + f"_hash{abs(hash(unique_name)) % 10000}"

    return f"{param_list} || FILTER: {unique_name}"


def save_image_to_buffer(arr):
    """Convert numpy array image to a BytesIO JPEG"""
    try:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Save buffer error: {e}")
        return None


def adjust_image(arr, params):
    """Applies image adjustments (brightness, contrast, vignette, etc.)"""
    if arr is None:
        return None

    # Ensure RGB
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=2)
    elif arr.shape[2] < 3:
        arr = np.repeat(arr[:, :, :1], 3, axis=2)

    out = arr.astype(np.float32)

    try:
        # Basic parameters
        brightness = validate_param(params.get("brightness", 0)) * 2.55
        exposure = validate_param(params.get("exposure", 0)) * 1.5
        fading = validate_param(params.get("fading", 0)) * 2.55
        temperature = validate_param(params.get("temperature", 0)) * 0.5
        tint = validate_param(params.get("tint", 0)) * 0.5

        # Light adjustment
        out = np.clip(out + brightness + exposure + fading, 0, 255)
        out[..., 0] = np.clip(out[..., 0] + temperature, 0, 255)
        out[..., 2] = np.clip(out[..., 2] - temperature, 0, 255)
        out[..., 1] = np.clip(out[..., 1] + tint, 0, 255)

        # Shadows & highlights
        gray = np.mean(out, axis=2)
        highlights = validate_param(params.get("highlights", 0))
        shadows = validate_param(params.get("shadows", 0))
        out += (gray > 128)[..., None] * (highlights * 0.5) + (gray <= 128)[..., None] * (shadows * 0.5)
        out = np.clip(out, 0, 255)

        # Convert to PIL and apply enhancements
        img = Image.fromarray(out.astype(np.uint8))
        img = apply_pil_enhancements(
            img,
            contrast=params.get("contrast", 0),
            saturation=params.get("saturation", 0),
            vibrance=params.get("vibrance", 0),
            enhance=params.get("enhance", 0),
            smoothness=params.get("smoothness", 0),
            ambiance=params.get("ambiance", 0),
            texture=params.get("texture", 0),
            clarity=params.get("clarity", 0),
            dehaze=params.get("dehaze", 0),
        )

        img_np = np.array(img, dtype=np.float32)

        # Noise, grain, vignette
        img_np = add_noise_and_grain(
            img_np,
            params.get("noise", 0),
            params.get("color_noise", 0),
            params.get("grain_amount", 0),
            params.get("grain_size", 1),
            params.get("grain_roughness", 1),
        )

        img_np = apply_vignette(
            img_np,
            params.get("vignette_amount", 0) / 100.0,
            params.get("vignette_midpoint", 50) / 100.0,
            int(params.get("vignette_feather", 50)),
            params.get("vignette_roundness", 0),
        )

        # Optional sharpen
        sharpen = validate_param(params.get("sharpen_amount", 0))
        if sharpen:
            img_np = np.array(ImageEnhance.Sharpness(Image.fromarray(img_np.astype(np.uint8)))
                              .enhance(1 + sharpen / 50), dtype=np.float32)

        # Optional hue shift
        hue = validate_param(params.get("hue", 0))
        if hue:
            hsv = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 180
            img_np = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        return np.clip(img_np, 0, 255)

    except Exception as e:
        print(f"[Error in adjust_image] {e}")
        return arr


def hand_blur(img, strength, hand_x, hand_y, hand_radius, hand_feather):
    """Apply blur keeping a circular region around (x, y) sharp."""
    h, w = img.shape[:2]
    blurred = apply_gaussian_blur(img, strength)

    # Clamp values to valid range
    x, y = int(np.clip(hand_x, 0, w - 1)), int(np.clip(hand_y, 0, h - 1))
    r = int(np.clip(hand_radius, 10, min(w, h) // 2))

    # Create circular mask
    mask = np.zeros((h, w), np.float32)
    cv2.circle(mask, (x, y), r, 1.0, -1)

    # Feather and blend
    k = max(3, int(hand_feather * 2) + 1) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None]
    return cv2.convertScaleAbs(img * mask + blurred * (1 - mask))


def focus_blur(img, strength, focus_region='center', feather=50):
    """Apply focus blur keeping one region sharp (center, left, right, top, bottom)."""
    h, w = img.shape[:2]
    blurred = apply_gaussian_blur(img, strength)
    mask = np.zeros((h, w), np.float32)

    regions = {
        'center': (w // 4, h // 4, 3 * w // 4, 3 * h // 4),
        'left': (0, 0, w // 2, h),
        'right': (w // 2, 0, w, h),
        'top': (0, 0, w, h // 2),
        'bottom': (0, h // 2, w, h)
    }
    x1, y1, x2, y2 = regions.get(focus_region, (0, 0, w, h))
    mask[y1:y2, x1:x2] = 1

    # Feather and blend
    k = max(3, int(feather) * 2 + 1) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None]
    return cv2.convertScaleAbs(img * mask + blurred * (1 - mask))


def oval_blur(img, strength, width_ratio=0.45, height_ratio=0.25, feather=50):
    """Apply oval (elliptical) blur around image center."""
    h, w = img.shape[:2]
    blurred = apply_gaussian_blur(img, strength)

    # Create elliptical mask
    mask = np.zeros((h, w), np.float32)
    center = (w // 2, h // 2)
    axes = (int(width_ratio * w), int(height_ratio * h))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)

    # Feather edges (ensure odd kernel size)
    k = max(3, int(feather) * 2 + 1) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None]

    # Blend sharp center and blurred surroundings
    return cv2.convertScaleAbs(img * mask + blurred * (1 - mask))


def radial_blur(img, strength, radius_ratio=0.3, feather=50):
    """Apply radial blur from center outward."""
    h, w = img.shape[:2]
    blurred = apply_gaussian_blur(img, strength)

    # Create circular mask (1 = sharp center, 0 = blurred edges)
    mask = np.zeros((h, w), np.float32)
    cv2.circle(mask, (w // 2, h // 2), int(radius_ratio * min(w, h)), 1, -1)

    # Feather mask edges
    k = max(3, int(feather) * 2 + 1) | 1  # ensure odd kernel
    mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None]

    # Blend sharp and blurred regions
    return cv2.convertScaleAbs(img * mask + blurred * (1 - mask))


def linear_blur(img, strength, strip_position=0.5, strip_width=0.3, feather=50):
    """Apply linear blur with smooth transition."""
    h, w = img.shape[:2]
    blurred = apply_gaussian_blur(img, strength)

    # Create base mask (1 = sharp, 0 = blurred)
    mask = np.zeros((h, w), np.float32)
    axis = 1 if w >= h else 0  # horizontal or vertical orientation
    size = max(w, h)
    
    # Compute strip region
    center = int(strip_position * size)
    half_strip = int(strip_width * size / 2)
    start, end = max(0, center - half_strip), min(size, center + half_strip)
    
    # Fill the strip area
    if axis == 1:
        mask[:, start:end] = 1
    else:
        mask[start:end, :] = 1

    # Feather edges
    k = max(3, int(feather) * 2 + 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None]

    # Blend sharp and blurred regions
    return cv2.convertScaleAbs(img * mask + blurred * (1 - mask))


def load_image_from_bytes(file_bytes):
    """Load image file into numpy array"""
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return np.array(img).astype(np.float32)
    except Exception as e:
        print(f"Image load error: {e}")
        return None