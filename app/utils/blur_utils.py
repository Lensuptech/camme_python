import numpy as np
from PIL import Image, ImageFilter, ImageDraw


def apply_gaussian_blur(img, strength):
    """Replicate your custom Gaussian blur using PIL."""
    strength = np.clip(strength, 0, 100)

    sigma = (strength / 100.0) * 25 + 0.1
    sigma = float(np.clip(sigma, 0.1, 25))

    # PIL supports Gaussian blur via radius → equals sigma
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))



def feather_mask(mask_np, feather):
    """Feathers mask using PIL Gaussian blur."""
    feather_px = max(3, int(feather * 2) + 1)
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_blur = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_px))
    return np.array(mask_blur).astype(np.float32) / 255.0

# Linear Blur
def linear_blur(img, strength, strip_position=0.5, strip_width=0.3, feather=50):
    """Linear directional blur using PIL (no cv2)."""

    img_np = np.array(img) if not isinstance(img, np.ndarray) else img
    h, w = img_np.shape[:2]

    blurred = apply_gaussian_blur(img_np, strength)

    # Create mask
    mask = np.zeros((h, w), dtype=np.float32)

    longest = max(w, h)
    strip_px = int(strip_width * longest)
    center = int(strip_position * longest)

    start = max(0, center - strip_px // 2)
    end   = min(longest, center + strip_px // 2)

    # Orientation
    if w >= h:
        mask[:, start:end] = 1.0
    else:
        mask[start:end, :] = 1.0

    # Feather mask
    mask = feather_mask(mask, feather)
    mask = mask[..., None]  # make it 3D

    result = img_np * mask + blurred * (1 - mask)
    return result.astype(np.uint8)



# Radial Blur
def radial_blur(img, strength, radius_ratio=0.3, feather=50):
    """Radial blur using PIL (no cv2)."""

    img_np = np.array(img) if not isinstance(img, np.ndarray) else img
    h, w = img_np.shape[:2]

    blurred = apply_gaussian_blur(img_np, strength)

    # Create circular mask
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    cx, cy = w // 2, h // 2
    radius = int(radius_ratio * min(w, h))

    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)

    # Feather
    mask_np = np.array(mask).astype(np.float32) / 255.0
    mask_np = feather_mask(mask_np, feather)
    mask_np = mask_np[..., None]  # shape → (h, w, 1)

    result = img_np * mask_np + blurred * (1 - mask_np)
    return result.astype(np.uint8)


def oval_blur(img, strength, width_ratio=0.45, height_ratio=0.25, feather=50):
    """Elliptical center-focus blur using PIL."""

    # Convert to numpy
    img_np = np.array(img) if not isinstance(img, np.ndarray) else img
    h, w = img_np.shape[:2]

    # Apply background blur
    blurred = apply_gaussian_blur(img_np, strength)

    # --- Create elliptical mask ---
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    cx, cy = w // 2, h // 2
    rx = int(width_ratio * w)
    ry = int(height_ratio * h)

    draw.ellipse(
        (cx - rx, cy - ry, cx + rx, cy + ry),
        fill=255
    )

    # Convert mask to numpy (+ feathering)
    mask_np = np.array(mask).astype(np.float32) / 255.0
    mask_np = feather_mask(mask_np, feather)
    mask_np = mask_np[..., None]  # convert to (h,w,1)

    # Blend sharp + blurred
    result = img_np * mask_np + blurred * (1 - mask_np)
    return result.astype(np.uint8)


# Focus Blur (custom region)
def focus_blur(img, strength, focus_region="center", feather=50):
    """Selective blur except for a defined region using PIL."""

    # Convert to numpy if needed
    img_np = np.array(img) if not isinstance(img, np.ndarray) else img
    h, w = img_np.shape[:2]

    # Background blur
    blurred = apply_gaussian_blur(img_np, strength)

    # --- REGION MASK ---
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    regions = {
        "center": (w // 4, h // 4, 3 * w // 4, 3 * h // 4),
        "left":   (0, 0, w // 2, h),
        "right":  (w // 2, 0, w, h),
        "top":    (0, 0, w, h // 2),
        "bottom": (0, h // 2, w, h),
    }

    x1, y1, x2, y2 = regions.get(focus_region, (0, 0, w, h))

    # Fill focus region
    draw.rectangle((x1, y1, x2, y2), fill=255)

    # Convert mask to numpy + feather
    mask_np = np.array(mask).astype(np.float32) / 255.0
    mask_np = feather_mask(mask_np, feather)
    mask_np = mask_np[..., None]

    # Blend focused + blurred
    result = img_np * mask_np + blurred * (1 - mask_np)
    return result.astype(np.uint8)


# Hand-Drawn Blur
def hand_blur(img, strength, hand_x, hand_y, hand_radius, hand_feather):
    """
    User-drawn circular 'spot focus' blur using PIL.
    Inside circle = sharp
    Outside = blurred
    """

    # Convert to numpy if image is PIL
    img_np = np.array(img) if not isinstance(img, np.ndarray) else img
    h, w = img_np.shape[:2]

    # Background blur
    blurred = apply_gaussian_blur(img_np, strength)

    # --- Create mask (white = sharp, black = blurred) ---
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    # Clamp safe values
    hand_x = int(np.clip(hand_x, 0, w - 1))
    hand_y = int(np.clip(hand_y, 0, h - 1))
    hand_radius = int(np.clip(hand_radius, 10, min(w, h) // 2))

    # Draw filled circle
    draw.ellipse(
        (
            hand_x - hand_radius,
            hand_y - hand_radius,
            hand_x + hand_radius,
            hand_y + hand_radius
        ),
        fill=255
    )

    # Convert mask → numpy and feather it
    mask_np = np.array(mask).astype(np.float32) / 255.0
    mask_np = feather_mask(mask_np, hand_feather)
    mask_np = mask_np[..., None]  # Add channel dimension

    # Blend sharp + blurred regions
    result = img_np * mask_np + blurred * (1 - mask_np)
    return result.astype(np.uint8)