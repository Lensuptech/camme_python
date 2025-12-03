import cv2
import numpy as np


# -----------------------------------------------------------
# Gaussian Blur – Common helper
# -----------------------------------------------------------

def apply_gaussian_blur(img, strength):
    """Apply gaussian blur based on strength 0–100."""
    strength = np.clip(strength, 0, 100)

    # Convert strength → sigma range 0.1 to 25
    sigma = (strength / 100.0) * 25 + 0.1
    sigma = np.clip(sigma, 0.1, 25)

    # Kernel size must be odd
    kernel = int(sigma * 3) * 2 + 1
    kernel = max(3, kernel + (kernel % 2 == 0))

    return cv2.GaussianBlur(img, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)


# -----------------------------------------------------------
# Linear Blur
# -----------------------------------------------------------

def linear_blur(img, strength, strip_position=0.5, strip_width=0.3, feather=50):
    """Linear directional blur with soft feathering."""

    h, w = img.shape[:2]
    blurred = apply_gaussian_blur(img, strength)

    mask = np.zeros((h, w), dtype=np.float32)

    longest = max(w, h)
    strip_px = int(strip_width * longest)
    center = int(strip_position * longest)

    start = max(0, center - strip_px // 2)
    end = min(longest, center + strip_px // 2)

    # Decide orientation based on shape
    if w >= h:
        mask[:, start:end] = 1.0
    else:
        mask[start:end, :] = 1.0

    # Feathering
    fk = max(3, int(feather * 2) + 1)
    fk += fk % 2 == 0
    mask = cv2.GaussianBlur(mask, (fk, fk), 0)[..., None]

    return (img * mask + blurred * (1 - mask)).astype(np.uint8)


# Radial Blur
def radial_blur(img, strength, radius_ratio=0.3, feather=50):
    """Center-sharp radial blur (lens blur)."""

    h, w = img.shape[:2]
    blurred = apply_gaussian_blur(img, strength)

    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    radius = int(radius_ratio * min(w, h))

    cv2.circle(mask, center, radius, 1.0, -1)

    fk = max(3, int(feather * 2) + 1)
    fk += fk % 2 == 0
    mask = cv2.GaussianBlur(mask, (fk, fk), 0)[..., None]

    return (img * mask + blurred * (1 - mask)).astype(np.uint8)


# Oval / Elliptical Blur
def oval_blur(img, strength, width_ratio=0.45, height_ratio=0.25, feather=50):
    """Elliptical center-focus blur."""

    h, w = img.shape[:2]
    blurred = apply_gaussian_blur(img, strength)

    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    axes = (int(width_ratio * w), int(height_ratio * h))

    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

    fk = max(3, int(feather * 2) + 1)
    fk += fk % 2 == 0
    mask = cv2.GaussianBlur(mask, (fk, fk), 0)[..., None]

    return (img * mask + blurred * (1 - mask)).astype(np.uint8)


# Focus Blur (custom region)
def focus_blur(img, strength, focus_region="center", feather=50):
    """Selective blur except for a defined region."""

    h, w = img.shape[:2]
    blurred = apply_gaussian_blur(img, strength)

    mask = np.zeros((h, w), dtype=np.float32)

    regions = {
        "center": (w // 4, h // 4, 3 * w // 4, 3 * h // 4),
        "left": (0, 0, w // 2, h),
        "right": (w // 2, 0, w, h),
        "top": (0, 0, w, h // 2),
        "bottom": (0, h // 2, w, h),
    }

    x1, y1, x2, y2 = regions.get(focus_region, (0, 0, w, h))
    mask[y1:y2, x1:x2] = 1.0

    fk = max(3, int(feather * 2) + 1)
    fk += fk % 2 == 0
    mask = cv2.GaussianBlur(mask, (fk, fk), 0)[..., None]

    return (img * mask + blurred * (1 - mask)).astype(np.uint8)


# Hand-Drawn Blur
def hand_blur(img, strength, hand_x, hand_y, hand_radius, hand_feather):
    """
    User-drawn circular "spot focus" blur.
    Inside circle = sharp  
    Outside = blurred
    """

    h, w = img.shape[:2]
    blurred = apply_gaussian_blur(img, strength)

    # Create mask
    mask = np.zeros((h, w), dtype=np.float32)

    # Clamp values safely
    hand_x = int(np.clip(hand_x, 0, w - 1))
    hand_y = int(np.clip(hand_y, 0, h - 1))
    hand_radius = int(np.clip(hand_radius, 10, min(w, h) // 2))

    cv2.circle(mask, (hand_x, hand_y), hand_radius, 1.0, -1)

    fk = max(3, int(hand_feather * 2) + 1)
    fk += fk % 2 == 0
    mask = cv2.GaussianBlur(mask, (fk, fk), 0)[..., None]

    return (img * mask + blurred * (1 - mask)).astype(np.uint8)