import io
import cv2
import numpy as np
from PIL import Image, ImageOps
import os
import time
import base64
from flask import request, jsonify, send_file
from utilities.image_utility import (linear_blur, 
                           radial_blur, 
                           oval_blur, 
                           focus_blur, 
                           apply_gaussian_blur, 
                           generate_filter_name, 
                           save_image_to_buffer, 
                           hand_blur, 
                           adjust_image, 
                           load_image_from_bytes)


def edit_image_file(image_file):
    img_arr = load_image_from_bytes(image_file.read())
    if img_arr is None:
        return jsonify({"error": "Failed to load image"}), 400

    # Extract adjustment parameters
    adjustment_keys = [
        "brightness", "contrast", "saturation", "exposure", "highlights", "shadows", "vibrance",
        "temperature", "hue", "fading", "enhance", "smoothness", "ambiance", "noise", "color_noise",
        "inner_spotlight", "outer_spotlight", "tint", "texture", "clarity", "dehaze", "grain_amount",
        "grain_size", "grain_roughness", "sharpen_amount", "sharpen_radius", "sharpen_detail",
        "sharpen_masking", "vignette_amount", "vignette_midpoint", "vignette_feather",
        "vignette_roundness", "vignette_highlights", "grayscale", "invert"
    ]
    adjustments = {k: float(request.form.get(k, 0)) for k in adjustment_keys}

    # Apply adjustments
    processed_img = adjust_image(img_arr, adjustments)
    if processed_img is None:
        return jsonify({"error": "Failed to adjust image"}), 500

    # Convert processed image to buffer
    buf = save_image_to_buffer(processed_img)
    if not buf:
        return jsonify({"error": "Failed to create image buffer"}), 500

    # Generate filter name
    filter_name = generate_filter_name(adjustments)

    # Send processed image
    buf.seek(0)
    return send_file(
        buf,
        mimetype="image/png",
        download_name=f"processed_{filter_name}.png",
    ), 200


def blur_image_endpoint(img):
    h, w = img.shape[:2]
    form = request.form.get

    # Parse main parameters
    blur_type = form('blur_type', 'linear').lower()
    strength = np.clip(float(form('blur_strength', 50)), 1, 100)
    feather = np.clip(float(form('feather', 50)), 1, 100)

    # Common optional parameters
    params = {
        "strip_position": float(form('strip_position', 0.5)),
        "strip_width": float(form('strip_width', 0.3)),
        "radius_ratio": float(form('radius_ratio', 0.3)),
        "width_ratio": float(form('width_ratio', 0.45)),
        "height_ratio": float(form('height_ratio', 0.25)),
        "focus_region": form('focus_region', 'center'),
        "hand_x": int(form('hand_x', w // 2)),
        "hand_y": int(form('hand_y', h // 2)),
        "hand_radius": float(form('hand_radius', 50)),
        "hand_feather": float(form('hand_feather', feather))
    }

    # Apply selected blur
    blur_map = {
        "linear": lambda: linear_blur(img, strength, params["strip_position"], params["strip_width"], feather),
        "radial": lambda: radial_blur(img, strength, params["radius_ratio"], feather),
        "circular": lambda: radial_blur(img, strength, params["radius_ratio"], feather),
        "oval": lambda: oval_blur(img, strength, params["width_ratio"], params["height_ratio"], feather),
        "focus": lambda: focus_blur(img, strength, params["focus_region"], feather),
        "hand": lambda: hand_blur(img, strength, params["hand_x"], params["hand_y"], params["hand_radius"], params["hand_feather"])
    }

    out = blur_map.get(blur_type, lambda: apply_gaussian_blur(img, strength))()

    # Return as JPEG
    _, buffer = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/jpeg')


def enhance_image(file_bytes):
    """Auto-enhance image contrast"""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = ImageOps.autocontrast(img)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg", download_name="enhanced.jpg")


def save_image(image_file):
    img_arr = load_image_from_bytes(image_file.read())
    if img_arr is None:
        return jsonify({"isSuccess": False, "message": "Failed to load image"}), 400
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    filename = f"saved_{int(time.time())}.jpg"
    path = os.path.join(SAVE_FOLDER, filename)
    # Save image
    Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8)).save(path, "JPEG", quality=95)

    # Convert to base64
    with open(path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    return jsonify({
        "isSuccess": True,
        "message": "Saved successfully",
        "filePath": path,
        "image_base64": img_b64
    })