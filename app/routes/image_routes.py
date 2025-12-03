from flask import Blueprint, request, jsonify, send_file
from app.utils.image_utils import load_image_from_bytes, save_image_to_buffer, adjust_image
from app.utils.filters_utils import generate_filter_name
import numpy as np
import io
import time
import os
import cv2
from PIL import Image, ImageOps
import base64
from app.utils.blur_utils import (
    linear_blur, radial_blur, oval_blur, focus_blur,
    hand_blur, apply_gaussian_blur
)

# Blueprint
image_bp = Blueprint("image", __name__)

# Folder for saving images
SAVE_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "saved_images"
)
os.makedirs(SAVE_FOLDER, exist_ok=True)


# Health Check
@image_bp.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Image Editor API running successfully!"})


# EDIT IMAGE (filters, adjustments)
@image_bp.route("/edit", methods=["POST"])
def edit_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        # Load image
        file_bytes = request.files["image"].read()
        img_arr = load_image_from_bytes(file_bytes)
        if img_arr is None:
            return jsonify({"error": "Failed to load image"}), 400

        # Expected adjustment params
        expected_keys = [
            "brightness", "contrast", "saturation", "exposure", "highlights",
            "shadows", "vibrance", "temperature", "hue", "fading", "enhance",
            "smoothness", "ambiance", "noise", "color_noise",
            "inner_spotlight", "outer_spotlight", "tint", "texture", "clarity",
            "dehaze", "grain_amount", "grain_size", "grain_roughness",
            "sharpen_amount", "sharpen_radius", "sharpen_detail",
            "sharpen_masking", "vignette_amount", "vignette_midpoint",
            "vignette_feather", "vignette_roundness", "vignette_highlights",
            "grayscale", "invert"
        ]

        adjustments = {k: request.form.get(k, 0) for k in expected_keys}

        processed_img = adjust_image(img_arr, adjustments)
        if processed_img is None:
            return jsonify({"error": "Image processing failed"}), 500

        # Save output into buffer
        buf = save_image_to_buffer(processed_img)
        if buf is None:
            return jsonify({"error": "Failed to write output image"}), 500

        filter_name = generate_filter_name(adjustments)

        buf.seek(0)
        return send_file(
            buf,
            mimetype='image/png',
            as_attachment=False,
            download_name=f"processed_{filter_name}.png",
            conditional=False
        )

    except Exception as e:
        return jsonify({
            "error": "Image edit failed",
            "details": str(e)
        }), 500


# BLUR ENDPOINT
@image_bp.route("/blur", methods=["POST"])
def blur_endpoint():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        h, w = img.shape[:2]

        # Main params
        blur_type = request.form.get("blur_type", "linear").lower()
        strength = np.clip(float(request.form.get("blur_strength", 50)), 1, 100)
        feather = np.clip(float(request.form.get("feather", 50)), 1, 100)

        # Type-specific
        strip_position = float(request.form.get("strip_position", 0.5))
        strip_width = float(request.form.get("strip_width", 0.3))
        radius_ratio = float(request.form.get("radius_ratio", 0.3))
        width_ratio = float(request.form.get("width_ratio", 0.45))
        height_ratio = float(request.form.get("height_ratio", 0.25))
        focus_region = request.form.get("focus_region", "center")

        # Hand blur
        hand_x = int(request.form.get("hand_x", w // 2))
        hand_y = int(request.form.get("hand_y", h // 2))
        hand_radius = float(request.form.get("hand_radius", 50))
        hand_feather = float(request.form.get("hand_feather", feather))

        # Processing
        if blur_type == "linear":
            out = linear_blur(img, strength, strip_position, strip_width, feather)
        elif blur_type in ["radial", "circular"]:
            out = radial_blur(img, strength, radius_ratio, feather)
        elif blur_type == "oval":
            out = oval_blur(img, strength, width_ratio, height_ratio, feather)
        elif blur_type == "focus":
            out = focus_blur(img, strength, focus_region, feather)
        elif blur_type == "hand":
            out = hand_blur(img, strength, hand_x, hand_y, hand_radius, hand_feather)
        else:
            out = apply_gaussian_blur(img, strength)

        # Encode result
        _, buffer = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype="image/jpeg"
        )

    except Exception as e:
        return jsonify({"error": f"Blur processing failed: {str(e)}"}), 500


# ENHANCE IMAGE (Auto contrast)
@image_bp.route("/enhance", methods=["POST"])
def enhance_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file_bytes = request.files["image"].read()
        arr = load_image_from_bytes(file_bytes)
        if arr is None:
            return jsonify({"error": "Failed to load image"}), 400

        img = Image.fromarray(arr.astype(np.uint8))
        img = ImageOps.autocontrast(img)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        buf.seek(0)

        return send_file(buf, mimetype="image/jpeg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# SAVE IMAGE LOCALLY
@image_bp.route("/save", methods=["POST"])
def save_image_endpoint():
    try:
        if "image" not in request.files:
            return jsonify({"isSuccess": False, "message": "No image uploaded"}), 400

        file_bytes = request.files["image"].read()
        arr = load_image_from_bytes(file_bytes)
        if arr is None:
            return jsonify({"isSuccess": False, "message": "Failed to load image"}), 400

        filename = f"saved_{int(time.time())}.jpg"
        path = os.path.join(SAVE_FOLDER, filename)

        Image.fromarray(arr.astype(np.uint8)).save(path, "JPEG", quality=95)

        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        return jsonify({
            "isSuccess": True,
            "message": "Saved successfully",
            "filePath": path,
            "image_base64": img_b64
        })

    except Exception as e:
        return jsonify({"isSuccess": False, "message": f"Save failed: {str(e)}"}), 500