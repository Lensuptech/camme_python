from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import requests
from services.image_service import (
    enhance_image, 
    save_image, 
    edit_image_file, 
    blur_image_endpoint
)

image_bp = Blueprint("image_routes", __name__)


NODE_API_URL = "https://new-camme-backend.onrender.com/api/v1/image" 


@image_bp.route("/api/v1/filters", methods=["GET"])
def get_filters():
    try:
        resp = requests.get(NODE_API_URL, timeout=60)
        filters = resp.json()
        return jsonify(filters)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@image_bp.route("/", methods=["GET"])
def index():
    return jsonify({"message":"Image Editor API running successfully!"})


@image_bp.route("/edit", methods=["POST"])
def edit_image():
    """Handles image upload, applies adjustments, and returns the processed image."""
    try:
        image_file = request.files.get("image")
        if not image_file:
            return jsonify({"error": "No image uploaded"}), 400
        return edit_image_file(image_file)

    except Exception as e:
        return jsonify({
            "error": "Image processing failed",
            "details": str(e)
        }), 500



@image_bp.route("/blur", methods=["POST"])
def blur_endpoint():
    """Enhanced blur endpoint with simplified logic and robust handling."""
    if 'image' not in request.files:
        return jsonify(error="No image uploaded"), 400
    try:
        # Decode image
        img_file = request.files['image']
        img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify(error="Invalid image file"), 400
        return blur_image_endpoint(img)

    except Exception as e:
        return jsonify(error=f"Blur processing failed: {e}"), 500


@image_bp.route("/enhance", methods=["POST"])
def enhance_route():
    """Auto-enhance image contrast"""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        return enhance_image(request.files["image"].read())

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@image_bp.route("/save", methods=["POST"])
def save_image_endpoint():
    """Saves an uploaded image, returns file path and Base64 data."""
    try:
        image_file = request.files.get("image")
        if not image_file:
            return jsonify({"isSuccess": False, "message": "No image uploaded"}), 400
        return save_image(image_file)

    except Exception as e:
        return jsonify({"isSuccess": False, "message": f"Save failed: {e}"}), 500