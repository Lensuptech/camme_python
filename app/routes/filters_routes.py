from flask import Blueprint, jsonify
import requests

filters_bp = Blueprint("filters", __name__)

NODE_API_URL = "https://new-camme-backend.onrender.com/api/v1/image"

@filters_bp.route("/filters", methods=["GET"])
def get_filters():
    """
    Get Filters
    ---
    tags:
      - Filters
    responses:
      200:
        description: Filters returned successfully
      500:
        description: Error while fetching filters
    """
    try:
        resp = requests.get(NODE_API_URL, timeout=60)
        resp.raise_for_status()
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500