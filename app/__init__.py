from flask import Flask
from app.routes.image_routes import image_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(image_bp, url_prefix="/image")
    return app
