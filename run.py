from flask import Flask
from flasgger import Swagger
from app.routes.image_routes import image_bp
from app.routes.filters_routes import filters_bp

def create_app():
    app = Flask(__name__)

    # Swagger config
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": 'apispec_1',
                "route": '/apispec_1.json',
                "rule_filter": lambda rule: True,  # include all routes
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/docs/"
    }

    Swagger(app, config=swagger_config)

    # Register routes
    app.register_blueprint(image_bp, url_prefix="/api")
    app.register_blueprint(filters_bp, url_prefix="/api")

    return app
