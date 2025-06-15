from flask import Flask
from flask_cors import CORS
from .interview.dash import create_dash_app

def create_app(config_object=None):
    app = Flask(__name__)
    if config_object:
        app.config.from_object(config_object)

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    from .cv_filtration.routes import cv_bp
    from .recommendation.routes import rec_bp
    from .interview.routes import intr_bp


    app.register_blueprint(cv_bp, url_prefix='/api')
    app.register_blueprint(rec_bp, url_prefix='/api')
    app.register_blueprint(intr_bp, url_prefix='/api/interview')

    create_dash_app(app)

    return app
