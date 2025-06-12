from flask import Flask
from flask_socketio import SocketIO

socketio = SocketIO()

def create_app(config_object=None):
    app = Flask(__name__)
    if config_object:
        app.config.from_object(config_object)

    # Initialize extensions
    socketio.init_app(app, cors_allowed_origins="*")

    # Register Blueprints
    from .cv_filtration.routes import cv_bp
    from .recommendation.routes import rec_bp
    from .technical_skills.routes import tech_bp
    # from .soft_skills import soft_bp

    app.register_blueprint(cv_bp, url_prefix='/api')
    app.register_blueprint(rec_bp, url_prefix='/api')
    app.register_blueprint(tech_bp, url_prefix='/api')
    # app.register_blueprint(soft_bp, url_prefix='/soft-skills')

    # SocketIO event handlers
    # from . import socketio_handlers

    return app
