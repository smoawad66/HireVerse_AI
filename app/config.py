import os

class BaseConfig:
    # SECRET_KEY = os.getenv('SECRET_KEY', 'dev')
    pass

class DevelopmentConfig(BaseConfig):
    DEBUG = True

class ProductionConfig(BaseConfig):
    DEBUG = False
