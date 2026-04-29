# models/__init__.py — Easy import for all models

from models.cnn_basic       import CNNBasic
from models.cnn_deep        import CNNDeep
from models.cnn_lightweight import CNNLightweight

# Model registry — used by train.py and app.py
MODEL_REGISTRY = {
    "CNN Basic"       : CNNBasic,
    "CNN Deep"        : CNNDeep,
    "CNN Lightweight" : CNNLightweight,
}

def get_model(model_name):
    """
    Get model class by name.
    Usage: model = get_model("CNN Basic")()
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]()