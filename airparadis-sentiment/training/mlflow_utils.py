import mlflow
from training.config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME, REGISTER_MODEL

def init_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

def log_common_tags(model_family: str, approach: str):
    mlflow.set_tag("project", "airparadis_sentiment")
    mlflow.set_tag("model_family", model_family)   # sklearn / keras / bert
    mlflow.set_tag("approach", approach)           # baseline / advanced_dl / bert

def log_metrics_dict(metrics: dict, prefix: str = ""):
    for k, v in metrics.items():
        mlflow.log_metric(f"{prefix}{k}", float(v))

def maybe_register_model(model_uri: str, registered_name: str):
    if not REGISTER_MODEL:
        return None
    return mlflow.register_model(model_uri, registered_name)
