import argparse
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from training.config import DATA_PATH, MODEL_NAME_BASELINE
from training.data_utils import load_data, split_train_val_test
from training.metrics_utils import compute_binary_metrics
from training.mlflow_utils import init_mlflow, log_common_tags, log_metrics_dict, maybe_register_model

def main(max_features: int, ngram_max: int, C: float):
    init_mlflow()

    df = load_data(DATA_PATH)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_train_val_test(df)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, ngram_max),
            lowercase=True
        )),
        ("clf", LogisticRegression(
            C=C,
            max_iter=200,
            class_weight="balanced"
        ))
    ])

    with mlflow.start_run(run_name="baseline_tfidf_logreg"):
        log_common_tags(model_family="sklearn", approach="baseline")

        mlflow.log_param("max_features", max_features)
        mlflow.log_param("ngram_max", ngram_max)
        mlflow.log_param("C", C)

        pipe.fit(X_train, y_train)

        val_proba = pipe.predict_proba(X_val)[:, 1]
        test_proba = pipe.predict_proba(X_test)[:, 1]

        val_metrics = compute_binary_metrics(y_val, val_proba)
        test_metrics = compute_binary_metrics(y_test, test_proba)

        log_metrics_dict(val_metrics, prefix="val_")
        log_metrics_dict(test_metrics, prefix="test_")

        mlflow.sklearn.log_model(pipe, artifact_path="model", registered_model_name=None)

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        maybe_register_model(model_uri, MODEL_NAME_BASELINE)

        print("Baseline terminé. Metrics test:", test_metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_features", type=int, default=50000)
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--C", type=float, default=2.0)
    args = ap.parse_args()
    main(args.max_features, args.ngram_max, args.C)
