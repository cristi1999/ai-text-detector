import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
)
import optuna_utils
from optuna.samplers import TPESampler


def get_base_classifier(trial, model_type, scale_pos_weight=1):
    match model_type:
        case "xgb":
            params = optuna_utils.get_xgb_params(trial, scale_pos_weight)
            classifier = XGBClassifier(**params)
        case "svm":
            params = optuna_utils.get_svm_params(trial)
            classifier = SVC(**params)
        case "lr":
            params = optuna_utils.get_lr_params(trial)
            classifier = LogisticRegression(**params)
        case "rf":
            params = optuna_utils.get_rf_params(trial)
            classifier = RandomForestClassifier(**params)
        case "knn":
            params = optuna_utils.get_knn_params(trial)
            classifier = KNeighborsClassifier(**params)
        case "gaussian_nb":
            params = optuna_utils.get_gaussian_nb_params(trial)
            classifier = GaussianNB(**params)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")
    return classifier


def define_classifier(base_classifier, sparse=True):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "tf_idf",
                TfidfVectorizer(stop_words="english", max_features=5000),
                0,
            ),
            ("scaler", StandardScaler(), slice(1, None)),
        ],
        remainder="passthrough",
    )

    classifier = (
        Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "calibrated_classifier",
                    CalibratedClassifierCV(base_classifier, method="isotonic"),
                ),
            ]
        )
        if sparse
        else Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "to_array",
                    FunctionTransformer(lambda x: x.toarray(), accept_sparse=True),
                ),
                (
                    "calibrated_classifier",
                    CalibratedClassifierCV(base_classifier, method="isotonic"),
                ),
            ]
        )
    )
    return classifier


def objective(trial, model_type, X, y, sparse=True):
    scale_pos_weight = sum(y == 0) / sum(y == 1)
    base_classifier = get_base_classifier(trial, model_type, scale_pos_weight)
    classifier = define_classifier(base_classifier, sparse)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = cross_validate(
        classifier, X, y, cv=skf, scoring=["neg_brier_score", "roc_auc"], verbose=10
    )
    brier_score = results["test_neg_brier_score"].mean()
    roc_auc = results["test_roc_auc"].mean()
    return brier_score, roc_auc


def hyperparameter_search(X, y, model_type, n_trials=20, sparse=True):
    objective_fn = lambda trial: objective(trial, model_type, X, y, sparse)
    study = optuna.create_study(
        directions=["maximize", "maximize"], sampler=TPESampler()
    )
    study.optimize(objective_fn, n_trials=n_trials, n_jobs=-1)

    print("Number of finished trials: ", len(study.trials))
    best_configs = {}
    for i, trial in enumerate(study.best_trials):
        brier_score = -trial.values[0]
        roc_auc = -trial.values[1]
        best_configs[f"trial_{i}"] = {"params": trial.params} | {
            "values": {"brier_score": brier_score, "roc_auc": roc_auc}
        }
    return best_configs


def evaluate_classifier(y_proba, y_test):
    dummy_brier_pred = [y_test.value_counts(normalize=True).max()] * len(y_test)
    ref_brier = brier_score_loss(y_test, dummy_brier_pred)

    accuracy = accuracy_score(y_test, y_proba > 0.5)
    roc_auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    brier_skill = 1 - brier / ref_brier
    precision = precision_score(y_test, y_proba > 0.5)
    recall = recall_score(y_test, y_proba > 0.5)
    f1 = f1_score(y_test, y_proba > 0.5)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Brier Skill Score: {brier_skill:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


def evaluate_classifier_cv(base_classifier, X, y, sparse=True):
    classifier = define_classifier(base_classifier, sparse)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = cross_validate(
        classifier,
        X,
        y,
        cv=skf,
        scoring=["neg_brier_score", "roc_auc", "accuracy"],
        verbose=10,
    )
    brier_scores = -results["test_neg_brier_score"]
    roc_auc_scores = results["test_roc_auc"]
    accuracy_scores = results["test_accuracy"]
    return brier_scores, roc_auc_scores, accuracy_scores
