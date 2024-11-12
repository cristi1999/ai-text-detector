def get_xgb_params(trial, scale_pos_weight=1):
    params = {
        "scale_pos_weight": scale_pos_weight,
        "n_jobs": -1,
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, 50),
        "max_depth": trial.suggest_int("max_depth", 1, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.5, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-9, 1.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-9, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-9, 1.0, log=True),
    }
    return params


def get_svm_params(trial):
    params = {
        "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
        "n_jobs": -1,
        "gamma": trial.suggest_float("gamma", 1e-9, 1.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
        "degree": trial.suggest_int("degree", 1, 5),
        "class_weight": "balanced",
    }
    return params


def get_lr_params(trial):
    params = {
        "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
        "n_jobs": -1,
        "solver": "liblinear",
        "max_iter": trial.suggest_int("max_iter", 100, 1000, 100),
        "class_weight": "balanced",
    }
    return params


def get_rf_params(trial):
    params = {
        "class_weight": "balanced",
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, 50),
        "n_jobs": -1,
        "max_depth": trial.suggest_int("max_depth", 1, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["auto", "sqrt", "log2"]
        ),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }
    return params


def get_knn_params(trial):
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 10),
        "n_jobs": -1,
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "algorithm": trial.suggest_categorical(
            "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
        ),
        "leaf_size": trial.suggest_int("leaf_size", 10, 50, 10),
        "p": trial.suggest_int("p", 1, 5),
    }
    return params


def get_gaussian_nb_params(trial):
    params = {
        "var_smoothing": trial.suggest_float("var_smoothing", 1e-9, 1e-3, log=True)
    }
    return params
