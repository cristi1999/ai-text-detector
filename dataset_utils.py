from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(df, target_col, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    return train_df, test_df


def create_stratified_train_test_dfs(dir_path="csv_files", target_col="label"):
    path = Path(dir_path)
    train_dfs = []
    test_dfs = []

    for file in path.iterdir():
        df_ = pd.read_csv(file)
        df_["dataset"] = file.stem
        train_df, test_df = stratified_split(df_, target_col)
        train_dfs.append(train_df)
        test_dfs.append(test_df)
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    return train_df, test_df
