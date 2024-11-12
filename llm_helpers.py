import json
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import ClassLabel, Dataset, Features, Value
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def prepare_datasets(train_df, val_df):
    class_names = ["human", "chatgpt"]
    features = Features(
        {"text": Value("string"), "label": ClassLabel(num_classes=2, names=class_names)}
    )

    train_ds = Dataset.from_pandas(train_df, features=features)
    val_ds = Dataset.from_pandas(val_df, features=features)

    return train_ds, val_ds


def chunk_text(text, tokenizer, max_token_count=512):
    min_token_count = max_token_count // 8
    tokens = tokenizer.encode(text, add_special_tokens=True)
    chunks = [
        tokens[i : i + max_token_count] for i in range(0, len(tokens), max_token_count)
    ]
    if len(chunks) > 1:
        chunks = list(filter(lambda x: len(x) >= min_token_count, chunks))

    chunked_text = [
        tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks
    ]

    return chunked_text


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    y_true = labels
    y_pred = np.argmax(logits, axis=-1)
    y_proba = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    brier_score = brier_score_loss(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "brier_score": brier_score,
        "roc_auc": roc_auc,
    }

    return metrics_dict
