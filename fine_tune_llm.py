import warnings
import llm_helpers
import pandas as pd
import logging
import argparse
import json
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
)
import pandas as pd
from transformers import EarlyStoppingCallback, ProgressCallback
from peft import LoraConfig, get_peft_model, TaskType

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


def main(args):
    model_name = args.model_name
    use_peft = args.use_peft
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    chunk_text = args.chunk_text

    logger.info(f"Model name: {model_name}")
    logger.info(f"Use PEFT: {use_peft}")
    logger.info(f"Train data path: {train_data_path}")
    logger.info(f"Validation data path: {val_data_path}")
    logger.info(f"Chunk text: {chunk_text}")

    device = llm_helpers.get_device()
    logger.info(f"Using device: {device}")

    train_df = pd.read_csv(train_data_path)[["text", "label"]].reset_index(drop=True)
    val_df = pd.read_csv(val_data_path)[["text", "label"]].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    if chunk_text:
        logger.info("Chunking text...")
        train_df["text"] = train_df["text"].apply(
            lambda x: llm_helpers.chunk_text(x, tokenizer)
        )
        train_df = train_df.explode("text").reset_index(drop=True)

    train_ds, val_ds = llm_helpers.prepare_datasets(train_df, val_df)

    if use_peft:
        if model_name in ["bert-base-uncased", "roberta-base", "distilroberta-base"]:
            target_modules = ["query", "key", "value"]
        else:
            target_modules = ["q_lin", "k_lin", "v_lin"]
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_modules,
            bias="none",
            lora_dropout=0.05,
            task_type=TaskType.SEQ_CLS,
        )

        model = get_peft_model(model, lora_config)

    data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    def tokenize(batch):
        return tokenizer(
            batch["text"], padding="max_length", truncation=True, return_tensors="pt"
        )

    train_ds_encoded = train_ds.map(tokenize, batched=True)
    val_ds_encoded = val_ds.map(tokenize, batched=True)

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=5, early_stopping_threshold=0.001
    )

    progress_callback = ProgressCallback()

    # training_args = TrainingArguments(
    #     f"{model_name}-trainer",
    #     num_train_epochs=15,
    #     per_device_train_batch_size=32,
    #     fp16=True,
    #     per_device_eval_batch_size=32,
    #     # logging_steps=100,
    #     evaluation_strategy="epoch",
    #     logging_strategy='epoch',
    #     save_strategy='epoch',
    #     # eval_steps=100,
    #     metric_for_best_model="eval_loss", # it can be also brier_score etc
    #     load_best_model_at_end=True,
    #     lr_scheduler_type="cosine",
    #     warmup_ratio=0.1,
    # )

    training_args = TrainingArguments(
        f"{model_name}-trainer-chunk-{chunk_text}",
        num_train_epochs=15,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=1000,
        evaluation_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        metric_for_best_model="eval_loss",  # it can be also brier_score etc
        load_best_model_at_end=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_ds_encoded,
        eval_dataset=val_ds_encoded,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=llm_helpers.compute_metrics,
        callbacks=[early_stopping, progress_callback],
    )

    logger.info("Starting training...")
    trainer.train()
    with open(
        f"fine_tuned_{model_name}_chunk_{chunk_text}_train_log_history.json", "w"
    ) as outfile:
        json.dump(trainer.state.log_history, outfile)
    logger.info("Training completed!")
    logger.info("Saving model...")

    trainer.save_model(f"fine-tuned-{model_name}-chunk-{chunk_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to fine-tune a language model")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "bert-base-uncased",
            "distilbert-base-uncased",
            "roberta-base",
            "distilroberta-base",
        ],
        default="distilbert-base-uncased",
        help="Model name",
    )
    parser.add_argument("--use_peft", action="store_true", help="Use PEFT adapter")
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="train_df.csv",
        help="Path to train data file",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="test_df.csv",
        help="Path to validation data file",
    )
    parser.add_argument(
        "--chunk_text",
        action="store_true",
        help="Chunk text into smaller pieces to accomodate model input size",
    )
    args = parser.parse_args()
    main(args)


# Stage 1

# python fine_tune_llm.py --model_name distilbert-base-uncased --use_peft
# python fine_tune_llm.py --model_name bert-base-uncased --use_peft
# python fine_tune_llm.py --model_name roberta-base --use_peft
# python fine_tune_llm.py --model_name distilroberta-base --use_peft

# python fine_tune_llm.py --model_name distilbert-base-uncased
# python fine_tune_llm.py --model_name bert-base-uncased
# python fine_tune_llm.py --model_name roberta-base
# python fine_tune_llm.py --model_name distilroberta-base

# Stage 2

# python fine_tune_llm.py --model_name distilbert-base-uncased --chunk_text
# python fine_tune_llm.py --model_name bert-base-uncased --chunk_text
# python fine_tune_llm.py --model_name roberta-base --chunk_text
# python fine_tune_llm.py --model_name distilroberta-base --chunk_text

# python fine_tune_llm.py --model_name distilbert-base-uncased
# python fine_tune_llm.py --model_name bert-base-uncased
# python fine_tune_llm.py --model_name roberta-base
# python fine_tune_llm.py --model_name distilroberta-base
