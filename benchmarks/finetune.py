import argparse
import json
import os
import pickle
from datetime import datetime

import evaluate
import torch
from datasets import load_dataset
from models import PalmForBindingPrediction
from sklearn.metrics import average_precision_score
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from utils import create_dataset, fix_seed

MAX_LENGTH = 185


def main(args):
    timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    datasets = load_dataset("COGNANO/AVIDa-SARS-CoV-2", data_files={"train": "train.csv"})
    datasets = datasets["train"].train_test_split(test_size=0.1, seed=args.seed)
    datasets["valid"] = datasets["test"]
    datasets["test"] = load_dataset("COGNANO/AVIDa-SARS-CoV-2", data_files={"test": "test.csv"})[
        "test"
    ]
    tokenized_datasets, tokenizer = create_dataset(datasets, palm_type=args.palm_type)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    with open(args.embeddings_file, "rb") as f:
        embeddings_dict = pickle.load(f)
    tokenized_datasets = tokenized_datasets.map(
        lambda examples: {"antigen_embeddings": embeddings_dict[examples["Ag_label"]]}
    )
    tokenized_datasets = tokenized_datasets.remove_columns(["Ag_label"])
    model = PalmForBindingPrediction(palm_type=args.palm_type, model_path=args.model_path)

    def compute_metrics(eval_pred):
        logits = torch.sigmoid(torch.tensor(eval_pred.predictions))
        labels = eval_pred.label_ids
        predictions = torch.round(logits)

        accuracy = evaluate.load("accuracy")
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")
        f1 = evaluate.load("f1")
        mcc = evaluate.load("matthews_correlation")
        roc_auc = evaluate.load("roc_auc")

        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(
            torch.tensor(eval_pred.predictions), torch.tensor(labels, dtype=torch.float)
        ).item()

        accuracy_score = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        precision_score = precision.compute(
            predictions=predictions, references=labels, average="binary"
        )["precision"]
        recall_score = recall.compute(
            predictions=predictions, references=labels, average="binary"
        )["recall"]
        f1_score = f1.compute(predictions=predictions, references=labels, average="binary")["f1"]
        mcc_score = mcc.compute(predictions=predictions, references=labels)["matthews_correlation"]
        roc_auc_score = roc_auc.compute(prediction_scores=logits, references=labels)["roc_auc"]
        pr_auc_score = average_precision_score(labels, logits)

        return {
            "loss": loss,
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
            "mcc": mcc_score,
            "roc_auc": roc_auc_score,
            "pr_auc": pr_auc_score,
        }

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        seed=args.seed,
        data_seed=args.seed,
        fp16=True,
        eval_strategy="epoch",
        logging_strategy="epoch",
        logging_dir=os.path.join(args.save_dir, "logs", timestamp),
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=1e-6,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()
    test_result = trainer.evaluate(tokenized_datasets["test"])
    print(test_result)
    with open(os.path.join(args.save_dir, "test_result.json"), "w") as f:
        json.dump(test_result, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings-file",
        default="./benchmarks/data/embeddings.pkl",
        type=str,
        help="Path of embeddings file (default: ./benchmarks/data/embeddings.pkl",
    )
    parser.add_argument(
        "--palm-type",
        default="VHHBERT",
        type=str,
        help="PALM type must be one of ['VHHBERT', 'VHHBERT-w/o-PT', 'AbLang', 'AntiBERTa2', 'AntiBERTa2-CSSP', 'ESM-2', 'IgBert', 'ProtBert'].",
    )
    parser.add_argument(
        "--save-dir",
        default="./saved",
        type=str,
        help="Save directory path (default: ./saved)",
    )
    parser.add_argument(
        "--epochs",
        default=30,
        type=int,
        help="Number of epochs (default: 30)",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="The size of batch (default: 32)",
    )
    parser.add_argument(
        "--seed",
        default=123,
        type=int,
        help="Random seed (default: 123)",
    )
    args = parser.parse_args()
    fix_seed(args.seed)
    main(args)
