import argparse
import os
from datetime import datetime

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    RobertaConfig,
    RobertaForMaskedLM,
    Trainer,
    TrainingArguments,
)
from utils import fix_seed

MAX_LENGTH = 185
EVAL_STEPS = 2500


def main(args):
    timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    tokenizer = BertTokenizerFast(
        vocab_file=args.vocab_file,
        do_lower_case=False,
        do_basic_tokenize=False,
        unk_token="<unk>",
        sep_token="</s>",
        pad_token="<pad>",
        cls_token="<s>",
        mask_token="<mask>",
        tokenize_chinese_chars=False,
    )
    datasets = load_dataset(
        "COGNANO/VHHCorpus-2M",
        data_files={
            "train": "train.csv",
            "validation": "valid.csv",
        },
    )
    datasets = datasets.map(lambda examples: {"VHH_sequence": " ".join(examples["VHH_sequence"])})
    tokenized_datasets = datasets.map(
        lambda examples: tokenizer(examples["VHH_sequence"], return_special_tokens_mask=True),
        batched=True,
    )
    tokenized_datasets = tokenized_datasets.remove_columns(["VHH_sequence", "token_type_ids"])
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )
    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=MAX_LENGTH,
    )
    model = RobertaForMaskedLM(config)
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        seed=args.seed,
        data_seed=args.seed,
        fp16=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        logging_strategy="steps",
        logging_steps=EVAL_STEPS,
        logging_dir=os.path.join(args.save_dir, "logs", timestamp),
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        learning_rate=1e-4,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
    )

    def compute_metrics(eval_preds):
        metric = evaluate.load("accuracy")
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return metric.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab-file",
        type=str,
        required=True,
        help="Path of the vocabulary file.",
    )
    parser.add_argument(
        "--save-dir",
        default="./saved",
        type=str,
        help="Save directory path (default: ./saved)",
    )
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        help="Number of epochs (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        default=128,
        type=int,
        help="The size of mini-batch (default: 128)",
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
