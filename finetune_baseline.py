import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import random
from collections import Counter
from utils import NusaXDataset, NusaTranslationDataset, TatoebaDataset, BUCCDataset, LinceMTDataset, PhincDataset, LinceSADataset, MassiveIntentDataset, Sib200Dataset, NollySentiDataset, MTOPIntentDataset, FIREDataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset
import evaluate
import argparse
import os
import json

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_tokenized_dataset(lang_id, tokenizer, max_length):
    def tokenize_function(example):
        return tokenizer(example['source'], truncation=True, padding='max_length')
    
    train_dataset = Dataset.from_dict(dataset.train_data[lang_id])
    val_dataset = Dataset.from_dict(dataset.valid_data[lang_id])
    test_dataset = Dataset.from_dict(dataset.test_data[lang_id])
    tokenized_train_datasets= train_dataset.map(tokenize_function, batched=True)
    tokenized_val_datasets = val_dataset.map(tokenize_function, batched=True)
    tokenized_test_datasets = test_dataset.map(tokenize_function, batched=True)
    tokenized_train_datasets = tokenized_train_datasets.rename_column("target", "labels")  
    tokenized_val_datasets = tokenized_val_datasets.rename_column("target", "labels")
    tokenized_test_datasets = tokenized_test_datasets.rename_column("target", "labels")
    tokenized_train_datasets.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
    tokenized_val_datasets.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
    tokenized_test_datasets.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
    return tokenized_train_datasets, tokenized_val_datasets, tokenized_test_datasets

def compute_metrics(eval_preds):
    """Computes accuracy."""
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds  # raw scores of size (batch_size, num_classes), labels 0-2
    predictions = np.argmax(logits, axis=-1)  # take max of last dimension (num_classes)
    return metric.compute(predictions=predictions, references=labels)

def train_eval_loop(dataset_name, lang_id, output_dir, model_name, label_num, compute_metrics):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=label_num)
    model_output_dir = f"model_out/{dataset_name}_{model_name}_{lang_id}"
    BATCH_SIZE = 8
    max_length = tokenizer.model_max_length
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_train_datasets, tokenized_val_datasets, tokenized_test_datasets = get_tokenized_dataset(lang_id, tokenizer, max_length)

    training_args = TrainingArguments(
                do_train=True,
                do_eval=True,
                output_dir=model_output_dir,
                num_train_epochs=10,
                weight_decay=0.01,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                save_strategy="epoch",
                save_total_limit=1,
                eval_strategy="epoch",
                greater_is_better=True,
                metric_for_best_model="accuracy",
                load_best_model_at_end=True,
                logging_dir=f"finetune_logs/{dataset_name}_{model_name}_{lang_id}",
                logging_steps=100,
            )
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,       # Number of evaluation steps to wait for improvement
        early_stopping_threshold=0.0     # Threshold for measuring the new optimum
    )

    trainer = Trainer(
                model=model,
                args=training_args,
                tokenizer=tokenizer,
                data_collator=data_collator,
                train_dataset=tokenized_train_datasets,
                eval_dataset=tokenized_val_datasets,
                compute_metrics=compute_metrics,
                callbacks=[early_stopping_callback]
            )
    print("start training")
    trainer.train()

    metric = evaluate.load("f1")
    model.eval()
    for batch in tokenized_test_datasets:
        batch = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['labels']}
        with torch.no_grad():
            outputs = model(batch['input_ids'].unsqueeze(0).to('cuda'), attention_mask=batch['attention_mask'].unsqueeze(0).to('cuda'))

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=[predictions], references=[batch["labels"]])

    f1 = metric.compute(average='micro')
    if not os.path.exists(f"{output_dir}/{dataset_name}/"):
        os.makedirs(f"{output_dir}/{dataset_name}/")
    with open(f"{output_dir}/{dataset_name}/{lang_id}.json", "w") as f:
        dict_ = {"dataset_name":dataset_name, "lang_id":lang_id, "f1": f1["f1"]}
        json.dump(dict_, f)
    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--model", type=str, default="xlm-roberta-base", help="model name")
    parser.add_argument("--dataset", type=str, default="nusax", help="dataset name")
    parser.add_argument("--prompt", type=str, default="", help="prompt")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--src_lang", type=str, default="x", help="source language")
    args = parser.parse_args()
    
    print("###########################")
    print("dataset:", args.dataset)
    print("model:", args.model)
    print("prompt:", args.prompt)
    print("seed:", args.seed)
    print("cross:", args.cross)
    print("src_lang:", args.src_lang)
    print("###########################")

    set_seed(args.seed)

    if args.cross:
        output_dir = f"finetune_classification_cross_{args.src_lang}"
    else:
        output_dir = "finetune_classification"


    if args.dataset == "nusax":
        dataset = NusaXDataset(prompt=args.prompt, task="classification")
        label_num = 3
    if args.dataset == "lince_sa":
        dataset = LinceSADataset(prompt=args.prompt)
        label_num = 3
    if args.dataset == "massive_intent":
        dataset = MassiveIntentDataset(prompt=args.prompt)
        label_num = 60
    if args.dataset == "sib200":
        dataset = Sib200Dataset(prompt=args.prompt)
        label_num = 7
    if args.dataset == "nollysenti":
        dataset = NollySentiDataset(prompt=args.prompt, task="classification")
        label_num = 2
    if args.dataset == "mtop_intent":
        dataset = MTOPIntentDataset(prompt=args.prompt)
        label_num = 113
    
    if args.dataset == 'fire':
        dataset = FIREDataset(prompt=args.prompt)
        for lang in dataset.LANGS:
            print("language: ", lang)
            lang_id = lang
            if lang == 'malayalam':
                label_num = 4
            elif lang == 'tamil':
                label_num = 5
            MODEL_NAME = args.model
            train_eval_loop(args.dataset, lang_id, output_dir, MODEL_NAME, label_num, compute_metrics)
    else:
        for lang in dataset.LANGS:
            print("language: ", lang)
            lang_id = lang
            MODEL_NAME = args.model
            train_eval_loop(args.dataset, lang_id, output_dir, MODEL_NAME, label_num, compute_metrics)
