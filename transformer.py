import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk

from datasets import load_dataset, load_from_disk
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification
)
import evaluate


from data import load_data


def index_safe(s, c):
    try:
        return s.index(c)
    except:
        return -1


def count(s, c):
    x = 0
    for i in range(len(s)):
        if s[i] == c:
            x += 1
    return x


def label_seq(sent, labels):
    new_seq = []
    label_i = 0
    char_c = 0
    reg_tok = nltk.RegexpTokenizer("[\w'\"]+|[.,;:?!\-\(\)]")
    seq = reg_tok.tokenize(sent)
    label_seq = [0] * len(seq)

    for i, c in enumerate(seq):
        try:
            word_len = len(c)
            j = index_safe(c, "'")
            if j != -1:
                j_ = len(c) - j
                if j_ <= 6:
                    label_seq[i] = j_

                label_i += 1

            j = count(c, '"')
            if j > 0:
                label_i += 1

            if j == 1:
                j = index_safe(c, '"')
                if j == 0:
                    label_seq[i] = 7
                else:
                    label_seq[i] = 8
            elif j > 1:
                label_seq[i] = 9
            new_seq.append(c.replace('"', "").replace("'", ""))

            if label_i == len(labels):
                new_seq[i + 1 :] = seq[i + 1 :]
                break

        except:
            new_seq.append(c)

        char_c += word_len + 1

    return new_seq, label_seq


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            if label > 0:
                label = 2 * label - 1
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label > 0:
                label *= 2

            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels_token_cls(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["text"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["label"]
    new_labels = []

    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def tokenize_and_align_labels_seq2seq(examples, tokenizer):
    tokenized_inputs = tokenizer.batch_encode_plus(
        examples["text"],
        pad_to_max_length=True,
        max_length=512,
        is_split_into_words=True,
        return_tensors="pt",
    )

    labels = []
    for i, l in enumerate(examples["label"]):
        word_ids = tokenized_inputs.word_ids(i)
        new_l = align_labels_with_tokens(l, word_ids)
        labels.append([str(w) for w in new_l])

    tokenized_labels = tokenizer.batch_encode_plus(
        labels,
        pad_to_max_length=True,
        max_length=512,
        is_split_into_words=True,
        return_tensors="pt",
    )
    tokenized_inputs["decoder_attention_mask"] = tokenized_labels["attention_mask"]
    tokenized_inputs["decoder_input_ids"] = tokenized_labels["input_ids"]
    labels = tokenized_labels["input_ids"].clone().detach()
    labels[labels[:, :] == tokenizer.pad_token_id] = -100
    tokenized_inputs["labels"] = labels

    return tokenized_inputs


def prepare_dataset(tokenizer, preprocess_func, cfg):
    sentences, labels = load_data(cfg.INPUT)
    seq_labels = [label_seq(s, l) for s, l in zip(sentences, labels)]
    labels = [l for _, l in seq_labels]
    train_seq_labels, test_seq_labels = train_test_split(
        seq_labels, test_size=cfg.TEST_SPLIT, stratify=[max(l) for l in labels]
    )
    df = pd.DataFrame(train_seq_labels, columns=["text", "label"])
    df.to_json(os.path.join(cfg.INPUT, "train_data.json"), orient="records", lines=True)
    df = pd.DataFrame(test_seq_labels, columns=["text", "label"])
    df.to_json(os.path.join(cfg.INPUT, "test_data.json"), orient="records", lines=True)
    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(cfg.INPUT, "train_data.json"),
            "test": os.path.join(cfg.INPUT, "test_data.json"),
        },
    )
    tokenized_dataset = dataset.map(
        lambda x: preprocess_func(x, tokenizer),
        batched=True,
        load_from_cache_file=False,
    )
    tokenized_dataset = tokenized_dataset.remove_columns(["label", "text"])
    tokenized_dataset.save_to_disk(os.path.join(cfg.INPUT, "tokenized_dataset_t5"))
    return tokenized_dataset


def get_classes():
    base_classes = np.arange(0, 10)
    classes = np.concatenate([base_classes * 2, base_classes[1:] * 2 - 1])
    classes = np.sort(classes)
    classes = list(map(int, classes))
    id2label = {0: "O"}
    for i in range(1, len(classes), 2):
        if i < 13:
            pre = "AP"
        else:
            pre = "QU"
        id2label[classes[i]] = f"B-{pre}-{base_classes[i//2] + 1}"
        id2label[classes[i + 1]] = f"I-{pre}-{base_classes[i//2] + 1}"

    label2id = {v: k for k, v in id2label.items()}
    return base_classes, id2label, label2id


def to_int_safe(s):
    try:
        return int(s)
    except:
        return 0


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


def get_metric_func_seq2seq(label2id, base_classes, tokenizer):
    seqeval = evaluate.load("seqeval")
    label_list = list(label2id.keys())

    def compute_metrics(p):
        predictions, labels = p
        predictions = predictions[0]

        true_predictions = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        decoded_preds = tokenizer.batch_decode(
            true_predictions, skip_special_tokens=True
        )
        decoded_labels = tokenizer.batch_decode(true_labels, skip_special_tokens=True)

        decoded_preds = [
            [
                label_list[to_int_safe(t)]
                if 0 <= to_int_safe(t) < len(label_list)
                else "O"
                for t in seq.split()
            ]
            for seq in decoded_preds
        ]
        decoded_labels = [
            [
                label_list[to_int_safe(t)]
                if 0 <= to_int_safe(t) < len(label_list)
                else "O"
                for t in seq.split()
            ]
            for seq in decoded_labels
        ]
        decoded_preds = [
            [t for t in decoded_preds[i]]
            + ["O"] * (len(decoded_labels[i]) - len(decoded_preds[i]))
            for i in range(len(decoded_preds))
        ]

        results = seqeval.compute(predictions=decoded_preds, references=decoded_labels)

        res = {}

        for c in base_classes[1:]:
            pre = "AP" if c < 7 else "QU"
            label_type = f"{pre}-{c}"
            try:
                res[f"{label_type}_f1"] = results[label_type]["f1"]
                res[f"{label_type}_number"] = results[label_type]["number"]
            except:
                print(f"{label_type} does not exist.")

        res["overall_f1"] = results["overall_f1"]

        return res

    return compute_metrics


def get_metric_func_token_cls(label2id, base_classes):
    seqeval = evaluate.load("seqeval")
    label_list = list(label2id.keys())

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)

        res = {}

        for c in base_classes[1:]:
            pre = "AP" if c < 7 else "QU"
            label_type = f"{pre}-{c}"
            try:
                res[f"{label_type}_f1"] = results[label_type]["f1"]
                res[f"{label_type}_number"] = results[label_type]["number"]
            except:
                print(f"{label_type} does not exist.")

        return res

    return compute_metrics

def main(cfg):
    base_classes, id2label, label2id = get_classes()
    if cfg.MODEL == 'BERT':
        tokenizer = AutoTokenizer.from_pretrained(cfg.BERT.BACKBONE)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        model = AutoModelForTokenClassification.from_pretrained(
            cfg.BERT.BACKBONE,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )
        compute_metrics = get_metric_func_token_cls(label2id, base_classes)
        preprocess_func = tokenize_and_align_labels_token_cls
    else:
        tokenizer = T5TokenizerFast.from_pretrained(cfg.T5.BACKBONE)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
        model = T5ForConditionalGeneration.from_pretrained(
            cfg.T5.BACKBONE,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )
        compute_metrics = get_metric_func_seq2seq(label2id, base_classes, tokenizer)
        preprocess_func = tokenize_and_align_labels_seq2seq


    model_cfg = getattr(cfg, cfg.MODEL)

    if cfg.DATA.LOAD_DATASET_FROM_DISK and os.path.exists(
        os.path.join(cfg.INPUT, cfg.DATA.DATASET_NAME)
    ):
        tokenized_dataset = load_from_disk(
            os.path.exists(os.path.join(cfg.INPUT, cfg.DATA.DATASET_NAME))
        )
    else:
        tokenized_dataset = prepare_dataset(tokenizer, preprocess_func, cfg)

    training_args = TrainingArguments(
        output_dir=cfg.OUTPUT,
        learning_rate=cfg.TRANSFORMER_SOLVER.LR,
        per_device_train_batch_size=cfg.TRANSFORMER_SOLVER.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=cfg.TRANSFORMER_SOLVER.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=cfg.TRANSFORMER_SOLVER.GRAD_ACC_STEPS,
        gradient_checkpointing=cfg.TRANSFORMER_SOLVER.GRAD_CKPT,
        num_train_epochs=cfg.TRANSFORMER_SOLVER.EPOCHS,
        weight_decay=cfg.TRANSFORMER_SOLVER.WEIGHT_DECAY,
        fp16=cfg.TRANSFORMER_SOLVER.FP16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    if model_cfg.RESUME_FROM_CKPT:
        ckpt_path = model_cfg.CKPT_PATH
    else:
        ckpt_path = False
    
    trainer.train(ckpt_path)
