import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import tensorflow as tf
from tensorflow.keras.utils import pad_sequences
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import Tokenizer

from data import load_data


def build_tokenizer(texts):
    tokenizer = Tokenizer(char_level=True, lower=False)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def label_seq(seq, labels, tokenizer):
    new_seq = []
    label_seq = []
    for i, c in enumerate(seq[:-1]):
        if i in labels:
            continue

        new_seq.append(c)

        if i + 1 in labels:
            next_char = tokenizer.index_word[seq[i + 1]]
            if next_char == "'":
                label_seq.append(1)
            elif next_char == '"':
                label_seq.append(2)
            else:
                raise ValueError("Noooo!")
        else:
            label_seq.append(0)

    new_seq.append(seq[-1])
    label_seq.append(0)

    return new_seq, label_seq


def create_tensors(sequences, labels, tokenizer, maxlen):
    seq_labels = [label_seq(s, l, tokenizer) for s, l in zip(sequences, labels)]
    sequences = [s for s, l in seq_labels]
    labels = [l for s, l in seq_labels]
    weights = [[1] * len(l) for l in labels]

    padded_sequences = pad_sequences(sequences, padding="post", maxlen=maxlen)
    padded_labels = pad_sequences(labels, padding="post", maxlen=maxlen)
    weights = pad_sequences(weights, padding="post", maxlen=maxlen)
    weights[padded_labels > 0] = 100

    return padded_sequences, padded_labels, weights


def to_tf_dataset(inputs, outputs, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
    dataset = dataset.map(lambda x, y: (x, tf.expand_dims(tf.cast(y, "float32"), 1)))
    dataset = dataset.batch(batch_size)

    return dataset


def train_model(cfg, train_inputs, train_outputs, train_sample_weights, test_dataset, n_chars):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(cfg.LSTM.MAX_LEN,)),
            tf.keras.layers.Embedding(n_chars, cfg.LSTM.HIDDEN_DIM, name="char_emb", input_length=cfg.LSTM.MAX_LEN),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    cfg.LSTM.HIDDEN_DIM, name="encoder", input_shape=(n_chars, cfg.LSTM.HIDDEN_DIM), return_sequences=True
                )
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(cfg.LSTM.HIDDEN_DIM, return_sequences=True, name="decoder")
            ),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(3, activation="softmax", name="predictor")
            ),
        ],
        name="char_binary_lstm",
    )
    optimizer = tf.keras.optimizers.Adam()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True,
    )
    save_ckpt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(cfg.OUTPUT, "char_lstm"), save_best_only=True
    )
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(cfg.OUTPUT, "training.log"))

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizer,
        sample_weight_mode="temporal",
    )
    model.summary()
    model.fit(
        train_inputs,
        train_outputs,
        validation_data=test_dataset,
        epochs=cfg.LSTM.SOLVER.EPOCHS,
        sample_weight=np.expand_dims(train_sample_weights, -1),
        callbacks=[early_stopping, save_ckpt, csv_logger],
    )
    return model

def evaluate_model(model, test_dataset):
    y_pred = model.predict(test_dataset)
    y_pred = y_pred.argmax(-1)
    y_true = test_dataset.map(lambda x, y: y)
    y_true = np.concatenate([x for x in y_true], axis=0)

    y_true = y_true.reshape(-1, 1).squeeze()
    y_pred = y_pred.reshape(-1, 1).squeeze()

    print(classification_report(y_true, y_pred))


def main(cfg):
    sentences, labels = load_data(cfg.INPUT)
    tokenizer = build_tokenizer(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    inputs, outputs, sample_weights = create_tensors(sequences, labels, tokenizer, cfg.LSTM.MAX_LEN)
    (
        train_inputs,
        test_inputs,
        train_outputs,
        test_outputs,
        train_sample_weights,
        _,
    ) = train_test_split(
        inputs, outputs, sample_weights, test_size=0.1, stratify=outputs.max(axis=-1)
    )
    test_dataset = to_tf_dataset(test_inputs, test_outputs, batch_size=cfg.LSTM.SOLVER.BATCH_SIZE)
    model = train_model(cfg, train_inputs, train_outputs, train_sample_weights, test_dataset, len(tokenizer.index_word) + 1)
    evaluate_model(model, test_dataset)
