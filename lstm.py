import pickle
from functools import reduce
import re
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import pad_sequences
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import Tokenizer


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

        if i+1 in labels:
            next_char = tokenizer.index_word[seq[i+1]]
            if next_char == "'":
                label_seq.append(1)
            elif next_char == '"':
                label_seq.append(2)
            else:
                raise ValueError('Noooo!')
        else:
            label_seq.append(0)

    new_seq.append(seq[-1])
    label_seq.append(0)

    return new_seq, label_seq

def create_tensors(sequences, labels):
    seq_labels = [label_seq(s, l) for s, l in zip(sequences, labels)]
    sequences = [s for s, l in seq_labels]
    labels = [l for s, l in seq_labels]
    weights = [[1] * len(l) for l in labels]

    padded_sequences = pad_sequences(sequences, padding='post', maxlen=t)
    padded_labels = pad_sequences(labels, padding='post', maxlen=t)
    weights = pad_sequences(weights, padding='post', maxlen=t)
    weights[padded_labels > 0] = 100

    return padded_sequences, padded_labels, weights

def to_tf_dataset(inputs, outputs, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
    dataset = dataset.map(lambda x, y: (x, tf.expand_dims(tf.cast(y, 'float32'), 1)))
    dataset = dataset.batch(batch_size)

    return dataset


def train_model(cfg):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(t,)),
            tf.keras.layers.Embedding(k, h, name='char_emb', input_length=t),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(h, name='encoder', input_shape=(k, h), return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(h, return_sequences=True, name='decoder')),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3, activation="softmax", name='predictor')),
        ],
        name='char_binary_lstm'
    )
    optimizer = tf.keras.optimizers.Adam()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True,
    )
    save_ckpt = tf.keras.callbacks.ModelCheckpoint(f'{path}char_lstm', save_best_only=True)
    csv_logger = tf.keras.callbacks.CSVLogger(f'{path}training.log')

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        optimizer=optimizer,
        sample_weight_mode='temporal'
    )
    model.summary()
    model.fit(train_inputs, train_outputs, validation_data=val_dataset, epochs=epochs, sample_weight=np.expand_dims(train_sample_weights, -1), callbacks=[early_stopping, save_ckpt, csv_logger])

