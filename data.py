import os
import pickle
import argparse

from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

from datasets import load_dataset

nltk.download("punkt")


def get_sentences_from_silicone():
    datasets = [
        "dyda_da",
        "dyda_e",
        "iemocap",
        "maptask",
        "meld_e",
        "meld_s",
        "mrda",
        "oasis",
        "sem",
        "swda",
    ]
    sentences = []
    for d in datasets:
        dataset = load_dataset("silicone", d)
        for split in dataset:
            for x in tqdm(dataset[split]):
                sentence = x["Utterance"]
                if "'" in sentence or '"' in sentence:
                    sentences.append(sentence)
    return sentences


def sentences_summary(sentences):
    n = len(sentences)
    n_lens = [len(s) for s in sentences]
    print(f"#{n}")
    print("Avg:", round(sum(n_lens) / n, 2))
    print("Max:", max(n_lens))
    print("Min:", min(n_lens))


def filter_by_len(sentences, min_len=50, max_len=500):
    sentences = [s for s in sentences if min_len <= len(s) <= max_len]
    sentences = sorted(sentences, key=lambda s: len(s))
    return sentences


def get_sentences_from_wiki():
    dataset = load_dataset(
        "wikipedia", date="20221120", language="simple", beam_runner="DirectRunner"
    )
    sentences = []
    for i, x in tqdm(enumerate(dataset["train"]), total=len(dataset["train"])):
        text = x["text"]
        ss = sent_tokenize(text)
        ss = [s for s in ss if "'" in s or '"' in s]
        sentences += ss

    return sentences


def filter_unlabeled(sentences, labels):
    sentences_f = []
    labels_f = []
    for s, l in zip(sentences, labels):
        if len(l) > 0:
            sentences_f.append(s)
            labels_f.append(l)
    return sentences_f, labels_f


def find_indices(text, c):
    return [i for i in range(len(text)) if text[i] in c]


def load_data(path):
    sentences = pickle.load(open(os.path.join(path, "sentences.pkl"), "rb"))
    labels = pickle.load(open(os.path.join(path, "labels.pkl"), "rb"))
    return sentences, labels


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-len", type=int, default=50, help="Discard sentences with length less than this value")
    parser.add_argument("--max-len", type=int, default=500, help="Discard sentences with length greater than this value")
    parser.add_argument("--silicone", action="store_true", help="Include informal datasets")
    parser.add_argument("--wiki", action="store_true", help="Include formal dataset")
    parser.add_argument("--output-path", default="./data/", help="Output path for gathered data")

    return parser.parse_args()


def main(args):
    all_sentences = []
    if args.silicone:
        all_sentences += get_sentences_from_silicone()
    if args.wiki:
        all_sentences += get_sentences_from_wiki()

    assert len(all_sentences) > 0

    all_sentences = filter_by_len(
        all_sentences, min_len=args.min_len, max_len=args.max_len
    )
    all_labels = [find_indices(s, "'\"") for s in all_sentences]
    all_sentences, all_labels = filter_unlabeled(all_sentences, all_labels)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    pickle.dump(
        all_sentences, open(os.path.join(args.output_path, "sentences.pkl"), "wb")
    )
    pickle.dump(all_labels, open(os.path.join(args.output_path, "labels.pkl"), "wb"))


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
