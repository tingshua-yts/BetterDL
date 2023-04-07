"""Build vocabularies of words and labels from datasets"""
import argparse
from collections import Counter
import json
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=1, help="Minimum count for words in the dataset", type=int)
parser.add_argument('--data_dir', default='dbpedia/dbpedia_csv', help="Directory containing the dataset")

# Hyper parameters for the vocab
NUM_OOV_BUCKETS = 1 # number of buckets (= number of ids) for unknown words
PAD_WORD = '<pad>'


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9\'\`]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\`", "\'", text)
    text = text.strip().lower()
    return text

def save_vocab_to_txt_file(vocab, txt_path):
    """Writes one token per line, 0-based line id corresponds to the id of the token.
    Args:
        vocab: (iterable object) yields token
        txt_path: (stirng) path to vocab file
    """
    with open(txt_path, "w") as f:
        f.write("\n".join(token for token in vocab))


def save_dict_to_json(d, json_path):
    """Saves dict to json file
    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def update_vocab(txt_path, vocab):
    """Update word and tag vocabulary from dataset
    Args:
        txt_path: (string) path to file, one sentence per line
        vocab: (dict or Counter) with update method
    Returns:
        dataset_size: (int) number of elements in the dataset
    """
    with open(txt_path) as f:
        for i, line in enumerate(f):
            fields = line.strip().split(',')
            if len(fields) < 3: continue
            text = clean_str(fields[2])
            tokens = text.split()
            tokens = [w.strip("'") for w in tokens if len(w.strip("'")) > 0]
            vocab.update(tokens)
    return i + 1


if __name__ == '__main__':
    args = parser.parse_args()

    # Build word vocab with train and test datasets
    print("Building word vocabulary...")
    words = Counter()
    size_train_sentences = update_vocab(os.path.join(args.data_dir, 'train.csv'), words)
    size_test_sentences = update_vocab(os.path.join(args.data_dir, 'test.csv'), words)
    print("- done.")

    # Only keep most frequent tokens
    words = [tok for tok, count in words.items() if count >= args.min_count_word]

    # Add pad tokens
    if PAD_WORD not in words: words.append(PAD_WORD)

    # Save vocabularies to file
    print("Saving vocabularies to file...")
    save_vocab_to_txt_file(words, os.path.join(args.data_dir, 'words.txt'))
    print("- done.")

    # Save datasets properties in json file
    sizes = {
        'train_size': size_train_sentences,
        'test_size': size_test_sentences,
        'vocab_size': len(words) + NUM_OOV_BUCKETS,
        'pad_word': PAD_WORD,
        'num_oov_buckets': NUM_OOV_BUCKETS
    }
    save_dict_to_json(sizes, os.path.join(args.data_dir, 'dataset_params.json'))

    # Logging sizes
    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))