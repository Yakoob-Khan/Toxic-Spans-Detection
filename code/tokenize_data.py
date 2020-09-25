import time

import string
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig

from load_dataset import load_dataset

def get_toxic_span(span):
    toxic_spans = []
    cur = [None, None]
    for offset in span:
        if not cur[0]:
            cur = [offset, offset]
        elif offset == cur[1] + 1:
            cur[1] = offset
        else:
            toxic_spans.append(cur)
            cur = [offset, offset]

    if cur[0] and cur[1]:
        toxic_spans.append(cur)
    
    return toxic_spans


def remove_punctuation(word):
    return word.translate(str.maketrans('', '', string.punctuation))
    

def tokenize_and_preserve_labels(tokenizer, text, span):
    toxic_spans = get_toxic_span(span)
    toxic_words = { remove_punctuation(word) for start, end in toxic_spans for word in text[start:end+1].split()}
    
    tokenized_sentence, labels = [], []

    for word in text.split():
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        label = 1 if remove_punctuation(word) in toxic_words else 0

        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)
    
    return tokenized_sentence, labels
        

def tokenize_data(texts, spans):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tokenized_texts_and_labels = [tokenize_and_preserve_labels(tokenizer, text, span) for text, span in zip(texts, spans)]
    
    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    return tokenized_texts, labels
    

if __name__ == '__main__':
    start = time.time()
    texts, spans = load_dataset('../data/tsd_trial.csv')
    tokenize_data(texts, spans)
    end = time.time()
    print(f"Time: {end-start}s")