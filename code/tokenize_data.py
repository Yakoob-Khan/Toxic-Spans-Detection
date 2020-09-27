import time
import string

from transformers import BertTokenizer
from load_dataset import load_dataset


def tokenize_and_preserve_labels(tokenizer, text, span):
    tokenized_sentence, labels = [], []

    n = len(text)
    chars = [0] * n
    for s in span: chars[s] = 1

    p1, p2 = 0, 0
    while p1 < n:
        while p2 < n and chars[p2] == chars[p1]:
            p2 += 1

        label = chars[p1]
        for word in text[p1:p2].split():
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            tokenized_sentence.extend(tokenized_word)
            labels.extend([label] * n_subwords)

        p1 = p2

    return tokenized_sentence, labels


def tokenize_data(texts, spans):
    print('> Tokenizing data..')

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tokenized_texts_and_labels = [tokenize_and_preserve_labels(tokenizer, text, span) for text, span in zip(texts, spans)]
    
    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    print('-- Done!\n')

    return tokenized_texts, labels
    

if __name__ == '__main__':
    start = time.time()
    texts, spans = load_dataset('../data/tsd_train.csv')
    tokenized_texts, labels = tokenize_data(texts, spans)
    end = time.time()
    print(f"Time: {end-start}s")