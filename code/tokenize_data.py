import string
import numpy as np

from transformers import BertTokenizerFast
from load_dataset import load_dataset


tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

def split_and_identify_labels(text, span):
    tokenized_sentence, labels = [], []

    n = len(text)
    chars = [0] * n
    for s in span: chars[s] = 1

    p1, p2 = 0, 0
    while p1 < n:
        while p2 < n and chars[p2] == chars[p1]:
            p2 += 1

        tokens = text[p1:p2].split()
        size = len(tokens)
        label = [2]*size if chars[p1] == 1 else [0]*size
        if chars[p1] == 1:
            label[0] = 1
        
        tokenized_sentence.extend(tokens)
        labels.extend(label)

        p1 = p2

    return tokenized_sentence, labels


def tokenize_data(texts, spans):
    print('> Tokenizing data..')

    tokenized_texts_and_labels = [split_and_identify_labels(text, span) for text, span in zip(texts, spans)]
    
    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    print('-- Done!\n')

    return tokenized_texts, labels
    

def encode_text(tokenized_texts):
    text_encodings = tokenizer(tokenized_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    return text_encodings


def encode_labels(labels, text_encodings):
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, text_encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # TODO: hacky fix as two training data points array size is off by one error. not sure why yet..
        if doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)].size != len(doc_labels):
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels[:-1]
        else:
            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

