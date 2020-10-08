import string
import numpy as np

from transformers import BertTokenizerFast
from load_dataset import load_dataset


tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

def split_and_identify_labels(text, span):
    """ 
      - Split a post into words using space delimeter.
      - For each word, identify its corresponding label (toxic vs non-toxic).
    """
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
        label = [1] * size if chars[p1] == 1 else [0] * size
    
        tokenized_sentence.extend(tokens)
        labels.extend(label)

        p1 = p2

    return tokenized_sentence, labels


def tokenize_data(texts, spans):
    """
      - Split each posts into space delimeted words and get its corresponding labels
      - 1: toxic, 0: non-toxic
    """
    print('> Tokenizing data..')
    
    tokenized_texts_and_labels = [split_and_identify_labels(text, span) for text, span in zip(texts, spans)]
    
    # separate the tokenized texts and labels
    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    print('-- Done!\n')

    return tokenized_texts, labels
    

def encode_text(tokenized_texts):
    # use the BERT tokenizer to generate the embedding for the input texts
    text_encodings = tokenizer(tokenized_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    return text_encodings


def encode_labels(labels, text_encodings):
    """
      - Use the offset mappings from the BERT tokenizer to identify the label
        of each sub-token.
      - 1: toxic, 0: non-toxic
      - CLS, SEP and PAD tokens are set to -100.
    """
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

        # Need to set the label of sub-token BPE's to be same as the head BPE
        doc_enc_labels = doc_enc_labels.tolist()

        # find the [SEP] token index
        sep_token_index = None
        for i in range(len(doc_enc_labels)-1, -1, -1):
          if doc_enc_labels[i] != -100:
            sep_token_index = i + 1
            break
        
        last_label = None
        for i in range(1, sep_token_index):
          # remember the last label if sub-token label is not -100
          if doc_enc_labels[i] != -100:
            last_label = doc_enc_labels[i]
          
          # set bpe of sub-token same as respective head bpe
          elif doc_enc_labels[i] == -100:
            doc_enc_labels[i] = last_label
        
        # append the bpe labels encodings
        encoded_labels.append(doc_enc_labels)

    return encoded_labels

