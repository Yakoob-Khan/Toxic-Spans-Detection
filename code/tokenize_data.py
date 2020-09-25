import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig

from load_dataset import load_dataset


def get_toxic_spans(span):
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


def get_labels(text, toxic_spans):
    toxic_spans = [tuple(span) for span in toxic_spans]
    toxic_spans = set(toxic_spans)
    print(toxic_spans)
    words = text.split()
    start = 0
    labels = [0] * len(words)
    for i, word in enumerate(words):
        end = start + len(word)
        print(text[start:end], start, end)
        if (start, end-1) in toxic_spans:
            labels[i] = 1
        
        start += len(word) + 1

    print(labels)
    return labels


    


def tokenize_and_preserve_labels(text, span):
    pass


def tokenize_data(texts, spans):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    for i, span in enumerate(spans):
        if i < 1:
            text = texts[i]
            print(text)
            toxic_spans = get_toxic_spans(span)
            labels = get_labels(text, toxic_spans)
            
        else:
            break

        
    # for i, text in enumerate(texts):
    #     if i < 3:
    #         print(text)
    #         print(tokenizer.tokenize(text), '\n')
    #     else:
    #         break
        
        


if __name__ == '__main__':
    texts, spans = load_dataset('../data/tsd_trial.csv')
    tokenize_data(texts, spans)