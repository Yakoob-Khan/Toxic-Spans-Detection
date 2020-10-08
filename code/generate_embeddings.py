from tokenize_data import encode_text, encode_labels

def generate_embeddings(tokenized_texts, labels):
    # generate the text and label embeddings
    text_encodings = encode_text(tokenized_texts)
    label_encodings = encode_labels(labels, text_encodings)

    return text_encodings, label_encodings
