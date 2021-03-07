def preserve_labels(text_encoding, span):
  labels = [0] * len(text_encoding.tokens)
  toxic_indices = set(span)
  for i, offset in enumerate(text_encoding.offsets):
    # labels for CLS, SEP and PAD tokens are set to -100.
    if offset == (0, 0):
      labels[i] = -100
    
    else:
      # check if any character indices of this sub-token has gold label toxic 
      for k in range(offset[0], offset[1]):
        if k in toxic_indices: 
          # toxic, so set label to 1.
          labels[i] = 1
          break
  
  return labels
      

def tokenize_data(tokenizer, texts, spans):
    text_encodings = tokenizer(texts, return_offsets_mapping=True, padding=True, truncation=True)
    labels = [preserve_labels(text_encodings[i], span) for i, span in enumerate(spans)]
    return text_encodings, labels


def tokenize_testset(tokenizer, texts):
    text_encodings = tokenizer(texts, return_offsets_mapping=True, padding=True, truncation=True)
    dummy_labels = [[0] * len(tokens) for i, tokens in enumerate(text_encodings.input_ids)]
    return text_encodings, dummy_labels


def tokenize_sentences(tokenizer, sentences):
    return tokenizer(sentences, return_offsets_mapping=True, padding=True, truncation=True)
