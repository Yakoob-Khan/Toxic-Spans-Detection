
def toxic_character_offsets(post_num, tokens, offset_mapping, prediction, val_sentences_info):
  # post_num_to_sentence_nums = val_sentences_info['val_post_to_sentence_num'][post_num]
  # sentence_num_to_label_pred = val_sentences_info['val_sentence_pred']
  # sentence_num_to_spans = val_sentences_info['val_sentence_spans']
  # val_post = val_sentences_info['val_texts'][post_num]
  # sentences = val_sentences_info['post_to_sentences_classification'][val_post]
  
  toxic_offsets = set()
  n = len(tokens)
  for i, (token, offset, pred_label) in enumerate(zip(tokens, offset_mapping, prediction)):
    if i == 0:
      continue
    if token == '[PAD]':
      break
    # if token is predicted to be toxic
    if pred_label == 1:
      # find the sentence number that this token belongs to
      # for span in sentences.keys():
      #   # split by comma to get the start and end offset spans
      #   a, d = span.split(',')
      #   # remove parenthesis
      #   a, d = int(a[1:]), int(d[:-1])
      #   # get the span of this token
      #   b, c = offset
      #   # check if the token is located within this sentence 
      #   if a <= b and c <= d:
      #     break
      
      # if sentences[span]['Pred'] == 1:
      # proceed to add the token character offsets only if the respective sentence
      # of this token is predicted toxic by the sentence classifier.

      # add the characters offsets of this token
      toxic_offsets = toxic_offsets.union({index for index in range(offset[0], offset[1])})

      # if head bpe, check if previous token is also predicted toxic.
      # if so, then toxic phrase has been found. 
      if '#' not in token:
        if i - 1 >= 0 and offset_mapping[i-1][1] - 1 in toxic_offsets:
          # print(tokens[i-1], tokens[i], offset_mapping[i-1], offset)
          toxic_offsets = toxic_offsets.union({index for index in range(offset_mapping[i-1][1], offset[0])})

      # if sub-token is predicted toxic
      if '#' in token:
        # ensure all previous bpe's of this token is predicted toxic as well
        k = i - 1
        while k >= 0 and '#' in tokens[k]:
          toxic_offsets = toxic_offsets.union({index for index in range(offset_mapping[k][0], offset_mapping[k][1])})
          k -= 1
        # head bpe of this sub-token is toxic as well
        if k >= 0:
          # toxic_offsets.extend([index for index in range(offset_mapping[k][0], offset_mapping[k][1])])
          toxic_offsets = toxic_offsets.union({index for index in range(offset_mapping[k][0], offset_mapping[k][1])})
        
      # any sub-tokens ahead should be predicted toxic as well
      k = i + 1
      while k < n and '#' in tokens[k]:
        toxic_offsets = toxic_offsets.union({index for index in range(offset_mapping[k][0], offset_mapping[k][1])})
        k += 1
  
  return sorted(list(toxic_offsets))


def character_offsets(val_text_encodings, val_offset_mapping, predictions, val_sentences_info):
  return [toxic_character_offsets(i, val_text_encodings[i].tokens, offset_mapping, prediction, val_sentences_info) for i, (offset_mapping, prediction) in enumerate(zip(val_offset_mapping, predictions))]
