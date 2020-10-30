
def toxic_character_offsets(tokens, offset_mapping, prediction):
  toxic_offsets = set()
  n = len(tokens)
  for i, (token, offset, pred_label) in enumerate(zip(tokens, offset_mapping, prediction)):
    if i == 0:
      continue
    if token == '[PAD]':
      break
    # if token is predicted to be toxic
    if pred_label == 1:
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


def character_offsets(val_text_encodings, val_offset_mapping, predictions):
  return [toxic_character_offsets(val_text_encodings[i].tokens, offset_mapping, prediction) for i, (offset_mapping, prediction) in enumerate(zip(val_offset_mapping, predictions))]
