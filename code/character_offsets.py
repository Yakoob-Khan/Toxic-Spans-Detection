import numpy as np

def toxic_character_offsets(tokenizer, input_id, offset_mapping, prediction):
  text = tokenizer.decode(np.array(input_id), skip_special_tokens=True)
  toxic_offsets = []
  prev = 0

  for i in range(1, len(offset_mapping)):
    x, y = offset_mapping[i]
    if (x, y) == (0, 0):
      # PAD seq tokens, break out of loop
      break

    # check if current sub-token is head BPE (x = 0) or sub BPE
    x = prev + 1 if i > 1 and x == 0 else prev
    y = x + (offset_mapping[i][1] - offset_mapping[i][0])

    if prediction[i] == 1:
      # add toxic character offset predictions
      toxic_offsets.extend([k for k in range(x, y)])
    
    prev = y 
  
  return toxic_offsets


def character_offsets(tokenizer, input_ids, val_offset_mapping, predictions):
  toxic_char_preds = []
  for input_id, offset_mapping, prediction in zip(input_ids, val_offset_mapping, predictions):
    toxic_offsets = toxic_character_offsets(tokenizer, input_id, offset_mapping, prediction)
    toxic_char_preds.append(toxic_offsets)
  
  return toxic_char_preds

