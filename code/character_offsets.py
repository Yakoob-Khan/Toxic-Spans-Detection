
def toxic_character_offsets(offset_mapping, prediction):
  toxic_offsets = []
  for i, (offset, pred_label) in enumerate(zip(offset_mapping, prediction)):
    # if token is predicted to be toxic, add the character indices
    if pred_label == 1:
      toxic_offsets.extend([index for index in range(offset[0], offset[1])])
  
  return toxic_offsets


def character_offsets(val_offset_mapping, predictions):
  return [toxic_character_offsets(offset_mapping, prediction) for offset_mapping, prediction in zip(val_offset_mapping, predictions)]
