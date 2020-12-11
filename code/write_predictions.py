from utils import _contiguous_ranges

def toxic_strings(text, span):
    toxic_spans = _contiguous_ranges(span)
    return [text[i: j+1] for i, j in toxic_spans]


def write_toxic_strings(file_path, val_texts, gold_char_offsets, toxic_char_preds):
  """
  Each line is comma separated lists
  [original text],[gold_toxic],[pred_toxic]
  """
  f = open(file_path, "w")

  for text, gold, pred in zip(val_texts, gold_char_offsets, toxic_char_preds):
    gold_toxic = toxic_strings(text, gold)
    pred_toxic = toxic_strings(text, pred)
    f.write(f"Text: {text} \n")
    f.write(f"Gold: {str(gold_toxic)} \n")
    f.write(f"Pred: {str(pred_toxic)} \n")
    f.write(f'\n')

  f.close()

