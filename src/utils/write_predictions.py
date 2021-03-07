from utils.helper import _contiguous_ranges

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

def write_test_strings(file_path, test_texts, toxic_char_preds):
  f = open(file_path, "w")
  for text, pred in zip(test_texts, toxic_char_preds):
    pred_toxic = toxic_strings(text, pred)
    f.write(f"Text: {text} \n")
    f.write(f"Pred: {str(pred_toxic)} \n")
    f.write(f'\n')
  f.close()


def write_toxic_strings_with_prob(file_path, val_texts, toxic_char_preds, gold_char_offsets=None):
  """
  Each line is comma separated lists
  [original text],[gold_toxic],[pred_toxic]
  """
  if gold_char_offsets is None:
    # no labels in test set
    f = open(file_path, "w")
    for text, pred in zip(val_texts, toxic_char_preds):
      pred_toxic = toxic_strings(text, pred[0])
      f.write(f"Text: {text} \n")
      f.write(f"Pred: {str(pred_toxic)} \n")
      f.write(f'Scores: {str(pred[1])}\n')
      f.write(f'\n')
    f.close()
  else:
    # has labels in val set
    f = open(file_path, "w")
    for text, gold, pred in zip(val_texts, gold_char_offsets, toxic_char_preds):
      gold_toxic = toxic_strings(text, gold)
      pred_toxic = toxic_strings(text, pred[0])
      f.write(f"Text: {text} \n")
      f.write(f"Gold: {str(gold_toxic)} \n")
      f.write(f"Pred: {str(pred_toxic)} \n")
      f.write(f'Scores: {str(pred[1])}\n')
      f.write(f'\n')
    f.close()


def create_submission_file(filepath, toxic_char_preds):
  # Ref: https://github.com/ipavlopoulos/toxic_spans/blob/master/ToxicSpans_SemEval21.ipynb
  # write in a prediction file named "spans-pred.txt"
  with open(filepath, "w") as out:
    for uid, text_scores in enumerate(toxic_char_preds):
      out.write(f"{str(uid)}\t{str(text_scores)}\n")
