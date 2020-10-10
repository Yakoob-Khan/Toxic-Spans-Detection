import numpy as np

from character_offsets import character_offsets

def system_precision_recall_f1(toxic_char_preds, gold_char_offsets):

  def per_post_precision_recall_f1(predictions, gold):
    if len(gold) == 0:
        return [1.0, 1.0, 1.0] if len(predictions) == 0 else [0.0, 0.0, 0.0]

    if len(predictions) == 0:
        return [0.0, 0.0, 0.0]
    
    nom = len(set(predictions).intersection(set(gold)))
    precision = nom / len(set(predictions))
    recall = nom / len(set(gold))
    f1_score = (2 * nom) / (len(set(predictions)) + len(set(gold)))

    return [float(precision), float(recall), float(f1_score)]
  
  # get the respective metrics per post
  precision_recall_f1_scores = [per_post_precision_recall_f1(toxic_offsets, gold_offsets) for toxic_offsets, gold_offsets in zip(toxic_char_preds, gold_char_offsets)]
  
  # compute average precision, recall and f1 score of all posts
  return np.array(precision_recall_f1_scores).mean(axis=0)


def compute_metrics(pred, gold_char_offsets, val_offset_mapping):
  # get the sub-token predictions made by the model
  predictions = pred.predictions.argmax(-1)

  # retrieve the toxic character offsets of these predictions
  toxic_char_preds = character_offsets(val_offset_mapping, predictions)

  # compute the precision, recall and f1 score on the validation set
  precision, recall, f1 = system_precision_recall_f1(toxic_char_preds, gold_char_offsets)

  return {
    'precision': precision,
    'recall': recall,
    'f1': f1
  }
