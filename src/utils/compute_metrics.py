import numpy as np

from post_process.character_offsets import character_offsets_with_thresholding, character_offsets_with_late_fusion

def system_precision_recall_f1(toxic_char_preds, gold_char_offsets):

  def per_post_precision_recall_f1(predictions, gold):
    if len(gold) == 0:
        return [1.0, 1.0, 1.0] if len(predictions) == 0 else [0.0, 0.0, 0.0]

    if len(predictions) == 0:
        return [0.0, 0.0, 0.0]
    
    predictions_set = set(predictions)
    gold_set = set(gold)
    nom = len(predictions_set.intersection(gold_set))
    precision = nom / len(predictions_set)
    recall = nom / len(gold_set)
    f1_score = (2 * nom) / (len(predictions_set) + len(gold_set))

    return [float(precision), float(recall), float(f1_score)]
  
  # get the respective metrics per post
  precision_recall_f1_scores = [per_post_precision_recall_f1(toxic_offsets, gold_offsets) for toxic_offsets, gold_offsets in zip(toxic_char_preds, gold_char_offsets)]
  
  # compute average precision, recall and f1 score of all posts
  return np.array(precision_recall_f1_scores).mean(axis=0)


def compute_metrics(pred, gold_char_offsets, val_offset_mapping, val_text_encodings, val_sentences_info, threshold=-float('inf')):
  # get the sub-token predictions made by the model
  predictions = pred.predictions.argmax(-1)
  prediction_scores = pred.predictions

  # retrieve the toxic character offsets of these predictions
  toxic_char_preds_object = character_offsets_with_thresholding(val_text_encodings, val_offset_mapping, predictions, val_sentences_info, prediction_scores, threshold)
  toxic_char_offsets = [span[0] for span in toxic_char_preds_object]

  # compute the precision, recall and f1 score on the validation set
  precision, recall, f1 = system_precision_recall_f1(toxic_char_offsets, gold_char_offsets)

  return {
    'precision': precision,
    'recall': recall,
    'f1': f1
  }


