import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def relevant_labels(gold_labels, predictions):
  """
    - Only consider the labels for the sub-token BPE's
    - ignore the labels for CLS, SEP, PAD tokens BPE's 
  """
  relevant_golds, relevant_predicts = [], []
  for gold_label, prediction in zip(gold_labels, predictions):
    # find the SEP token index
    sep_token_idx = np.where(gold_label[1:] == -100)[0][0] + 1

    # add the relevant bpe gold and prediction labels
    relevant_golds.append(gold_label[1: sep_token_idx])
    relevant_predicts.append(prediction[1:sep_token_idx])
    
  return np.array(relevant_golds), np.array(relevant_predicts)


def compute_metrics(pred):
  # extract the gold labels and model predictions
  gold_labels, predictions = pred.label_ids, pred.predictions.argmax(-1)

  # consider gold labels and predictions only for text sub-tokens.
  y_trues, y_preds = relevant_labels(gold_labels, predictions)
  
  total_precision = total_recall = total_f1 = total_accuracy = 0

  # loop through each prediction 
  for y_true, y_pred in zip(y_trues, y_preds):
    # get metrics for each prediction
    precision, recall, f1, s = precision_recall_fscore_support(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    
    # add respective score to total
    total_precision += precision
    total_recall += recall
    total_f1 += f1
    total_accuracy += accuracy
  
  n = y_trues.shape[0]
  # return the average respective scores
  return {
      'f1': total_f1 / n,
      'accuracy': total_accuracy / n,
      'precision': total_precision / n,
      'recall': total_recall / n,
  }

def f1_system_score(toxic_char_pred, gold_char_offsets):
  def f1(predictions, gold):
    """
      Credit: https://github.com/ipavlopoulos/toxic_spans/blob/master/evaluation/semeval2021.py
      F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
      >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
      :param predictions: a list of predicted offsets
      :param gold: a list of offsets serving as the ground truth
      :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1. if len(predictions) == 0 else 0.
    if len(predictions) == 0:
        return 0.
    nom = 2*len(set(predictions).intersection(set(gold)))
    denom = len(set(predictions))+len(set(gold))
    return float(nom)/float(denom)
  
  f1_scores = [f1(toxic_offsets, gold_offsets) for toxic_offsets, gold_offsets in zip(toxic_char_pred, gold_char_offsets)]
  
  return np.array(f1_scores).mean()
