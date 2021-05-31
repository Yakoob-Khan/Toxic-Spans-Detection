
def write_bert_confidence_scores(gold_labels_encodings, prediction_scores, experiment_name):
  y_true, scores = [], []
  for gold_labels, pred in zip(gold_labels_encodings, prediction_scores):
    # ignore the [CLS] and [PAD] tokens
    start, end = 1, gold_labels[1:].index(-100) + 1
    # add the true labels
    y_true.extend(gold_labels[start:end])
    # keep the scores of the toxic prediction label only
    probs = [prob[1] for prob in pred[start:end]]
    scores.extend(probs)
  
  # write the true labels and confidence scores in a text file for visualization later
  f = open(experiment_name, "w")
  f.write(f"{y_true} \n")
  f.write(f"{scores} \n")
  f.close()




  
    

    
    