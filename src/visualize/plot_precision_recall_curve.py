import matplotlib.pyplot as plt
import numpy as np
import json
import random

from sklearn.metrics import precision_recall_curve, auc
from ast import literal_eval

plt.figure(dpi=600)
# plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-darkgrid')

plt.rc('axes', labelsize=18)
plt.rc('font', size=12.5)  
palette = plt.get_cmap('tab10')


def plot_precision_recall_curve(scores_files, labels):
  data = {}
  # read all the score files
  for score_file, experiment in zip(scores_files, labels):
    f = open(score_file, 'r')
    i = 0
    for line in f:
      # convert to array
      arr = literal_eval(line)

      # write the information to the data dictionary
      if experiment not in data:
        data[experiment] = dict()
      
      if i == 0:
        # gold labels
        data[experiment]['y_true'] = arr

      if i == 1:
        # probabilities
        data[experiment]['y_scores'] = arr

      i += 1
  
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  num = 0

  # Random model. Horizontal line where y = Positive / Total
  positive = sum([label for label in data['BERToxic']['y_true'] if label == 1])
  total = len(data['BERToxic']['y_true'])
  y = positive / total
  num_points = 1000
  precision = [y] * num_points
  recall, delta = [0] * num_points, 1 / num_points
  for i in range(1, num_points):
    recall[i] = recall[i-1] + delta 
  area = round(auc(recall, precision), 3)
  p, = plt.plot(recall, precision, marker='', color=palette(num), linewidth=1.5, alpha=1, label=f"Random ({area})")
  num += 1

  num_points = 0
  for experiment in data:
    # retrieve the true labels and prediction scores
    y_true = data[experiment]['y_true']
    y_scores = data[experiment]['y_scores']
    # use sklearn function to generate the points
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    num_points = len(precision)
    area = round(auc(recall, precision), 3)
    # plot precision-recall curve
    p, = plt.plot(recall, precision, marker='', color=palette(num), linewidth=1.5, alpha=1, label=f"{experiment} ({area})")
    num += 1

  # MT-DNN
  y_true = data['BERToxic']['y_true']
  with open('./ensemble_modeling/multi_task_learning/ner_test_scores_epoch_2.json') as f:
    output = json.load(f)
    preds = output['predictions']
  y_scores = []
  for pred in preds: y_scores.extend(pred[1:-1])
  # use sklearn function to generate the points
  precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
  area = round(auc(recall, precision), 3)

  # plot precision-recall curve
  p, = plt.plot(recall, precision, marker='', color=palette(num), linewidth=1.5, alpha=1, label=f"BERT Multi-task ({area})")

  # with open('./ensemble_modeling/multi_task_learning/ner_test_scores_epoch_2.json') as f:
  #   output = json.load(f)
  #   preds = output['scores']
  #   print(len(y_true))
  #   print(len(preds[len(preds)-len(y_true):]))

  plt.legend(loc='best', bbox_to_anchor=(0.1, 0.2, 0.5, 0.5))

  print(f"> Saved figure as ./output/precision_recall_curve.pdf")
  plt.savefig('./output/precision_recall_curve.pdf')

scores_files = ['output/scores_SpaCy.txt', 'output/scores_BERToxic.txt', 
                'output/scores_EDA.txt', 'output/scores_HateXplain.txt', 
                'output/scores_Late_Fusion.txt'
                ]

names = ['SpaCy', 'BERToxic', '+ EDA', '+ HateXplain', 'BERT Late Fusion']
plot_precision_recall_curve(scores_files, names)
