from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

plt.figure(dpi=600)
plt.rc('axes', labelsize=16)
plt.rc('font', size=13)      

def create_confusion_matrix(test_encodings, test_predictions, test_labels_encodings):
  y_true, y_pred = [], []
  
  for i, (pred, gold) in enumerate(zip(test_predictions, test_labels_encodings)):
    sep_token = 1
    tokens = test_encodings[i].tokens
    while tokens[sep_token] != '[SEP]':
      sep_token += 1
    y_true.extend(gold[1: sep_token])
    y_pred.extend(pred[1: sep_token])

  # Credits for heatmap code using Seaborn.
  # https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
  cf_matrix = confusion_matrix(y_true, y_pred)
  labels = ['Neutral', 'Toxic']
  
  ax = plt.axes()
  sns_plot = sns.heatmap(cf_matrix / np.sum(cf_matrix), 
                        annot=True, fmt='.2%',
                        xticklabels=labels, yticklabels=labels, 
                        ax = ax,
                        cmap="YlGnBu")

  ax.set_xlabel('Predicted')
  ax.set_ylabel('Actual')
  sns_plot.get_figure().savefig("confusion_matrix.pdf")

