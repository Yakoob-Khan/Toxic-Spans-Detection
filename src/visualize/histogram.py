import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ast import literal_eval

plt.figure(dpi=600)
plt.style.use('seaborn-darkgrid')
plt.rc('axes', labelsize=18)
plt.rc('font', size=12.5)  


def plot_histogram(data_dir, name):
  dataset = pd.read_csv(data_dir)
  dataset["spans"] = dataset.spans.apply(literal_eval)
  spans = [len(span) for span in dataset["spans"] if len(span) < 100]

  plt.hist(spans, density=False, bins=150)  
  plt.ylabel('Count')
  plt.xlabel('Span Length'); 
  plt.title(f"{name} Set");

  plt.savefig(f'./output/span_histogram_{name}.pdf')


# plot_histogram('../data/tsd_train.csv', 'Train')
# plot_histogram('../data/tsd_trial.csv', 'Dev')
plot_histogram('../data/tsd_test.csv', 'Test')