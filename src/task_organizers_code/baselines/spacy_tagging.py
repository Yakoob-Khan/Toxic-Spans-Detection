# Lint as: python3
"""Example tagging for Toxic Spans based on Spacy.

Requires:
  pip install spacy sklearn

Install models:
  python -m spacy download en_core_web_sm

"""

import ast
import csv
import random
import statistics
import sys
import time

import sklearn
import spacy
import numpy as np

sys.path.append('../evaluation')
import semeval2021
import fix_spans

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

def spans_to_ents(doc, spans, label):
  """Converts span indicies into spacy entity labels."""
  started = False
  left, right, ents = 0, 0, []
  for x in doc:
    if x.pos_ == 'SPACE':
      continue
    if spans.intersection(set(range(x.idx, x.idx + len(x.text)))):
      if not started:
        left, started = x.idx, True
      right = x.idx + len(x.text)
    elif started:
      ents.append((left, right, label))
      started = False
  if started:
    ents.append((left, right, label))
  return ents


def read_datafile(filename):
  """Reads csv file with python span list and text."""
  data = []
  with open(filename) as csvfile:
    reader = csv.DictReader(csvfile)
    count = 0
    for row in reader:
      fixed = fix_spans.fix_spans(
          ast.literal_eval(row['spans']), row['text'])
      data.append((fixed, row['text']))
  return data

# Defined function to return F1, precision and recall based on 
# https://competitions.codalab.org/competitions/25623#learn_the_details-evaluation
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


def main():
  """Train and eval a spacy named entity tagger for toxic spans."""
  # Read data for train set
  print('loading training data')
  train = read_datafile('../data/tsd_train.csv')

  # Read trial data for validation set
  validation = read_datafile('../data/tsd_trial.csv')

  # Read data for test set
  print('loading test data')
  test = read_datafile('../data/tsd_test.csv')

  # Convert training data to Spacy Entities
  nlp = spacy.load("en_core_web_sm")
  print('preparing training data')
  training_data = []
  for n, (spans, text) in enumerate(train):
    doc = nlp(text)
    ents = spans_to_ents(doc, set(spans), 'TOXIC')
    training_data.append((doc.text, {'entities': ents}))

  toxic_tagging = spacy.blank('en')
  toxic_tagging.vocab.strings.add('TOXIC')
  ner = nlp.create_pipe("ner")
  toxic_tagging.add_pipe(ner, last=True)
  ner.add_label('TOXIC')

  pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
  unaffected_pipes = [
      pipe for pipe in toxic_tagging.pipe_names
      if pipe not in pipe_exceptions]


  print('Training!')
  with toxic_tagging.disable_pipes(*unaffected_pipes):
    
    toxic_tagging.begin_training()
    for iteration in range(30):
      random.shuffle(training_data)
      losses = {}
      batches = spacy.util.minibatch(
          training_data, size=spacy.util.compounding(
              4.0, 32.0, 1.001))
      for batch in batches:
        texts, annotations = zip(*batch)
        toxic_tagging.update(texts, annotations, drop=0.5, losses=losses)
      print("Losses", losses)


  # Define helper function for evaluating datasets
  def evaluate(dateset):
    precision_recall_f1_scores = []
    for spans, text in dateset:
      pred_spans = []
      doc = toxic_tagging(text)
      for ent in doc.ents:
        pred_spans.extend(range(ent.start_char, ent.start_char + len(ent.text)))
      
      # score = semeval2021.f1(pred_spans, spans)
      precision_recall_f1_scores.append(per_post_precision_recall_f1(pred_spans, spans))

    # compute average precision, recall and f1 score of all posts
    return np.array(precision_recall_f1_scores).mean(axis=0)

  # Evaluate on dev and test sets
  print('Evaluation:')
  eval_precision, eval_recall, eval_f1 = evaluate(validation)
  test_precision, test_recall, test_f1 = evaluate(test)
  
  print(f'Dev set: Precision = {eval_precision}, Recall = {eval_recall}, F1 = {eval_f1}')
  print(f'Test set: Precision = {test_precision}, Recall = {test_recall}, F1 = {test_f1}')
  
 

if __name__ == '__main__':
  start = time.time()
  main()
  end = time.time()
  print(f"Time: {(end-start)/60} mins")
