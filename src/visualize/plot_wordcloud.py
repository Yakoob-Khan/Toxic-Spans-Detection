import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import itertools

from wordcloud import WordCloud
from ast import literal_eval

plt.figure(dpi=1000)
# plt.rc('figure', titlesize=40)  # fontsize of the figure title

def contiguous_ranges(span_list):
    """Extracts continguous runs [1, 2, 3, 5, 6, 7] -> [(1,3), (5,7)].
       Credit: https://github.com/ipavlopoulos/toxic_spans/blob/master/evaluation/fix_spans.py
    """
    output = []
    for _, span in itertools.groupby(
        enumerate(span_list), lambda p: p[1] - p[0]):
        span = list(span)
        output.append((span[0][1], span[-1][1]))
    return output

def plot_wordcloud(data_dir, name):
  dataset = pd.read_csv(data_dir)
  dataset["spans"] = dataset.spans.apply(literal_eval)
  spans = [contiguous_ranges(span) for span in dataset["spans"]]

  toxic_texts = []
  for i, text in enumerate(dataset['text']):
    toxic_words = ' '.join([text[a: b+1] for a, b in spans[i]])
    toxic_texts.append(toxic_words)



  # Referenced: https://www.datacamp.com/community/tutorials/wordcloud-python
  text = " ".join(text for text in toxic_texts)
  count = WordCloud().process_text(text)

  # Create and generate a word cloud image:
  wordcloud = WordCloud(max_words=100, background_color="white", width=1000, height=500).fit_words(count)

  # Display the generated image:
  fig = plt.figure(figsize=(20,10))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  # plt.title(f'{name} Set')
  fig.suptitle(f'{name} Set', fontsize=60)
  plt.savefig(f"output/wordcloud_{name}.pdf")


# plot_wordcloud('../data/tsd_train.csv', 'Train')
# plot_wordcloud('../data/tsd_trial.csv', 'Dev')
plot_wordcloud('../data/tsd_test.csv', 'Test')