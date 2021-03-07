import nltk
nltk.download('punkt')
nltk.download('wordnet')

import random
import pandas as pd 
import string
import ast
import argparse

from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import wordnet 
from pre_process.load_dataset import load_dataset
from utils.helper import _contiguous_ranges

parser = argparse.ArgumentParser()
parser.add_argument("--synonym_replacement", type=str, default='True', help="Synonym Replacement")
parser.add_argument("--random_insertion", type=str, default='False', help="Random Insertion")
parser.add_argument("--random_deletion", type=str, default='False', help="Random Deletion")
parser.add_argument("--random_swap", type=str, default='False', help="Random Swap")
parser.add_argument("--alpha", type=float, default=0.8, help="Alpha hyper-parameter to determine degree of augmentation")
parser.add_argument("--train_dir", type=str, default='../data/tsd_train.csv', help="File path to train directory")
parser.add_argument("--output", type=str, default='./augmented_data/tsd_eda.csv', help="Name of augmented dataset")

args = parser.parse_args()

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']


random.seed(42)

# Load the training set
training_texts, training_spans = load_dataset(args.train_dir)

# Use the NLTK tokenizer
tokenizer = WhitespaceTokenizer()

# Get the tokenized posts and their labels
tokenized_posts, gold_labels = [], []

# Tokenize the texts and get the corresponding labels
for text, span in zip(training_texts, training_spans):
  # Tokenize post
  tokens_offsets = tokenizer.span_tokenize(text)
  gold_offset_chars = set(span)

  # Determine label for each token
  tokenized_post, post_labels = [], []
  for i, j in tokens_offsets:
    # check if this token label is toxic
    toxic_label = 0
    for k in range(i, j):
      if k in gold_offset_chars:
        toxic_label = 1
        break
    
    # remove punctuation from the token
    tokenized_post.append(text[i:j].translate(str.maketrans('', '', string.punctuation)))
    post_labels.append(toxic_label)
  
  # add the tokenized post and its labels to training set
  tokenized_posts.append(tokenized_post)
  gold_labels.append(post_labels)


# EDA code modified from: 
# https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py 
def synonym_replacement(words, labels, n):
  new_words = words.copy()
  new_labels = labels.copy()
  random_word_list = list(set([word for word in words if word not in stop_words]))
  random.shuffle(random_word_list)

  num_replaced = 0
  for random_word in random_word_list:
    synonyms = get_synonyms(random_word)
    if len(synonyms) >= 1:
      synonym = random.choice(list(synonyms))
      newer_words, newer_labels = [], []
      for i, word in enumerate(new_words):
        if word == random_word:
          newer_words.extend(synonym.strip().split())
          newer_labels.extend([new_labels[i]] * len(synonym.strip().split()))
        else:
          newer_words.append(word)
          newer_labels.append(new_labels[i])

      new_words = newer_words
      new_labels = newer_labels
      num_replaced += 1
    
    if num_replaced >= n:
      break
  
  return new_words, new_labels


def get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym) 
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)


def random_deletion(words, labels, p):
  # if there's only one word, don't delete it
  if len(words) == 1:
    return words, labels

  # randomly delete words with probability p
  new_words = []
  new_labels = []
  for word, label in zip(words, labels):
    r = random.uniform(0, 1)
    if r > p:
      new_words.append(word)
      new_labels.append(label)
  
  # if end up deleting all words, just return a random word
  if len(new_words) == 0:
    rand_int = random.randint(0, len(words)-1)
    return [words[rand_int]], [labels[rand_int]]
  
  return new_words, new_labels


def random_swap(words, labels, n):
  new_words = words.copy()
  new_labels = labels.copy()
  for _ in range(n):
    new_words, new_labels = swap_word(new_words, new_labels)
  return new_words, new_labels


def swap_word(new_words, new_labels):
  random_idx_1 = random.randint(0, len(new_words)-1)
  random_idx_2 = random_idx_1
  counter = 0
  while random_idx_2 == random_idx_1:
    random_idx_2 = random.randint(0, len(new_words)-1)
    counter += 1
    if counter > 3:
      return new_words, new_labels
  
  new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
  new_labels[random_idx_1], new_labels[random_idx_2] = new_labels[random_idx_2], new_labels[random_idx_1]
  
  return new_words, new_labels


def random_insertion(words, labels, n):
  new_words = words.copy()
  new_labels = labels.copy()
  for _ in range(n):
    add_word(new_words, new_labels)
  
  return new_words, new_labels

def add_word(new_words, new_labels):
  synonyms = []
  counter = 0
  while len(synonyms) < 1:
    r = random.randint(0, len(new_words)-1)
    random_word, label = new_words[r], new_labels[r]
    synonyms = get_synonyms(random_word)
    counter += 1
    if counter >= 10:
      return
  
  random_synonym = synonyms[0]
  random_idx = random.randint(0, len(new_words)-1)
  new_words.insert(random_idx, random_synonym)
  new_labels.insert(random_idx, label)


alpha_sr = args.alpha
alpha_rs = args.alpha
alpha_ri = args.alpha
p_rd = args.alpha

augmented_text, augmented_labels = [], []
for tokens, labels in zip(tokenized_posts, gold_labels):
  num_words = len(tokens)

  if ast.literal_eval(args.synonym_replacement):
    # Synonym Replacement
    n_sr = max(1, int(alpha_sr * num_words))
    synonym_tokens, synonym_labels = synonym_replacement(tokens, labels, n_sr)
    augmented_text.append(synonym_tokens)
    augmented_labels.append(synonym_labels)

  if ast.literal_eval(args.random_deletion):
    # Random Deletion
    random_deleted_tokens, random_deleted_labels = random_deletion(tokens, labels, p_rd)
    augmented_text.append(random_deleted_tokens)
    augmented_labels.append(random_deleted_labels)

  if ast.literal_eval(args.random_swap):
    # Random Swap
    n_rs = max(1, int(alpha_rs * num_words))
    random_swap_tokens, random_swap_labels = random_swap(tokens, labels, n_rs)
    augmented_text.append(random_swap_tokens)
    augmented_labels.append(random_swap_labels)

  if ast.literal_eval(args.random_insertion):
    # Random Insertion
    n_ri = max(1, int(alpha_ri * num_words))
    random_inserted_tokens, random_inserted_labels = random_insertion(tokens, labels, n_ri)
    augmented_text.append(random_inserted_tokens)
    augmented_labels.append(random_inserted_labels)
  
  
# Convert the tokens and labels to toxic spans data format
toxic_spans_data = {'spans': [span for span in training_spans], 'text': [text for text in training_texts]}
# num = 0
for tokens, labels in zip(augmented_text, augmented_labels):
  # combine the tokens with spaces in between and get toxic span offsets
  post, span = '', []
  for i, token in enumerate(tokens):
    start = len(post)
    post += token + " "
    if labels[i] == 1:
      span.extend([index for index in range(start, len(post))])

  # adjust span offsets to remove unwanted spaces
  adjusted_post = post
  ajusted_span = []
  for a, b in _contiguous_ranges(span):
    if adjusted_post[b] == ' ':
      ajusted_span.extend([k for k in range(a, b)])
    else:
      ajusted_span.extend([k for k in range(a, b+1)])


  toxic_spans_data['spans'].append(ajusted_span)
  toxic_spans_data['text'].append(adjusted_post)

df = pd.DataFrame(toxic_spans_data)
df = df.sample(frac=1)

df.to_csv(args.output, index=False)
print(f'> Written {df.shape[0]} posts to {args.output} file!')

    
