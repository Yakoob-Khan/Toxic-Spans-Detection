import nltk

# For tokenizing sentences
nltk.download('punkt')
sentence_tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')


def toxic_character_offsets_with_thresholding(post_num, tokens, offset_mapping, prediction, val_sentences_info, prediction_score, threshold):
  toxic_offsets = []
  scores = []
  n = len(tokens)
  i = 1           # start from 1 as 0th token is [CLS]
  while i < n:
    # stop looping after processing all post tokens
    if tokens[i] == '[SEP]':
      break

    cur_toxic = []
    # if previous token is also predicted toxic, then toxic phrase found
    if len(toxic_offsets) > 0 and toxic_offsets[-1] == offset_mapping[i-1][1] - 1:
      cur_toxic.extend([index for index in range(offset_mapping[i-1][1], offset_mapping[i][0])])
    
    # add the characters offsets of this head BPE
    cur_toxic.extend([index for index in range(offset_mapping[i][0], offset_mapping[i][1])])
    cur_score = [(tokens[i], prediction_score[i].max())]
    cur_labels = [prediction[i]]
    
    # process all sub-tokens of the current head BPE
    i += 1
    while i < n and '##' in tokens[i]:
      cur_toxic.extend([index for index in range(offset_mapping[i][0], offset_mapping[i][1])])
      cur_score.append((tokens[i], prediction_score[i].max()))
      cur_labels.append(prediction[i])
      i += 1
    
    # word is predicted toxic if any sub-token is predicted toxic by model
    prediction_label = True if max(cur_labels) == 1 else False
    # prediction_label = True if min(cur_labels) == 1 else False

    
    # include cur_toxic offsets if any of the sub-token confidence score is greater than threshold
    confidence_values = [score for _, score in cur_score]
    passed_threshold = True if max(confidence_values) >= threshold else False
    # passed_threshold = True if min(confidence_values) >= threshold else False

    # include to global toxic offsets list only if both predicted label and threshold criteria passes
    if prediction_label and passed_threshold:
      toxic_offsets.extend(cur_toxic)
      scores.extend(cur_score)
  

  return toxic_offsets, scores


def toxic_character_offsets_with_late_fusion(post_num, text, tokens, offset_mapping, prediction, val_sentences_info, prediction_score, threshold):
  # retrieve the sentence classifications for this post
  sentence_classifications = val_sentences_info[text]
  # split the text into sentences
  sentence_spans = sentence_tokenizer.span_tokenize(text)

  toxic_offsets = []
  scores = []
  n = len(tokens)
  i = 1           # start from 1 as 0th token is [CLS]
  while i < n:
    # stop looping after processing all post tokens
    if tokens[i] == '[SEP]':
      break

    cur_toxic = []
    # if previous token is also predicted toxic, then toxic phrase found
    if len(toxic_offsets) > 0 and toxic_offsets[-1] == offset_mapping[i-1][1] - 1:
      cur_toxic.extend([index for index in range(offset_mapping[i-1][1], offset_mapping[i][0])])
    
    # add the characters offsets of this head BPE
    cur_toxic.extend([index for index in range(offset_mapping[i][0], offset_mapping[i][1])])
    cur_score = [(tokens[i], prediction_score[i].max())]
    cur_labels = [prediction[i]]
    
    # process all sub-tokens of the current head BPE
    i += 1
    while i < n and '##' in tokens[i]:
      cur_toxic.extend([index for index in range(offset_mapping[i][0], offset_mapping[i][1])])
      cur_score.append((tokens[i], prediction_score[i].max()))
      cur_labels.append(prediction[i])
      i += 1
    
    # word is predicted toxic if any sub-token is predicted toxic by model
    prediction_label = True if max(cur_labels) == 1 else False
    # prediction_label = True if min(cur_labels) == 1 else False

    
    # include cur_toxic offsets if any of the sub-token confidence score is greater than threshold
    confidence_values = [score for _, score in cur_score]
    passed_threshold = True if max(confidence_values) >= threshold else False
    # passed_threshold = True if min(confidence_values) >= threshold else False

    # ensure that at least one token is located in the toxic sentence
    toxic_sentence = False
    for idx in cur_toxic:
      for start_sen, end_sen in sentence_spans:
        if start_sen <= idx <= end_sen and sentence_classifications[f'({start_sen}, {end_sen})']['Pred'] == 1:
          toxic_sentence = True
          break

  
    # include to global toxic offsets list only if both predicted label and threshold criteria passes
    if prediction_label and passed_threshold and toxic_sentence:
      toxic_offsets.extend(cur_toxic)
      scores.extend(cur_score)
  

  return toxic_offsets, scores


def mt_dnn_post_character_offsets(post_num, labels, tokens, offsets):
  toxic_offsets = []
  n = len(tokens)
  i = 1           # start from 1 as 0th token is [CLS]
  while i < n:
    # stop looping after processing all post tokens
    if tokens[i] == '[SEP]':
      break

    cur_toxic = []
    # if previous token is also predicted toxic, then toxic phrase found
    if len(toxic_offsets) > 0 and toxic_offsets[-1] == offsets[i-1][1] - 1:
      cur_toxic.extend([index for index in range(offsets[i-1][1], offsets[i][0])])

    cur_toxic.extend([index for index in range(offsets[i][0], offsets[i][1])])
    cur_labels = [labels[i]]

    # process all sub-tokens of the current head BPE
    i += 1
    while i < n and '##' in tokens[i]:
      cur_toxic.extend([index for index in range(offsets[i][0], offsets[i][1])])
      cur_labels.append(labels[i])
      i += 1

    prediction_label = True if max(cur_labels) == 1 else False

    if prediction_label:
      toxic_offsets.extend(cur_toxic)

  return toxic_offsets

def character_offsets_with_thresholding(val_text_encodings, val_offset_mapping, predictions, val_sentences_info, prediction_scores, threshold=-float('inf')):
  return [toxic_character_offsets_with_thresholding(i, val_text_encodings[i].tokens, offset_mapping, prediction, val_sentences_info, prediction_scores[i], threshold) for i, (offset_mapping, prediction) in enumerate(zip(val_offset_mapping, predictions))]

def character_offsets_with_late_fusion(texts, val_text_encodings, val_offset_mapping, predictions, val_sentences_info, prediction_scores, threshold):
  return [toxic_character_offsets_with_late_fusion(i, texts[i], val_text_encodings[i].tokens, offset_mapping, prediction, val_sentences_info, prediction_scores[i], threshold) for i, (offset_mapping, prediction) in enumerate(zip(val_offset_mapping, predictions))]

def mt_dnn_character_offsets(tokens, predictions, offsets):
  return [mt_dnn_post_character_offsets(i, labels, tokens[i], offsets[i]) for i, labels in enumerate(predictions)]
