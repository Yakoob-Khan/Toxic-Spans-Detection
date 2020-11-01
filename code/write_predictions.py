def toxic_strings(text, span):
    toxic_spans = []
    cur = [None, None]
    for offset in span:
        if cur[0] is None:
            cur = [offset, offset]
        elif offset == cur[1] + 1:
            cur[1] = offset
        else:
            toxic_spans.append(cur)
            cur = [offset, offset]

    if cur[0] and cur[1]:
        toxic_spans.append(cur)

    return [text[i: j+1] for i, j in toxic_spans]


def write_toxic_strings(file_path, val_texts, gold_char_offsets, toxic_char_preds):
  """
  Each line is comma separated lists
  [original text],[gold_toxic],[pred_toxic]
  """
  f = open(file_path, "w")

  for text, gold, pred in zip(val_texts, gold_char_offsets, toxic_char_preds):
    gold_toxic = toxic_strings(text, gold)
    pred_toxic = toxic_strings(text, pred)
    f.write(f"{[text]},{str(gold_toxic)},{str(pred_toxic)} \n")

  f.close()

