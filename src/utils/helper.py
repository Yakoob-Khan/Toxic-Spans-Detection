# Code obtained from https://github.com/ipavlopoulos/toxic_spans/blob/master/evaluation/fix_spans.py
import string
import itertools

def _contiguous_ranges(span_list):
    """Extracts continguous runs [1, 2, 3, 5, 6, 7] -> [(1,3), (5,7)].
       Credit: https://github.com/ipavlopoulos/toxic_spans/blob/master/evaluation/fix_spans.py
    """
    output = []
    for _, span in itertools.groupby(
        enumerate(span_list), lambda p: p[1] - p[0]):
        span = list(span)
        output.append((span[0][1], span[-1][1]))
    return output

SPECIAL_CHARACTERS = string.whitespace
def fix_spans(spans, text, special_characters=SPECIAL_CHARACTERS):
    """
      Applies minor edits to trim spans and remove singletons.
      Credit: https://github.com/ipavlopoulos/toxic_spans/blob/master/evaluation/fix_spans.py
    """
    cleaned = []
    for begin, end in _contiguous_ranges(spans):
        while text[begin] in special_characters and begin < end:
            begin += 1
        while text[end] in special_characters and begin < end:
            end -= 1
        if end - begin > 1:
            cleaned.extend(range(begin, end + 1))
    return cleaned


